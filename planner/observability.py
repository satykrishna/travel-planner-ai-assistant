"""Optional LangSmith tracing with redacted inputs and outputs."""

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from functools import lru_cache

from langsmith import Client, trace, tracing_context
from urllib3.util.retry import Retry

from .guardrails import public_data, redact_pii

logger = logging.getLogger(__name__)
_METADATA_KEYS = {
    "request_id", "environment", "revision", "outcome", "model_calls", "search_calls",
    "ls_provider", "ls_model_name", "ls_model_type", "ls_temperature", "ls_max_tokens",
    "langgraph_node", "langgraph_step",
    "usage_metadata",
}
_ENDPOINTS = {"https://api.smith.langchain.com", "https://eu.api.smith.langchain.com"}
_PRIVATE_FIELDS = {"authorization", "api_key", "apikey", "x_api_key", "access_key",
                   "access_token", "password", "cookie", "set_cookie", "token",
                   "reasoning_content", "hidden_reasoning"}
_TRACE_CREDENTIAL = re.compile(r"\b(?:Bearer\s+[A-Za-z0-9._~+/-]+=*|lsv2_(?:pt|sk)_[A-Za-z0-9_-]{16,})", re.I)


def redact_trace_data(value):
    """Retain useful trace content while masking credentials and patterned PII."""
    if isinstance(value, str):
        return _TRACE_CREDENTIAL.sub("[redacted]", redact_pii(public_data(value)))
    if isinstance(value, dict):
        return {str(public_data(key)): ("[redacted]" if str(key).lower().replace("-", "_") in _PRIVATE_FIELDS
                                       else redact_trace_data(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_trace_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return redact_trace_data(value.model_dump())
    if value is None or type(value) in (bool, int, float):
        return value
    return redact_trace_data(str(value))


def safe_run_operations(operations):
    """Final privacy boundary before trace batches leave the server."""
    result = []
    for operation in operations:
        run = dict(operation)
        if "inputs" in run:
            run["inputs"] = redact_trace_data(run["inputs"])
        if "outputs" in run:
            run["outputs"] = redact_trace_data(run["outputs"])
        for key in ("events", "attachments", "serialized"):
            run.pop(key, None)
        if run.get("error"):
            run["error"] = redact_trace_data(str(run["error"]).splitlines()[0][:1000])
        extra = run.get("extra") or {}
        metadata = extra.get("metadata") or {}
        safe_metadata = {key: value for key, value in metadata.items() if key in _METADATA_KEYS}
        if "usage_metadata" in safe_metadata:
            safe_metadata["usage_metadata"] = _numeric_usage(safe_metadata["usage_metadata"])
        run["extra"] = {"metadata": safe_metadata}
        if "usage_metadata" in run:
            run["usage_metadata"] = _numeric_usage(run["usage_metadata"])
        # usage_metadata is kept intact for token counts and cost attribution.
        # All remaining strings still get credential and patterned PII masking.
        run = _redact_values(public_data(run))
        result.append(run)
    return result


def _numeric_usage(value):
    if not isinstance(value, dict):
        return {}
    return {key: (_numeric_usage(item) if isinstance(item, dict) else item)
            for key, item in value.items()
            if key in {"input_tokens", "output_tokens", "total_tokens", "input_token_details",
                       "output_token_details", "input_cost", "output_cost", "total_cost",
                       "cache_read", "cache_creation", "reasoning", "audio", "text", "image"}
            and (isinstance(item, dict) or type(item) in (int, float))}


def _redact_values(value):
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {key: _redact_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_values(item) for item in value]
    return value


@lru_cache(maxsize=1)
def get_client():
    key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not key or os.getenv("LANGSMITH_TRACING", "true").lower() != "true":
        return None
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    if endpoint not in _ENDPOINTS:
        logger.warning("LangSmith endpoint is unsupported; tracing disabled")
        return None
    try:
        return Client(
            api_url=endpoint, api_key=key,
            workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID") or None,
            hide_inputs=redact_trace_data, hide_outputs=redact_trace_data, omit_traced_runtime_info=True,
            process_buffered_run_ops=safe_run_operations,
            run_ops_buffer_size=64, run_ops_buffer_timeout_ms=100,
            timeout_ms=1000, retry_config=Retry(total=0),
        )
    except Exception:
        logger.warning("LangSmith initialization unavailable; planning remains enabled")
        return None


@dataclass
class Observation:
    client: Client | None = None
    run: object = None
    outcome: str = "cancelled"
    result: dict = field(default_factory=dict)

    def record(self, event):
        payload = json.loads(event)
        kind = payload.get("event")
        if kind == "result":
            self.outcome = "success"
            self.result = {"answer": payload.get("data", {}).get("answer", "")}
        elif kind == "error":
            self.outcome = "blocked_or_failed"
            self.result = {"error": payload.get("data", {}).get("message", "Planning could not complete.")}

    def finish(self, budget):
        if self.run is not None:
            try:
                self.run.end(
                    outputs=redact_trace_data(self.result or {"outcome": self.outcome}),
                    error=None if self.outcome == "success" else redact_trace_data(self.result.get("error", "Request did not complete successfully.")),
                    metadata={"outcome": self.outcome, **budget},
                )
            except Exception:
                logger.warning("LangSmith outcome recording unavailable")

    async def flush(self):
        if self.client is None:
            return
        try:
            # Vercel can freeze the worker immediately after the response.
            await asyncio.wait_for(asyncio.to_thread(self.client.flush, timeout=2), timeout=3)
        except Exception:
            logger.warning("LangSmith trace flush unavailable")


@contextmanager
def observe_request(query=""):
    client = get_client()
    project = os.getenv("LANGSMITH_PROJECT", "travel-planner")
    metadata = {"request_id": uuid.uuid4().hex,
                "environment": os.getenv("VERCEL_ENV", "development"),
                "revision": os.getenv("VERCEL_GIT_COMMIT_SHA", "local")}
    # Disables incidental global tracing when this feature has no configured key.
    # Request headers are deliberately never used as trace parents or metadata.
    stack = ExitStack()
    observation = Observation()
    try:
        stack.enter_context(tracing_context(enabled=client is not None, client=client,
                                            project_name=project, parent=False))
        if client is not None:
            run = stack.enter_context(trace("travel_plan_request", run_type="chain", inputs={"query": redact_trace_data(query)},
                                            client=client, project_name=project, metadata=metadata,
                                            tags=["travel-planner", metadata["environment"]]))
            observation = Observation(client=client, run=run)
    except Exception:
        with suppress(Exception):
            stack.close()
        stack = ExitStack()
        stack.enter_context(tracing_context(enabled=False))
        logger.warning("LangSmith request tracing unavailable; continuing without telemetry")
    try:
        yield observation
    finally:
        with suppress(Exception):
            stack.close()
