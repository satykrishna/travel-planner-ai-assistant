"""Vercel FastAPI entrypoint. Streams progress and a traceable final plan."""

import asyncio
import hmac
import json
import logging
import os
from contextlib import suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from urllib.parse import urlsplit

from planner.guardrails import (BudgetExceeded, GuardUnavailable, PolicyViolation, guard_input, guard_output, end_budget, public_data, start_budget, validate_untrusted)
from planner.limits import MAX_REQUEST_SECONDS, quota_ready, release, reserve

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=False)
app = FastAPI(title="Travel Planner AI Assistant", docs_url=None, redoc_url=None, openapi_url=None)
logger = logging.getLogger(__name__)
REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY")


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=5, max_length=6000)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value):
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Describe your trip in at least 5 characters.")
        return validate_untrusted(value)


def configuration():
    missing = any(not os.getenv(key, "").strip() for key in REQUIRED_KEYS)
    access_required = bool(os.getenv("APP_ACCESS_TOKEN"))
    # A deployed, paid search endpoint must not become open by accident.
    production_unprotected = bool(os.getenv("VERCEL")) and len(os.getenv("APP_ACCESS_TOKEN", "")) < 32
    return {"ready": not missing and not production_unprotected and quota_ready() and os.getenv("PLANNING_ENABLED", "true").lower() == "true", "access_required": access_required, "trace_enabled": False}


@app.get("/api/status")
def status():
    return configuration()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "public" / "index.html")


@app.get("/app.js", include_in_schema=False)
def script():
    return FileResponse(ROOT / "public" / "app.js", media_type="text/javascript")


@app.get("/styles.css", include_in_schema=False)
def stylesheet():
    return FileResponse(ROOT / "public" / "styles.css", media_type="text/css")


@app.get("/sample.json", include_in_schema=False)
def sample():
    return FileResponse(ROOT / "public" / "sample.json")


def encode(event, data):
    return json.dumps(public_data({"event": event, "data": data}), ensure_ascii=False, default=str) + "\n"


async def research_events(request, query):

    pending = None
    stream = None
    final_state = None
    try:
        from planner.workflow import get_workflow, initial_state

        yield encode("progress", {"node": "starting", "message": "Preparing your travel team"})
        async with asyncio.timeout(35):
            query = await guard_input(query)
        graph = get_workflow()
        stream = graph.astream(
            initial_state(query), config={"recursion_limit": 24},
            stream_mode=["updates", "values"], subgraphs=True,
        )
        async with asyncio.timeout(MAX_REQUEST_SECONDS):
            while True:
                if await request.is_disconnected():
                    return
                if pending is None:
                    pending = asyncio.create_task(anext(stream))
                done, _ = await asyncio.wait({pending}, timeout=10)
                if not done:
                    yield encode("heartbeat", {})
                    continue
                try:
                    namespace, mode, payload = pending.result()
                except StopAsyncIteration:
                    break
                pending = None
                if mode == "values":
                    if not namespace:
                        final_state = payload
                    continue
                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue
                    if node == "classify_user_query":
                        yield encode("routing", {"tasks": [{"target_agent": task["target_agent"], "focus": "Travel research"} for task in update.get("tasks", [])]})
                    if node in {"search_agent_node", "search_tool_node", "travel_scout_agent_node", "travel_scout_tool_node", "synthesizer_node"}:
                        yield encode("progress", {"node": node})
        if not final_state or not final_state.get("synthesized_answer"):
            raise RuntimeError("Workflow ended without an answer")
        async with asyncio.timeout(35):
            answer = await guard_output(final_state["synthesized_answer"])
        yield encode("result", {
            "answer": answer,
            "tasks": [{"target_agent": task["target_agent"], "focus": "Travel research"} for task in final_state.get("tasks", [])],
        })
    except (PolicyViolation, BudgetExceeded, GuardUnavailable) as error:
        yield encode("error", {"message": str(error)})
    except TimeoutError:
        yield encode("error", {"message": "This request reached the time limit. Try fewer destinations or a more focused search."})
    except Exception as error:
        logger.warning("Planner request failed: %s", type(error).__name__)
        yield encode("error", {"message": "The planner could not finish. Check the server's provider credentials and quotas, then try again."})
    finally:
        if pending and not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pending
        if stream:
            with suppress(Exception):
                await stream.aclose()


async def plan_events(request, query, lease):
    budget_token = start_budget()
    events = research_events(request, query)
    try:
        async with asyncio.timeout(MAX_REQUEST_SECONDS):
            async for event in events:
                yield event
    except TimeoutError:
        yield encode("error", {"message": "This plan reached its time limit. Try a shorter, more focused trip."})
    finally:
        await events.aclose()
        end_budget(budget_token)
        await release(lease)


@app.post("/api/plan")
async def plan(request: Request, authorization: str | None = Header(default=None)):
    access_token = os.getenv("APP_ACCESS_TOKEN", "")
    if access_token and not hmac.compare_digest((authorization or "").encode(), ("Bearer " + access_token).encode()):
        raise HTTPException(401, "Enter the app access key to create a plan.")
    if not configuration()["ready"]:
        raise HTTPException(503, "Live planning is paused while the host configures access and usage protection.")
    origin = request.headers.get("origin")
    if (request.headers.get("sec-fetch-site") == "cross-site" or
            (origin and urlsplit(origin).netloc != request.url.netloc)):
        raise HTTPException(403, "Please create your plan from this website.")
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise HTTPException(415, "Send your trip as JSON.")
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > 24576:
            raise HTTPException(413, "Your trip request is too large.")
        raw.extend(chunk)
    try:
        body = PlanRequest.model_validate_json(raw)
    except ValidationError:
        raise HTTPException(422, "Use a travel request of 5–6000 characters without credentials, code execution, or instruction overrides.") from None
    lease = await reserve()
    return StreamingResponse(
        plan_events(request, body.query, lease), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff"},
    )
