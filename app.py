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
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=False)
app = FastAPI(title="Travel Planner AI Assistant", docs_url=None, redoc_url=None)
logger = logging.getLogger(__name__)
REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY")


class PlanRequest(BaseModel):
    query: str = Field(min_length=5, max_length=6000)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value):
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Describe your trip in at least 5 characters.")
        return value


def configuration():
    missing = any(not os.getenv(key, "").strip() for key in REQUIRED_KEYS)
    access_required = bool(os.getenv("APP_ACCESS_TOKEN"))
    # A deployed, paid search endpoint must not become open by accident.
    production_unprotected = bool(os.getenv("VERCEL")) and not access_required
    return {"ready": not missing and not production_unprotected, "access_required": access_required}


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
    return json.dumps({"event": event, "data": data}, ensure_ascii=False, default=str) + "\n"


def message_record(message):
    text = message.content
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, default=str)
    return {
        "type": type(message).__name__,
        "content": text[:16000], "truncated": len(text) > 16000,
        "tool_calls": getattr(message, "tool_calls", []),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "name": getattr(message, "name", None),
    }


async def plan_events(request, query):
    from planner.dates import check_flight_hotel_dates
    from planner.workflow import get_workflow, initial_state

    pending = None
    stream = None
    final_state = None
    try:
        yield encode("progress", {"node": "starting", "message": "Preparing your travel team"})
        graph = get_workflow()
        stream = graph.astream(
            initial_state(query), config={"recursion_limit": 60},
            stream_mode=["updates", "values"], subgraphs=True,
        )
        async with asyncio.timeout(270):
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
                        yield encode("routing", {"tasks": update.get("tasks", [])})
                    if node in {"search_agent_node", "search_tool_node", "travel_scout_agent_node", "travel_scout_tool_node", "synthesizer_node"}:
                        yield encode("progress", {"node": node})
                        for key in ("search_agent_messages", "travel_scout_agent_msgs"):
                            for message in update.get(key, []):
                                yield encode("trace", {"agent": "search_agent" if key == "search_agent_messages" else "travel_scout_agent", **message_record(message)})
        if not final_state or not final_state.get("synthesized_answer"):
            raise RuntimeError("Workflow ended without an answer")
        checks = check_flight_hotel_dates(final_state)
        yield encode("result", {
            "answer": final_state["synthesized_answer"],
            "tasks": final_state.get("tasks", []),
            "findings": final_state.get("agent_results", []),
            "date_checks": checks,
        })
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


@app.post("/api/plan")
async def plan(body: PlanRequest, request: Request, authorization: str | None = Header(default=None)):
    access_token = os.getenv("APP_ACCESS_TOKEN", "")
    if access_token and not hmac.compare_digest(authorization or "", "Bearer " + access_token):
        raise HTTPException(401, "Enter the app access key to create a plan.")
    if not configuration()["ready"]:
        raise HTTPException(503, "Server setup is incomplete. Configure provider keys and an APP_ACCESS_TOKEN on Vercel.")
    return StreamingResponse(
        plan_events(request, body.query), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff"},
    )
