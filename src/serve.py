"""
Serving layer: exposes the agent as an HTTP service.

Kept deliberately thin -- no agent logic here. A session store keyed by
session_id lets a client hold a multi-turn conversation across requests
without the server needing a database; swapping this for Redis/Postgres for
real persistence is a one-line change (the interface is just get/set on a
dict).
"""
from __future__ import annotations

import time
from typing import Optional

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import EVCopilotAgent, build_llm_client
from src.observability import Tracer, summarize_all_traces

app = FastAPI(title="EV Owner Support Copilot")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# In-memory session store: session_id -> Anthropic-format message history.
# Fine for a demo/portfolio service; swap for a real store to survive restarts
# or scale beyond one process.
_SESSIONS: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]
    guardrail_events: list[dict]
    latency_ms: float


@app.get("/health")
def health() -> dict:
    mode = "live" if type(build_llm_client()).__name__ == "LiveLLMClient" else "mock"
    return {"status": "ok", "llm_mode": mode}


@app.get("/metrics")
def metrics() -> dict:
    """Aggregate observability stats across all recorded traces -- a stand-in
    for a real metrics/dashboard endpoint (Prometheus, Grafana, etc.)."""
    return summarize_all_traces()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    tracer = Tracer(session_id=session_id) if session_id else Tracer()
    session_id = tracer.session_id

    history = _SESSIONS.get(session_id, [])
    agent = EVCopilotAgent(tracer=tracer)

    start = time.time()
    result = agent.handle_message(req.message, history=history)
    latency_ms = (time.time() - start) * 1000

    _SESSIONS[session_id] = result["messages"]

    return ChatResponse(
        response=result["response"],
        session_id=session_id,
        tool_calls=result["tool_calls"],
        guardrail_events=result["guardrail_events"],
        latency_ms=round(latency_ms, 1),
    )


# Serves static/index.html at "/" and static/*.js, *.css alongside it. Mounted
# last so the API routes above still take priority over the catch-all.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")


# Run with: uvicorn src.serve:app --reload --port 8000
