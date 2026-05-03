"""
Observability: structured, machine-readable traces of every agent step.

Design note: this writes one JSON line per event (not per conversation) to a
JSONL file, because the eval harness and any future dashboard need to reason
about individual steps (which tool was called, how long the LLM call took,
whether a guardrail fired) rather than parsing a blob of chat transcript.
JSONL also means a trace file is streaming-appendable and grep-able without
loading the whole run into memory -- the same reason production tracing
systems (OpenTelemetry spans, LangSmith runs) use one-record-per-event.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.guardrails import redact_pii

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"
TRACES_DIR.mkdir(exist_ok=True)


@dataclass
class TraceEvent:
    trace_id: str
    session_id: str
    event_type: str  # "llm_call" | "tool_call" | "guardrail" | "turn_start" | "turn_end"
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


class Tracer:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.path = TRACES_DIR / f"{self.session_id}.jsonl"

    def _write(self, event: TraceEvent) -> None:
        record = asdict(event)
        # Redact anything PII-shaped before it ever touches disk.
        record["payload"] = json.loads(redact_pii(json.dumps(record["payload"])))
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def log(self, event_type: str, payload: dict[str, Any], latency_ms: float | None = None) -> None:
        self._write(
            TraceEvent(
                trace_id=uuid.uuid4().hex[:12],
                session_id=self.session_id,
                event_type=event_type,
                timestamp=time.time(),
                payload=payload,
                latency_ms=latency_ms,
            )
        )

    @contextmanager
    def timed(self, event_type: str, payload: dict[str, Any]):
        """Usage: with tracer.timed('tool_call', {'name': ...}) as extra: extra['result'] = ..."""
        start = time.time()
        extra: dict[str, Any] = {}
        try:
            yield extra
        finally:
            latency_ms = (time.time() - start) * 1000
            self.log(event_type, {**payload, **extra}, latency_ms=latency_ms)

    def read_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def summarize_all_traces() -> dict:
    """Aggregate stats across every trace file -- used by the eval report and a dashboard."""
    events: list[dict] = []
    for path in TRACES_DIR.glob("*.jsonl"):
        events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())

    tool_calls = [e for e in events if e["event_type"] == "tool_call"]
    llm_calls = [e for e in events if e["event_type"] == "llm_call"]
    guardrail_events = [e for e in events if e["event_type"] == "guardrail"]

    tool_counts: dict[str, int] = {}
    for e in tool_calls:
        name = e["payload"].get("name", "unknown")
        tool_counts[name] = tool_counts.get(name, 0) + 1

    avg_llm_latency = (
        sum(e["latency_ms"] for e in llm_calls if e.get("latency_ms")) / len(llm_calls) if llm_calls else 0
    )

    return {
        "total_events": len(events),
        "total_sessions": len({e["session_id"] for e in events}),
        "tool_call_counts": tool_counts,
        "guardrail_triggers": len(guardrail_events),
        "avg_llm_latency_ms": round(avg_llm_latency, 1),
    }
