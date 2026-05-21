"""
Observability: structured, machine-readable traces of every agent step.

Design note: this writes one JSON line per event (not per conversation) to a
JSONL file, because the eval harness and any future dashboard need to reason
about individual steps (which tool was called, how long the LLM call took,
whether a guardrail fired) rather than parsing a blob of chat transcript.
JSONL also means a trace file is streaming-appendable and grep-able without
loading the whole run into memory.

Every event is also emitted as an OpenTelemetry span, so this can plug into
a real tracing backend without touching the agent code. By default no
exporter is attached (spans are created but go nowhere -- near-zero
overhead), which keeps eval runs and local testing quiet. Set
OTEL_CONSOLE_EXPORT=1 to print spans to stdout, or OTEL_EXPORTER_OTLP_ENDPOINT
to ship them to a real OTLP collector (Jaeger, Honeycomb, etc.) via the
standard OpenTelemetry env var.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from src.guardrails import redact_pii

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"
TRACES_DIR.mkdir(exist_ok=True)

_otel_provider = TracerProvider(resource=Resource.create({"service.name": "voltpilot"}))
if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    _otel_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
elif os.environ.get("OTEL_CONSOLE_EXPORT"):
    _otel_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
otel_trace.set_tracer_provider(_otel_provider)
_otel_tracer = otel_trace.get_tracer("voltpilot")


def _span_attrs(payload: dict[str, Any]) -> dict[str, Any]:
    """Span attributes must be primitives or arrays of primitives -- anything
    else (a nested dict, a mixed list) gets flattened to a JSON string."""
    attrs: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, bool, int, float)):
            attrs[key] = value
        elif isinstance(value, list) and all(isinstance(item, (str, bool, int, float)) for item in value):
            attrs[key] = value
        else:
            attrs[key] = json.dumps(value, default=str)
    return attrs


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
        with _otel_tracer.start_as_current_span(event_type) as span:
            span.set_attributes(_span_attrs(payload))
            span.set_attribute("session_id", self.session_id)
            if latency_ms is not None:
                span.set_attribute("latency_ms", latency_ms)
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
        """Usage: with tracer.timed('tool_call', {'name': ...}) as extra: extra['result'] = ...

        Wraps the whole timed block in a real OTel span (so span duration
        reflects actual work, not just the JSONL write) and still logs the
        same JSONL event as before."""
        start = time.time()
        extra: dict[str, Any] = {}
        with _otel_tracer.start_as_current_span(event_type) as span:
            try:
                yield extra
            finally:
                latency_ms = (time.time() - start) * 1000
                merged = {**payload, **extra}
                span.set_attributes(_span_attrs(merged))
                span.set_attribute("session_id", self.session_id)
                span.set_attribute("latency_ms", latency_ms)
                self._write(
                    TraceEvent(
                        trace_id=uuid.uuid4().hex[:12],
                        session_id=self.session_id,
                        event_type=event_type,
                        timestamp=time.time(),
                        payload=merged,
                        latency_ms=latency_ms,
                    )
                )

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
