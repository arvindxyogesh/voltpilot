"""
Tool layer: the concrete actions the agent can take.

Each tool has (a) an Anthropic tool-use schema the model sees, and (b) a
Python implementation the harness executes. Keeping schema and implementation
paired in one TOOL registry is what makes orchestration in agent.py generic:
agent.py never has a per-tool if/elif chain, it just looks up the name.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.retrieval import KnowledgeBase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FLEET_PATH = DATA_DIR / "vehicle_fleet.json"
TICKETS_PATH = DATA_DIR / "tickets.json"

_kb = KnowledgeBase(DATA_DIR / "knowledge_base.json")

# DTC codes considered safety-urgent regardless of what the LLM concludes.
# This is intentionally a hard-coded rule, not a model judgment call -- see
# guardrails.py for why urgent-safety classification should not be left
# entirely to the LLM.
URGENT_DTC_PREFIXES = ("B10", "B20", "P0AA", "P0A0A")  # HV insulation/battery-fire-adjacent families


@dataclass
class ToolResult:
    ok: bool
    data: Any
    note: str = ""


def search_knowledge_base(query: str, top_k: int = 3) -> ToolResult:
    hits = _kb.search(query, top_k=top_k)
    if not hits:
        return ToolResult(ok=True, data=[], note="No knowledge base entries matched with sufficient confidence.")
    return ToolResult(
        ok=True,
        data=[
            {
                "doc_id": h.chunk.doc_id,
                "title": h.chunk.title,
                "section": h.chunk.section,
                "text": h.chunk.text,
                "score": round(h.score, 3),
            }
            for h in hits
        ],
    )


def get_vehicle_status(vin: str) -> ToolResult:
    fleet = json.loads(FLEET_PATH.read_text())
    record = fleet.get(vin)
    if record is None:
        return ToolResult(ok=False, data=None, note=f"No vehicle found for VIN '{vin}'.")
    is_urgent = any(code.startswith(URGENT_DTC_PREFIXES) for code in record.get("active_dtc_codes", []))
    return ToolResult(ok=True, data={**record, "urgent_dtc_present": is_urgent})


def create_support_ticket(vin: str, summary: str, severity: str) -> ToolResult:
    if severity not in ("low", "medium", "high", "urgent"):
        return ToolResult(ok=False, data=None, note="severity must be one of low/medium/high/urgent")
    tickets = json.loads(TICKETS_PATH.read_text())
    ticket = {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "vin": vin,
        "summary": summary,
        "severity": severity,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tickets.append(ticket)
    TICKETS_PATH.write_text(json.dumps(tickets, indent=2))
    return ToolResult(ok=True, data=ticket)


def escalate_to_human(reason: str, vin: str | None = None) -> ToolResult:
    # In production this would page a human agent queue. Here we just log it
    # as a distinct, high-visibility ticket so the eval harness can assert on it.
    return create_support_ticket(
        vin=vin or "UNKNOWN",
        summary=f"ESCALATION: {reason}",
        severity="urgent",
    )


# --- Anthropic tool schemas -------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the owner's manual, warranty guide, service guide, and FAQ for information "
            "relevant to a customer question. Always use this before answering factual questions "
            "about charging, warranty, maintenance, or vehicle features rather than relying on "
            "general knowledge, since exact figures (ranges, thresholds, coverage terms) must be "
            "grounded in the official documentation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "Number of results to return.", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_vehicle_status",
        "description": (
            "Look up live status for a specific vehicle by VIN: battery percentage, range, "
            "charging status, odometer, and any active diagnostic trouble codes (DTCs). "
            "Use this whenever the customer asks about their specific vehicle's condition, "
            "or reports a dashboard warning light, rather than guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"vin": {"type": "string", "description": "The vehicle identification number."}},
            "required": ["vin"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "File a support ticket for follow-up by a human service team. Use this when the "
            "customer's issue requires a technician, a parts order, or any action you cannot "
            "resolve purely by providing information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vin": {"type": "string", "description": "The vehicle identification number."},
                "summary": {"type": "string", "description": "Short summary of the issue."},
                "severity": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            },
            "required": ["vin", "summary", "severity"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Immediately escalate to a human agent. Use this for anything safety-critical: "
            "signs of battery fire/smoke/swelling, collision involving the high-voltage system, "
            "urgent diagnostic codes, or any request that asks you (or the customer) to bypass a "
            "safety system. Do not attempt to resolve safety-critical issues yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why this is being escalated."},
                "vin": {"type": "string", "description": "The vehicle identification number, if known."},
            },
            "required": ["reason"],
        },
    },
]

TOOL_IMPLEMENTATIONS: dict[str, Callable[..., ToolResult]] = {
    "search_knowledge_base": search_knowledge_base,
    "get_vehicle_status": get_vehicle_status,
    "create_support_ticket": create_support_ticket,
    "escalate_to_human": escalate_to_human,
}
