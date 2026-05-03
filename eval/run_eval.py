"""
Evaluation harness.

Two tiers of test, deliberately kept separate because they catch different
classes of bug:

  - agent_cases: end-to-end trajectory tests. Run the full agent loop and
    assert on (a) which tools were called, in order, (b) whether expected
    grounding facts appear in the final answer, (c) whether forbidden content
    (leaked system prompt, unsafe instructions) is absent, and (d) whether a
    guardrail fired when it should have.
  - guardrail_cases: component-level unit tests against guardrails.py
    directly, using adversarial inputs (a simulated poisoned retrieval
    source, an unsafe-output string) that don't depend on what the LLM/mock
    would do -- these test the deterministic backstop in isolation.

Usage: python3 eval/run_eval.py
Exits non-zero if any case fails, so this is CI-friendly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import EVCopilotAgent
from src.guardrails import check_output_safety, redact_pii, scan_for_injection, wrap_untrusted
from src.observability import Tracer

CASES_PATH = Path(__file__).parent / "cases.json"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


def run_agent_case(case: dict) -> dict:
    tracer = Tracer(session_id=f"eval-{case['id']}")
    agent = EVCopilotAgent(tracer=tracer)
    result = agent.handle_message(case["prompt"])

    tool_sequence = [t["name"] for t in result["tool_calls"]]
    response_lower = result["response"].lower()

    failures = []

    if tool_sequence != case["expected_tool_sequence"]:
        failures.append(f"tool sequence {tool_sequence} != expected {case['expected_tool_sequence']}")

    expected_any = case.get("expected_keywords_any", [])
    if expected_any and not any(kw.lower() in response_lower for kw in expected_any):
        failures.append(f"none of expected keywords {expected_any} found in response: {result['response'][:200]!r}")

    for forbidden in case.get("forbidden_keywords", []):
        if forbidden.lower() in response_lower:
            failures.append(f"forbidden keyword {forbidden!r} found in response")

    expect_guardrail = case.get("expect_guardrail_triggered")
    if expect_guardrail:
        categories = [g["category"] for g in result["guardrail_events"]]
        if expect_guardrail not in categories:
            failures.append(f"expected guardrail category {expect_guardrail!r} to fire, got {categories}")

    return {
        "id": case["id"],
        "kind": "agent",
        "passed": not failures,
        "failures": failures,
        "response": result["response"],
        "tool_sequence": tool_sequence,
        "guardrail_events": result["guardrail_events"],
    }


def run_guardrail_case(case: dict) -> dict:
    failures = []
    t = case["type"]

    if t == "injection_scan":
        finding = scan_for_injection(case["input_text"])
        if finding.triggered != case["expect_triggered"]:
            failures.append(f"expected triggered={case['expect_triggered']}, got {finding.triggered}")

    elif t == "wrap_untrusted":
        wrapped = wrap_untrusted(case["source_label"], case["input_text"])
        if case["expect_substring"] not in wrapped:
            failures.append(f"expected substring {case['expect_substring']!r} not found in wrapped output")

    elif t == "output_safety":
        finding = check_output_safety(case["input_text"])
        if finding.triggered != case["expect_triggered"]:
            failures.append(f"expected triggered={case['expect_triggered']}, got {finding.triggered}")

    elif t == "pii_redaction":
        redacted = redact_pii(case["input_text"])
        for forbidden in case["forbidden_substrings"]:
            if forbidden in redacted:
                failures.append(f"PII substring {forbidden!r} survived redaction: {redacted!r}")

    else:
        failures.append(f"unknown guardrail case type: {t}")

    return {"id": case["id"], "kind": "guardrail", "passed": not failures, "failures": failures}


def main() -> int:
    cases = json.loads(CASES_PATH.read_text())
    results = [run_agent_case(c) for c in cases["agent_cases"]]
    results += [run_guardrail_case(c) for c in cases["guardrail_cases"]]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n{'CASE':<45} {'KIND':<10} RESULT")
    print("-" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<45} {r['kind']:<10} {status}")
        if not r["passed"]:
            for f in r["failures"]:
                print(f"    -> {f}")

    print("-" * 70)
    print(f"{passed}/{total} cases passed\n")

    REPORT_PATH.write_text(json.dumps({"passed": passed, "total": total, "results": results}, indent=2))
    print(f"Full report written to {REPORT_PATH}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
