# EV Owner Support Copilot

VoltPilot is a customer-support agent for an EV company. It answers owner questions using a
knowledge base, checks live vehicle status, files tickets, and knows when to refuse and hand
off to a human (anything touching the high-voltage battery, for example).

It's a side project I built to get real, hands-on practice with the pieces that make up an
agentic system in production: tool-calling, retrieval, guardrails, evals, and serving. Small
enough to hold in your head, but every piece is real and testable.

## Architecture

```
   ┌───────────────────────┐   fetch('/chat')   ┌──────────────────────┐
   │  static/ (browser UI)  │ ─────────────────> │  FastAPI (src/serve) │
   │  index.html/app.js/css │ <───────────────── │  /chat /health /metrics
   └───────────────────────┘   JSON response     └──────────┬───────────┘
                                                              v
                    ┌─────────────────────┐
   user message --> │   EVCopilotAgent     │ --> response
                    │   (src/agent.py)     │
                    └──────────┬───────────┘
                               │ tool-calling loop (ReAct-style)
              ┌────────────────┼─────────────────┐
              v                v                  v
     ┌────────────────┐ ┌─────────────┐  ┌──────────────────┐
     │ search_knowledge│ │get_vehicle_ │  │ create_ticket /   │
     │ _base           │ │status       │  │ escalate_to_human │
     │ (src/retrieval) │ │(mock fleet) │  │ (mock ticket DB)  │
     └────────────────┘ └─────────────┘  └──────────────────┘
              │
              v
     every tool call + LLM call + guardrail trigger
     is written to a structured trace (src/observability.py)
     and screened by src/guardrails.py before/after use
```

`static/` is a plain HTML/CSS/JS chat UI, no build step. FastAPI mounts it directly
(`src/serve.py`), so `uvicorn src.serve:app` serves both the UI and the API from one process.
You get the conversation, the session id, and a trace panel showing latency/tool
calls/guardrail hits for each turn.

The agent loop itself (`EVCopilotAgent.handle_message`) is hand-rolled against the Anthropic
Messages API instead of LangGraph/AutoGen/CrewAI. Writing it by hand means the actual failure
modes — a tool that doesn't get called, a tool result you need to treat as untrusted, deciding
when to stop looping — are visible in the code instead of buried inside a framework.

## Where things live

| Area | File |
|---|---|
| Agent loop / orchestration | `src/agent.py` — tool-calling loop, system prompt, control flow (including chaining a second tool call when a status lookup turns up something urgent) |
| Retrieval | `src/retrieval.py` — pluggable `Embedder` interface (TF-IDF by default) over `data/knowledge_base.json` |
| Tools / data | `src/tools.py` — mock fleet telemetry and ticket store; where a real CRM/telematics API would plug in |
| Guardrails | `src/guardrails.py` — prompt-injection detection, untrusted-content tagging, unsafe-output backstop, PII redaction |
| Evaluation | `eval/` — 15-case suite: agent trajectories + guardrail unit tests |
| Serving | `src/serve.py` — FastAPI (`/chat`, `/health`, `/metrics`), plus the static UI |
| Observability | `src/observability.py` — JSONL trace of every LLM/tool call and guardrail hit; `/metrics` aggregates it |

## Running it

```bash
pip install -r requirements.txt

# Live mode (real Claude calls): requires an API key
export ANTHROPIC_API_KEY=sk-ant-...
python3 -c "from src.agent import EVCopilotAgent; a = EVCopilotAgent(); print(a.handle_message('How does DC fast charging work?')['response'])"

# Serve it (also serves the chat UI at http://localhost:8000/)
uvicorn src.serve:app --reload --port 8000
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"How long is the battery warranty?"}'
```

Open `http://localhost:8000/` for the chat UI once the server's running, or keep using curl —
both hit the same backend and session store.

No `ANTHROPIC_API_KEY`? `EVCopilotAgent` falls back to `MockLLMClient`, a deterministic
rule-based stand-in with no network calls. That's what the eval suite runs against by default,
so the whole thing (agent loop, tools, guardrails, tracing) works and is testable with zero
external dependencies. `LiveLLMClient` and `MockLLMClient` share one interface, so switching
to live mode doesn't touch anything else.

## Running the evals

```bash
python3 eval/run_eval.py
```

15 cases (10 end-to-end agent trajectories + 5 guardrail unit tests), writes
`eval/eval_report.json`. 15/15 pass against the mock client right now. Covers:

- **Grounding**: does the answer cite actual figures from the docs (charging voltages,
  warranty terms) instead of making them up?
- **Tool use**: does a vehicle-specific question trigger a status lookup? Does an urgent DTC
  found mid-lookup chain into an escalation instead of just getting reported back?
- **Graceful failure**: unknown VIN → clear "not found," not a fabricated status or a crash.
- **Safety refusal**: DIY high-voltage repair requests get refused regardless of phrasing,
  pointed to a certified technician.
- **Prompt injection**: an injection attempt in the user's message shouldn't leak the system
  prompt or change the rules.
- **Hallucination check**: asking about a feature that doesn't exist should get an honest "I
  don't know," not a fabricated answer.
- **Guardrail backstop**: if the model itself fails to refuse unsafe content (forced via a
  test-only trigger phrase), the deterministic output guardrail should still catch it.
- **Guardrail unit tests**: injection detection on a simulated poisoned KB entry,
  untrusted-content tagging, unsafe-output detection, benign-input false-positive check, PII
  redaction in logs.

Worth re-running in live mode (with `ANTHROPIC_API_KEY` set) to see how a real model's
judgment compares to the mock's keyword rules on the same cases.

## Known limitations

- Retrieval is TF-IDF, not real embeddings — fine for a ~15-document corpus, won't catch
  synonyms or paraphrasing the way a real embedding model would.
- The mock LLM is keyword-driven, shaped around this repo's own eval cases. It's not a stand-in
  for real model quality and won't generalize to phrasing outside those cases.
- Guardrails are regex heuristics, not a classifier or a second-model judge — fine for a demo,
  not for production.
- Fleet/ticket "backends" are JSON files standing in for a telematics API and a CRM.

## Ideas for later

- Swap `TfidfEmbedder` for real embeddings + a vector store (Chroma, pgvector).
- A second agent/judge call that critiques the first agent's answer before it's returned.
- A live-vs-mock eval report that scores agreement rate, instead of just eyeballing it.
- OpenTelemetry export from `observability.py` instead of JSONL.
