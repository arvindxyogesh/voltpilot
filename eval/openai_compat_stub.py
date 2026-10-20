"""
Contract test for OpenAICompatLLMClient, with no model download needed.

Starts a stand-in OpenAI-compatible /v1/chat/completions server whose "model"
is MockLLMClient, then runs the full eval suite through OpenAICompatLLMClient
over real HTTP. The stub translates OpenAI-format requests back into the
agent's Anthropic-style blocks with its own code (not the client's), so a
full pass means the client's translation holds up in both directions:
system prompt, tool schemas, tool_use -> tool_calls, tool_result -> "tool"
messages, chained tool calls, and the critique call.

This checks the plumbing, not model quality. For a real model:

    ollama pull qwen3:8b
    LLM_PROVIDER=openai_compat python3 eval/run_eval.py

Usage: python3 eval/openai_compat_stub.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI, Request

from src.agent import CRITIQUE_SYSTEM_PROMPT, UNGROUNDED_TRIGGER, MockLLMClient

STUB_MODEL = "mock-over-http"
app = FastAPI()
_mock = MockLLMClient()


def _to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    system = ""
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system = m["content"]
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            # Consecutive tool results belong to one user turn on the Anthropic side.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant":
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for call in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                               "input": json.loads(call["function"]["arguments"])})
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": role, "content": m["content"]})
    return system, out


def _completion(text: str | None, tool_calls: list[dict]) -> dict:
    return {
        "id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()), "model": STUB_MODEL,
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": {"role": "assistant", "content": text, **({"tool_calls": tool_calls} if tool_calls else {})},
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict:
    body = await request.json()
    system, messages = _to_anthropic(body["messages"])

    if system == CRITIQUE_SYSTEM_PROMPT:
        prompt = messages[-1]["content"]
        if UNGROUNDED_TRIGGER in prompt:
            return _completion("REVISE: states a specific claim with no supporting tool result.", [])
        return _completion("APPROVE", [])

    # Tools must arrive in OpenAI function format; hand them to the mock in Anthropic format.
    tools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
              "input_schema": t["function"]["parameters"]} for t in body.get("tools", [])]
    resp = _mock.create(system=system, messages=messages, tools=tools)

    text = "".join(b["text"] for b in resp.content if b["type"] == "text") or None
    tool_calls = [{"id": b["id"], "type": "function",
                   "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                  for b in resp.content if b["type"] == "tool_use"]
    return _completion(text, tool_calls)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    os.environ.update({
        "LLM_PROVIDER": "openai_compat",
        "LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "LLM_MODEL": STUB_MODEL,
        "LLM_API_KEY": "stub",
    })
    from eval import run_eval

    # Keep the committed mock-mode report intact; this run is a plumbing check.
    run_eval.REPORT_PATH = Path(__file__).parent / "openai_compat_stub_report.json"
    code = run_eval.main()
    server.should_exit = True
    return code


if __name__ == "__main__":
    raise SystemExit(main())
