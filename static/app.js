const messagesEl = document.getElementById("messages");
const composerEl = document.getElementById("composer");
const inputEl = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const modeBadge = document.getElementById("mode-badge");
const sessionIdEl = document.getElementById("session-id");
const resetBtn = document.getElementById("reset-btn");
const suggestionsEl = document.getElementById("suggestions");

const traceEmpty = document.getElementById("trace-empty");
const traceContent = document.getElementById("trace-content");
const traceLatency = document.getElementById("trace-latency");
const traceTools = document.getElementById("trace-tools");
const traceGuardrails = document.getElementById("trace-guardrails");
const traceCritique = document.getElementById("trace-critique");

let sessionId = null;

function addMessage(role, text, { pending = false, error = false } = {}) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role}${pending ? " msg-pending" : ""}${error ? " msg-error" : ""}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function renderTrace(result) {
  traceEmpty.classList.add("hidden");
  traceContent.classList.remove("hidden");
  traceLatency.textContent = `${result.latency_ms} ms`;

  traceTools.innerHTML = "";
  if (result.tool_calls.length === 0) {
    traceTools.innerHTML = '<span class="muted">none</span>';
  } else {
    for (const call of result.tool_calls) {
      const chip = document.createElement("span");
      chip.className = "chip tool";
      chip.textContent = call.name;
      traceTools.appendChild(chip);
    }
  }

  traceGuardrails.innerHTML = "";
  if (result.guardrail_events.length === 0) {
    traceGuardrails.innerHTML = '<span class="muted">none</span>';
  } else {
    for (const event of result.guardrail_events) {
      const chip = document.createElement("span");
      chip.className = "chip guardrail";
      chip.textContent = event.type || JSON.stringify(event);
      traceGuardrails.appendChild(chip);
    }
  }

  traceCritique.innerHTML = "";
  if (!result.critique) {
    traceCritique.innerHTML = '<span class="muted">not run</span>';
  } else {
    const chip = document.createElement("span");
    chip.className = `chip ${result.critique.approved ? "tool" : "guardrail"}`;
    chip.textContent = result.critique.approved ? "approved" : "revised";
    traceCritique.appendChild(chip);
    const note = document.createElement("div");
    note.className = "muted";
    note.style.marginTop = "4px";
    note.textContent = result.critique.note;
    traceCritique.appendChild(note);
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    modeBadge.textContent = data.llm_mode === "live" ? "live: Claude" : "mock mode";
    modeBadge.className = `mode-badge ${data.llm_mode === "live" ? "live" : "mock"}`;
  } catch (err) {
    modeBadge.textContent = "backend unreachable";
    modeBadge.className = "mode-badge down";
  }
}

async function sendMessage(text) {
  addMessage("user", text);
  const pendingMsg = addMessage("assistant", "Thinking…", { pending: true });
  sendBtn.disabled = true;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!res.ok) {
      throw new Error(`Server responded ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;
    sessionIdEl.textContent = sessionId;

    pendingMsg.querySelector(".msg-bubble").textContent = data.response;
    pendingMsg.classList.remove("msg-pending");
    renderTrace(data);
  } catch (err) {
    pendingMsg.querySelector(".msg-bubble").textContent =
      `Something went wrong talking to the backend: ${err.message}`;
    pendingMsg.classList.remove("msg-pending");
    pendingMsg.classList.add("msg-error");
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  sendMessage(text);
});

suggestionsEl.addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li) return;
  inputEl.value = li.textContent;
  inputEl.focus();
});

resetBtn.addEventListener("click", () => {
  sessionId = null;
  sessionIdEl.textContent = "no session yet";
  messagesEl.innerHTML = "";
  addMessage(
    "assistant",
    "New session started. Ask me about charging, battery health, warranty coverage, or your vehicle's status."
  );
  traceEmpty.classList.remove("hidden");
  traceContent.classList.add("hidden");
});

checkHealth();
