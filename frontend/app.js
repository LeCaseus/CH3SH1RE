// ── DOM refs ──────────────────────────────────────────────────
const chatWindow  = document.getElementById("chat_window");
const msgInput    = document.getElementById("msg_input");
const sendBtn     = document.getElementById("send_btn");
const fileInput   = document.getElementById("file_input");
const filePreview = document.getElementById("file_preview");
const fileNameEl  = document.getElementById("file_name");
const removeFile  = document.getElementById("remove_file");
const clearBtn    = document.getElementById("clear_btn");
const statusDot   = document.getElementById("status_dot");
const statusLabel = document.getElementById("status_label");
const welcomeMsg  = document.getElementById("welcome_msg");

const API = "http://127.0.0.1:8000";

// ── status check ──────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch(`${API}/`);
    if (res.ok) {
      statusDot.className = "status-dot online";
      statusLabel.textContent = "ONLINE";
    }
  } catch {
    statusDot.className = "status-dot offline";
    statusLabel.textContent = "OFFLINE";
  }
}

checkStatus();
setInterval(checkStatus, 10000);

// ── file attachment ────────────────────────────────────────────
let attachedFile = null;

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  attachedFile = file;
  fileNameEl.textContent = file.name;
  filePreview.style.display = "flex";
});

removeFile.addEventListener("click", () => {
  attachedFile = null;
  fileInput.value = "";
  filePreview.style.display = "none";
});

// ── auto-resize textarea ───────────────────────────────────────
msgInput.addEventListener("input", () => {
  msgInput.style.height = "auto";
  msgInput.style.height = Math.min(msgInput.scrollHeight, 120) + "px";
});

// ── send on enter (shift+enter for newline) ────────────────────
msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

// ── clear chat ─────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  // remove all messages but keep welcome
  const messages = chatWindow.querySelectorAll(".message");
  messages.forEach(m => m.remove());
  // show welcome again if chat is empty
  if (!welcomeMsg) {
    const welcome = document.createElement("div");
    welcome.className = "welcome";
    welcome.id = "welcome_msg";
    welcome.innerHTML = `
      <p class="welcome-eyebrow">HELLO ✦</p>
      <h2>I'M <em>CH3SH1RE.</em></h2>
      <p class="welcome-sub">Your private local AI. Ask me anything.</p>
    `;
    chatWindow.appendChild(welcome);
  } else {
    welcomeMsg.style.display = "";
  }
});

// ── core send function ─────────────────────────────────────────
async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;

  // hide welcome
  const welcome = document.getElementById("welcome_msg");
  if (welcome) welcome.style.display = "none";

  // render user message
  appendMessage("user", text, attachedFile?.name);

  // clear input
  msgInput.value = "";
  msgInput.style.height = "auto";
  const fileToSend = attachedFile;
  attachedFile = null;
  fileInput.value = "";
  filePreview.style.display = "none";

  // disable input while waiting
  setInputState(false);

  // show thinking indicator
  const thinkingEl = appendThinking();

try {
    const formData = new FormData();
    formData.append("message", text);
    if (fileToSend) formData.append("file", fileToSend);

    const res = await fetch(`${API}/chat`, {
      method: "POST",
      body: formData,
    });

    // first chunk has arrived — swap the thinking dots for a real (empty) bubble
    thinkingEl.remove();
    const aiWrap = appendMessage("ai", "");
    const bubble = aiWrap.querySelector(".msg-bubble");

    // read the stream and append each chunk directly into the bubble
    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bubble.textContent += decoder.decode(value, { stream: true });
      scrollToBottom();
    }

  } catch (err) {
    thinkingEl.remove();
    appendMessage("ai", "Connection error. Is CH3SH1RE running?");
  }
  
  setInputState(true);
  msgInput.focus();
}

// ── helpers ────────────────────────────────────────────────────
function appendMessage(role, text, fileName = null) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const label = document.createElement("p");
  label.className = "msg-label";
  label.textContent = role === "user" ? "YOU ✦" : "CH3SH1RE ✦";
  wrap.appendChild(label);

  if (fileName) {
    const tag = document.createElement("p");
    tag.className = "msg-file-tag";
    tag.textContent = `📎 ${fileName}`;
    wrap.appendChild(tag);
  }

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);

  chatWindow.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function appendThinking() {
  const wrap = document.createElement("div");
  wrap.className = "message ai thinking";

  const label = document.createElement("p");
  label.className = "msg-label";
  label.textContent = "CH3SH1RE ✦";
  wrap.appendChild(label);

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
  wrap.appendChild(bubble);

  chatWindow.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function setInputState(enabled) {
  msgInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}