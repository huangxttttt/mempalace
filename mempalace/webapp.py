#!/usr/bin/env python3
"""
webapp.py — Minimal local web UI for directory-based Q&A.

This keeps the stack dependency-free: the server uses Python's standard
library only, while MemPalace still handles indexing and retrieval.
"""

from __future__ import annotations

import html
import io
import json
import re
import traceback
import uuid
from contextlib import redirect_stdout
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import chromadb

from .answerer import ask_memories, build_chat_messages, build_context_block
from .config import MempalaceConfig
from .entity_detector import confirm_entities, detect_entities, scan_for_detection
from .llm_client import list_models, stream_chat_completion
from .miner import mine
from .room_detector_local import detect_rooms_local


class WebAppState:
    def __init__(self):
        self.current_dir = ""
        self.current_wing = ""
        self.current_session_id = ""
        self.model_base_url = "http://127.0.0.1:11434"
        self.model_api_key = ""
        self.model_name = ""
        self.retrieval_k = 5
        self.max_history_turns = 4
        self.chat_sessions = []
        self.chat_history = []
        self.last_question = ""
        self.last_answer = ""
        self.last_sources = []
        self.last_hits = []
        self.last_context = ""
        self.last_status = ""
        self.last_logs = ""
        self.last_error = ""

    @property
    def palace_path(self) -> str:
        return MempalaceConfig().palace_path

    @property
    def settings_path(self) -> Path:
        return Path.home() / ".mempalace" / "web_settings.json"

    @property
    def chat_history_path(self) -> Path:
        return Path.home() / ".mempalace" / "web_chat_history.json"

    def set_error(self, message: str):
        self.last_error = message

    def clear_error(self):
        self.last_error = ""

    def load_settings(self):
        try:
            if self.settings_path.exists():
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                self.model_base_url = data.get("model_base_url", self.model_base_url)
                self.model_api_key = data.get("model_api_key", self.model_api_key)
                self.model_name = data.get("model_name", self.model_name)
                self.retrieval_k = int(data.get("retrieval_k", self.retrieval_k))
                self.max_history_turns = int(data.get("max_history_turns", self.max_history_turns))
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    def load_chat_history(self):
        try:
            if self.chat_history_path.exists():
                data = json.loads(self.chat_history_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.chat_sessions = data.get("sessions", [])
                    self.current_session_id = data.get("current_session_id", "")
                    self._sync_current_session()
                elif isinstance(data, list):
                    self.chat_history = data
        except (OSError, json.JSONDecodeError, ValueError):
            self.chat_history = []
            self.chat_sessions = []

    def save_settings(self):
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_base_url": self.model_base_url,
            "model_api_key": self.model_api_key,
            "model_name": self.model_name,
            "retrieval_k": self.retrieval_k,
            "max_history_turns": self.max_history_turns,
        }
        self.settings_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_chat_history(self):
        self.chat_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.chat_history_path.write_text(
            json.dumps(
                {
                    "current_session_id": self.current_session_id,
                    "sessions": self.chat_sessions,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _sync_current_session(self):
        session = self.get_current_session()
        if session:
            self.chat_history = session.get("messages", [])
            self.current_dir = session.get("directory", self.current_dir)
            self.current_wing = session.get("wing", self.current_wing)
        elif self.chat_sessions:
            self.current_session_id = self.chat_sessions[0]["id"]
            self._sync_current_session()

    def get_current_session(self):
        for session in self.chat_sessions:
            if session["id"] == self.current_session_id:
                return session
        return None

    def ensure_session(self):
        session = self.get_current_session()
        if session:
            return session
        return self.create_session(directory=self.current_dir, wing=self.current_wing)

    def create_session(self, directory: str = "", wing: str = ""):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session = {
            "id": uuid.uuid4().hex[:12],
            "title": Path(directory).name if directory else "New Session",
            "directory": directory,
            "wing": wing,
            "created_at": created_at,
            "updated_at": created_at,
            "messages": [],
        }
        self.chat_sessions.insert(0, session)
        self.current_session_id = session["id"]
        self.chat_history = session["messages"]
        self.save_chat_history()
        return session

    def switch_session(self, session_id: str) -> bool:
        for session in self.chat_sessions:
            if session["id"] == session_id:
                self.current_session_id = session_id
                self.chat_history = session.get("messages", [])
                self.current_dir = session.get("directory", "")
                self.current_wing = session.get("wing", "")
                self.save_chat_history()
                return True
        return False

    def append_message_to_current_session(self, message: dict):
        session = self.ensure_session()
        session.setdefault("messages", []).append(message)
        session["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if message.get("question") and session.get("title") in ("New Session", Path(session.get("directory", "")).name):
            session["title"] = message["question"][:40]
        self.chat_history = session["messages"]
        self.save_chat_history()

    def delete_session(self, session_id: str) -> bool:
        before = len(self.chat_sessions)
        self.chat_sessions = [session for session in self.chat_sessions if session["id"] != session_id]
        if len(self.chat_sessions) == before:
            return False

        if self.current_session_id == session_id:
            if self.chat_sessions:
                self.current_session_id = self.chat_sessions[0]["id"]
                self._sync_current_session()
            else:
                self.current_session_id = ""
                self.current_dir = ""
                self.current_wing = ""
                self.chat_history = []
                self.create_session()
        self.save_chat_history()
        return True


STATE = WebAppState()
STATE.load_settings()
STATE.load_chat_history()
if not STATE.chat_sessions:
    STATE.create_session()


def _escape(value: str) -> str:
    return html.escape(value or "", quote=True)


def _capture_output(fn, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn(*args, **kwargs)
    return buffer.getvalue()


def index_directory(directory: str) -> str:
    target = Path(directory).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        raise ValueError(f"Directory not found: {target}")

    files = scan_for_detection(str(target))
    if files:
        detected = detect_entities(files)
        total = (
            len(detected["people"]) + len(detected["projects"]) + len(detected["uncertain"])
        )
        if total > 0:
            confirmed = confirm_entities(detected, yes=True)
            if confirmed["people"] or confirmed["projects"]:
                entities_path = target / "entities.json"
                with open(entities_path, "w", encoding="utf-8") as handle:
                    json.dump(confirmed, handle, indent=2, ensure_ascii=False)

    init_output = _capture_output(detect_rooms_local, str(target), True)
    mine_output = _capture_output(
        mine,
        project_dir=str(target),
        palace_path=MempalaceConfig().palace_path,
        wing_override=None,
        agent="mempalace-web",
        limit=0,
        dry_run=False,
        respect_gitignore=True,
        include_ignored=None,
    )
    return (init_output + "\n" + mine_output).strip()


def choose_directory_dialog(initial_dir: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Native folder picker is unavailable in this Python environment.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            initialdir=initial_dir or str(Path.home()),
            title="Choose A Directory For MemPalace",
            mustexist=True,
        )
    finally:
        root.destroy()
    return selected or ""


def directory_to_wing(directory: str) -> str:
    target = Path(directory).expanduser().resolve()
    return target.name.lower().replace(" ", "_").replace("-", "_")


def get_directory_index_status(wing: str) -> dict:
    if not wing:
        return {"exists": False, "drawers": 0, "files": 0}
    try:
        client = chromadb.PersistentClient(path=MempalaceConfig().palace_path)
        col = client.get_collection("mempalace_drawers")
        data = col.get(where={"wing": wing}, include=["metadatas"], limit=10000)
        metas = data.get("metadatas", [])
        files = {meta.get("source_file", "") for meta in metas if meta.get("source_file")}
        return {"exists": len(metas) > 0, "drawers": len(metas), "files": len(files)}
    except Exception:
        return {"exists": False, "drawers": 0, "files": 0}


def clear_directory_index(wing: str) -> int:
    if not wing:
        return 0
    try:
        client = chromadb.PersistentClient(path=MempalaceConfig().palace_path)
        col = client.get_collection("mempalace_drawers")
        data = col.get(where={"wing": wing}, include=[], limit=10000)
        ids = data.get("ids", [])
        if ids:
            col.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def reset_chat_state():
    STATE.chat_history = []
    STATE.last_question = ""
    STATE.last_answer = ""
    STATE.last_sources = []
    STATE.last_hits = []
    STATE.last_context = ""
    session = STATE.get_current_session()
    if session is not None:
        session["messages"] = []
        session["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    STATE.save_chat_history()


def format_sources(citations: list[dict]) -> str:
    if not citations:
        return ""

    lines = []
    for citation in citations:
        lines.append(
            f"{citation['wing']} / {citation['room']} / {citation['source_file']} "
            f"(match {citation['similarity']})"
        )
    return "\n".join(lines)


def render_hits(hits: list[dict]) -> str:
    if not hits:
        return ""

    blocks = []
    for idx, hit in enumerate(hits, 1):
        blocks.append(
            f"""
            <article class="hit">
              <div class="hit-meta">[{idx}] {_escape(hit['wing'])} / {_escape(hit['room'])}</div>
              <div class="hit-meta">Source: {_escape(hit['source_file'])} | Match: {_escape(str(hit['similarity']))}</div>
              <pre>{html.escape(hit['text'])}</pre>
            </article>
            """
        )
    return "\n".join(blocks)


def render_markdown(text: str) -> str:
    if not text:
        return ""

    code_blocks = []

    def _stash_code_block(match):
        language = match.group(1) or ""
        code = html.escape(match.group(2))
        class_attr = f' class="lang-{html.escape(language)}"' if language else ""
        placeholder = f"@@CODEBLOCK{len(code_blocks)}@@"
        code_blocks.append(f"<pre><code{class_attr}>{code}</code></pre>")
        return placeholder

    escaped = re.sub(r"```([\w+-]*)\n(.*?)```", _stash_code_block, text, flags=re.S)
    escaped = html.escape(escaped)
    escaped = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*\n]+)\*", r"<em>\1</em>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        r'<a href="\2" target="_blank" rel="noreferrer">\1</a>',
        escaped,
    )

    lines = escaped.splitlines()
    parts = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            continue

        if stripped.startswith("@@CODEBLOCK") and stripped.endswith("@@"):
            close_list()
            parts.append(stripped)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            close_list()
            level = len(heading_match.group(1))
            parts.append(f"<h{level}>{heading_match.group(2)}</h{level}>")
            continue

        list_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if list_match:
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{list_match.group(1)}</li>")
            continue

        close_list()
        parts.append(f"<p>{stripped}</p>")

    close_list()
    rendered = "\n".join(parts)
    for idx, block in enumerate(code_blocks):
        rendered = rendered.replace(f"@@CODEBLOCK{idx}@@", block)
    return rendered


def _render_status_panel() -> str:
    if not (STATE.last_status or STATE.last_logs or STATE.last_error):
        return ""

    status_class = "error" if STATE.last_error else "status"
    status_text = STATE.last_error or STATE.last_status
    return f"""
    <section class="card {status_class}">
      <h2>Status</h2>
      <pre>{html.escape(status_text)}</pre>
      <h3>Logs</h3>
      <pre>{html.escape(STATE.last_logs or "No logs yet")}</pre>
    </section>
    """


def _render_chat_history() -> str:
    if not STATE.chat_history:
        return """
        <section class="empty-chat">
          <h2>No messages yet</h2>
          <p>Open Settings first, choose a directory and model, then come back here to ask questions.</p>
        </section>
        """

    items = []
    for item in STATE.chat_history:
        user = f"""
        <article class="bubble user-bubble">
          <div class="bubble-role">You</div>
          <div class="bubble-time">{_escape(item.get('created_at', ''))}</div>
          <div class="bubble-text">{html.escape(item['question'])}</div>
        </article>
        """
        assistant = f"""
        <article class="bubble assistant-bubble">
          <div class="bubble-role">Assistant</div>
          <div class="bubble-time">{_escape(item.get('created_at', ''))}</div>
          <div class="markdown-body">{render_markdown(item['answer'])}</div>
          <details class="details-panel">
            <summary>Retrieved Results</summary>
            {render_hits(item["hits"])}
          </details>
        </article>
        """
        items.append(user + assistant)
    return "\n".join(items)


def _render_session_list() -> str:
    items = []
    for session in STATE.chat_sessions:
        active = "session-link active" if session["id"] == STATE.current_session_id else "session-link"
        title = session.get("title") or "Untitled"
        subtitle = session.get("directory") or "No directory"
        items.append(
            f"""
            <div class="session-row">
              <form method="post" action="/switch-session">
                <input type="hidden" name="session_id" value="{_escape(session['id'])}">
                <button type="submit" class="{active}">
                  <div class="session-title">{_escape(title)}</div>
                  <div class="session-meta">{_escape(subtitle)}</div>
                  <div class="session-meta">{_escape(session.get('updated_at', ''))}</div>
                </button>
              </form>
              <form method="post" action="/delete-session" class="session-delete-form">
                <input type="hidden" name="session_id" value="{_escape(session['id'])}">
                <button type="submit" class="session-delete" title="Delete session">×</button>
              </form>
            </div>
            """
        )
    return "\n".join(items)


def _render_chat_page() -> str:
    settings_hint = "ready" if STATE.current_dir and STATE.model_name else "incomplete"
    return f"""
    <div class="meta chat-meta">
      <span>Directory: {_escape(STATE.current_dir or "none")}</span>
      <span>Wing: {_escape(STATE.current_wing or "none")}</span>
      <span>Model: {_escape(STATE.model_name or "none")}</span>
      <span>Setup: {_escape(settings_hint)}</span>
      <span>History turns: {_escape(str(STATE.max_history_turns))}</span>
    </div>

    <section class="workspace">
      <aside class="session-sidebar card" id="session-sidebar">
        <div class="session-sidebar-head">
          <h2>Sessions</h2>
          <form method="post" action="/new-session">
            <button type="submit">New</button>
          </form>
        </div>
        <div class="session-list">
          {_render_session_list()}
        </div>
      </aside>

      <section class="chat-shell">
        <div class="chat-topbar">
          <button type="button" class="sidebar-toggle" id="sidebar-toggle">Sessions</button>
        </div>
        <section class="chat-stream" id="chat-stream">
          {_render_chat_history()}
        </section>

        <form class="composer" method="post" action="/api/ask" id="chat-form">
          <textarea
            name="question"
            id="chat-input"
            placeholder="问一个关于当前目录的问题"
            autocomplete="off"
            rows="1"
          >{_escape(STATE.last_question)}</textarea>
          <div class="composer-actions">
            <a class="clear-link" href="/clear-chat">Clear Chat</a>
            <button type="submit" id="send-button">Send</button>
          </div>
        </form>
      </section>
    </section>

    """


def _render_settings_page() -> str:
    index_status = get_directory_index_status(STATE.current_wing)
    indexed_label = "indexed" if index_status["exists"] else "not indexed"
    return f"""
    <section class="hero settings-hero">
      <div>
        <h1>Settings</h1>
        <p>Configure the directory, indexing lifecycle, and model endpoint here. The chat page uses only what is defined in these sections.</p>
      </div>
      <div class="meta settings-meta">
        <span>Palace: {_escape(STATE.palace_path)}</span>
        <span>Directory: {_escape(STATE.current_dir or "none")}</span>
        <span>Model: {_escape(STATE.model_name or "none")}</span>
        <span>Index: {_escape(indexed_label)}</span>
      </div>
    </section>

    <section class="settings-stack">
      <section class="settings-section light-section">
        <div class="section-copy">
          <h2>Directory</h2>
          <p>Pick the active knowledge source, inspect index coverage, and rebuild or clear the current directory when needed.</p>
        </div>
        <div class="settings-panel">
          <div class="stat-row">
            <div class="stat-chip">
              <span class="stat-label">Drawers</span>
              <strong>{_escape(str(index_status["drawers"]))}</strong>
            </div>
            <div class="stat-chip">
              <span class="stat-label">Files</span>
              <strong>{_escape(str(index_status["files"]))}</strong>
            </div>
          </div>
          <form method="post" action="/browse-directory">
            <button type="submit">Browse Folder</button>
          </form>
          <form method="post" action="/index">
            <label>
              Local directory path
              <input type="text" name="directory" value="{_escape(STATE.current_dir)}" placeholder="E:\\docs or C:\\work\\project">
            </label>
            <button class="secondary" type="submit">Initialize And Mine</button>
          </form>
          <form method="post" action="/clear-index">
            <button type="submit">Clear Current Directory Index</button>
          </form>
        </div>
      </section>

      <section class="settings-section dark-section">
        <div class="section-copy">
          <h2>Model</h2>
          <p>Use any OpenAI-compatible endpoint. Keep the interface restrained and reserve the blue accent for the actions that matter.</p>
        </div>
        <div class="settings-panel dark-panel">
          <form method="post" action="/settings">
            <label>
              Base URL
              <input type="text" name="base_url" value="{_escape(STATE.model_base_url)}" placeholder="http://127.0.0.1:11434">
            </label>
            <label>
              API key
              <input type="text" name="api_key" value="{_escape(STATE.model_api_key)}" placeholder="Leave empty for local models">
            </label>
            <label>
              Model
              <input type="text" name="model" value="{_escape(STATE.model_name)}" placeholder="qwen2.5:7b-instruct or gpt-4o-mini">
            </label>
            <label>
              Retrieval top K
              <input type="number" name="retrieval_k" min="1" max="20" value="{_escape(str(STATE.retrieval_k))}">
            </label>
            <label>
              Chat memory turns
              <input type="number" name="max_history_turns" min="0" max="12" value="{_escape(str(STATE.max_history_turns))}">
            </label>
            <button type="submit">Save Settings</button>
          </form>
          <form method="post" action="/test-model">
            <button class="ghost-light" type="submit">Test Model Connection</button>
          </form>
        </div>
      </section>

      <section class="settings-section light-section">
        <div class="section-copy">
          <h2>Status</h2>
          <p>Operational feedback stays here so the chat surface remains clean and focused.</p>
        </div>
        <div class="settings-panel">
          {_render_status_panel()}
        </div>
      </section>
    </section>
    """


def render_page(tab: str = "chat") -> str:
    chat_active = "nav-link active" if tab == "chat" else "nav-link"
    settings_active = "nav-link active" if tab == "settings" else "nav-link"
    body = _render_settings_page() if tab == "settings" else _render_chat_page()
    script_block = """
  <script>
    (function () {
      const stream = document.getElementById("chat-stream");
      const workspace = document.querySelector(".workspace");
      const sidebarToggle = document.getElementById("sidebar-toggle");
      if (stream) {
        stream.scrollTop = stream.scrollHeight;
      }

      const isMobile = window.matchMedia("(max-width: 640px)").matches;
      if (workspace && !isMobile) {
        const collapsed = window.localStorage.getItem("mempalace.sidebarCollapsed") === "true";
        if (collapsed) {
          workspace.classList.add("sidebar-collapsed");
        }
      }
      if (sidebarToggle && workspace && !isMobile) {
        sidebarToggle.addEventListener("click", function () {
          const collapsed = workspace.classList.toggle("sidebar-collapsed");
          window.localStorage.setItem("mempalace.sidebarCollapsed", String(collapsed));
        });
      }

      const form = document.getElementById("chat-form");
      const input = document.getElementById("chat-input");
      const button = document.getElementById("send-button");

      function autoGrow() {
        if (!input) return;
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 180) + "px";
      }

      if (form && input && button) {
        autoGrow();
        input.addEventListener("input", autoGrow);
        input.addEventListener("keydown", function (event) {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (input.value.trim()) {
              form.requestSubmit();
            }
          }
        });

        form.addEventListener("submit", async function (event) {
          event.preventDefault();
          const hasText = input.value.trim().length > 0;
          if (!hasText) {
            return;
          }

          const question = input.value;
          let assistantBubble = null;
          let assistantContent = null;
          let accumulated = "";

          if (stream) {
            const userBubble = document.createElement("article");
            userBubble.className = "bubble user-bubble";
            userBubble.innerHTML =
              '<div class="bubble-role">You</div>' +
              '<div class="bubble-time">just now</div>' +
              '<div class="bubble-text"></div>';
            userBubble.querySelector(".bubble-text").textContent = question;
            stream.appendChild(userBubble);

            assistantBubble = document.createElement("article");
            assistantBubble.className = "bubble assistant-bubble";
            assistantBubble.innerHTML =
              '<div class="bubble-role">Assistant</div>' +
              '<div class="bubble-time">just now</div>' +
              '<div class="thinking">Assistant is thinking...</div>' +
              '<div class="markdown-body" style="display:none;"></div>';
            assistantContent = assistantBubble.querySelector(".markdown-body");
            stream.appendChild(assistantBubble);
            stream.scrollTop = stream.scrollHeight;
          }

          input.setAttribute("disabled", "disabled");
          button.setAttribute("disabled", "disabled");
          button.classList.add("loading");
          button.textContent = "Waiting...";
          input.value = "";
          autoGrow();

          try {
            const response = await fetch(form.action, {
              method: "POST",
              headers: {
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
              },
              body: new URLSearchParams({ question })
            });

            if (!response.ok || !response.body) {
              const errorText = await response.text();
              throw new Error(errorText || "Request failed.");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
              const chunk = await reader.read();
              const value = chunk.value;
              const done = chunk.done;
              if (done) break;
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split("\\n");
              buffer = lines.pop() || "";

              for (const line of lines) {
                if (!line.trim()) continue;
                const payload = JSON.parse(line);
                if (payload.type === "delta" && assistantContent) {
                  accumulated += payload.content || "";
                  const thinking = assistantBubble.querySelector(".thinking");
                  if (thinking) thinking.remove();
                  assistantContent.style.display = "block";
                  assistantContent.textContent = accumulated;
                  stream.scrollTop = stream.scrollHeight;
                } else if (payload.type === "done" && assistantBubble) {
                  const thinking = assistantBubble.querySelector(".thinking");
                  if (thinking) thinking.remove();
                  if (assistantContent) {
                    assistantContent.style.display = "block";
                    assistantContent.innerHTML = payload.html || "";
                  }
                  if (payload.hits_html) {
                    const details = document.createElement("details");
                    details.className = "details-panel";
                    details.innerHTML = "<summary>Retrieved Results</summary>" + payload.hits_html;
                    assistantBubble.appendChild(details);
                  }
                  const timeNode = assistantBubble.querySelector(".bubble-time");
                  if (timeNode && payload.created_at) {
                    timeNode.textContent = payload.created_at;
                  }
                  stream.scrollTop = stream.scrollHeight;
                } else if (payload.error) {
                  throw new Error(payload.error);
                }
              }
            }
          } catch (error) {
            if (assistantBubble && assistantContent) {
              const thinking = assistantBubble.querySelector(".thinking");
              if (thinking) thinking.remove();
              assistantContent.style.display = "block";
              assistantContent.textContent = "Error: " + error.message;
            }
          } finally {
            input.removeAttribute("disabled");
            button.removeAttribute("disabled");
            button.classList.remove("loading");
            button.textContent = "Send";
            input.focus();
          }
        });
      }
    })();
  </script>"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MemPalace Local QA</title>
  <style>
    :root {{
      --bg: #f5f5f7;
      --panel: #ffffff;
      --surface: #fbfbfd;
      --ink: #1d1d1f;
      --muted: rgba(0, 0, 0, 0.56);
      --accent: #0071e3;
      --accent-2: #1d1d1f;
      --border: rgba(0, 0, 0, 0.08);
      --nav-bg: rgba(0, 0, 0, 0.8);
      --error: #7e2f2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "SF Pro Text", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 20px 40px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
      min-height: 48px;
      padding: 0 16px;
      border-radius: 999px;
      background: var(--nav-bg);
      backdrop-filter: saturate(180%) blur(20px);
    }}
    .brand {{
      color: #fff;
      font-size: 0.75rem;
      font-weight: 400;
      letter-spacing: -0.12px;
    }}
    .nav {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .nav-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 32px;
      padding: 0 14px;
      border-radius: 999px;
      text-decoration: none;
      color: rgba(255,255,255,0.92);
      border: 1px solid rgba(255,255,255,0.18);
      background: transparent;
      font-size: 0.88rem;
      font-weight: 400;
      letter-spacing: -0.22px;
    }}
    .nav-link.active {{
      background: var(--accent);
      color: white;
      border-color: transparent;
    }}
    .hero {{
      display: grid;
      gap: 12px;
      margin-bottom: 24px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-family: "SF Pro Display", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: clamp(2.5rem, 4vw, 3.5rem);
      font-weight: 600;
      line-height: 1.07;
      letter-spacing: -0.28px;
    }}
    h2 {{
      font-family: "SF Pro Display", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: 1.31rem;
      font-weight: 700;
      line-height: 1.19;
      letter-spacing: 0.23px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.47;
      letter-spacing: -0.37px;
    }}
    .settings-stack {{
      display: grid;
      gap: 24px;
    }}
    .settings-section {{
      display: grid;
      grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
      gap: 32px;
      align-items: start;
      padding: 40px;
      border-radius: 28px;
    }}
    .light-section {{
      background: var(--panel);
      box-shadow: rgba(0, 0, 0, 0.12) 3px 5px 30px 0px;
    }}
    .dark-section {{
      background: #000000;
      color: #ffffff;
      box-shadow: rgba(0, 0, 0, 0.22) 3px 5px 30px 0px;
    }}
    .section-copy {{
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .section-copy p {{
      max-width: 28rem;
    }}
    .dark-section .section-copy,
    .dark-section .section-copy p {{
      color: rgba(255, 255, 255, 0.88);
    }}
    .settings-panel {{
      display: grid;
      gap: 16px;
      padding: 24px;
      border-radius: 20px;
      background: var(--surface);
    }}
    .dark-panel {{
      background: #1d1d1f;
    }}
    .workspace {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }}
    .workspace.sidebar-collapsed {{
      grid-template-columns: 0 minmax(0, 1fr);
    }}
    .card {{
      background: var(--panel);
      border: 0;
      border-radius: 20px;
      padding: 24px;
      box-shadow: rgba(0, 0, 0, 0.12) 3px 5px 30px 0px;
    }}
    form {{ display: grid; gap: 12px; }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 0.88rem;
      line-height: 1.43;
      letter-spacing: -0.22px;
      color: var(--muted);
    }}
    input,
    textarea {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: white;
    }}
    textarea {{
      resize: vertical;
      min-height: 52px;
      max-height: 180px;
    }}
    button {{
      border: 1px solid transparent;
      border-radius: 999px;
      padding: 8px 16px;
      font: inherit;
      font-size: 1.06rem;
      font-weight: 400;
      line-height: 2.41;
      color: white;
      background: var(--accent);
      cursor: pointer;
    }}
    button.secondary {{
      background: var(--accent-2);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SF Mono", SFMono-Regular, ui-monospace, Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.92rem;
      line-height: 1.5;
      background: var(--surface);
      border-radius: 12px;
      padding: 14px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .chat-meta {{
      margin-bottom: 20px;
    }}
    .settings-meta {{
      margin-top: 4px;
    }}
    .chat-hero {{
      padding: 28px 0 20px;
    }}
    .settings-hero {{
      padding: 16px 0 8px;
    }}
    .status h2, .answer-card h2 {{ margin-bottom: 12px; }}
    .status h3, .answer-card h3 {{ margin: 14px 0 10px; font-size: 1rem; }}
    .error h2, .error pre {{ color: var(--error); }}
    .chat-shell {{
      display: grid;
      gap: 16px;
      min-height: calc(100vh - 120px);
      grid-template-rows: auto minmax(0, 1fr) auto;
      background: var(--panel);
      border-radius: 24px;
      padding: 20px;
      box-shadow: rgba(0, 0, 0, 0.12) 3px 5px 30px 0px;
    }}
    .session-sidebar {{
      position: sticky;
      top: 24px;
      max-height: 80vh;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 12px;
      transition: opacity 160ms ease, transform 160ms ease, padding 160ms ease, border-width 160ms ease;
      background: #000000;
      color: #ffffff;
      box-shadow: rgba(0, 0, 0, 0.22) 3px 5px 30px 0px;
    }}
    .workspace.sidebar-collapsed .session-sidebar {{
      opacity: 0;
      pointer-events: none;
      transform: translateX(-12px);
      padding: 0;
      border-width: 0;
      box-shadow: none;
    }}
    .session-sidebar-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .session-sidebar-head h2 {{
      color: #ffffff;
      font-size: 1.5rem;
      font-weight: 600;
      line-height: 1.1;
      letter-spacing: normal;
    }}
    .chat-topbar {{
      display: flex;
      justify-content: flex-start;
      align-items: center;
    }}
    .sidebar-toggle {{
      min-height: 42px;
      padding: 0 16px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--ink);
      font-weight: 400;
    }}
    .session-list {{
      overflow-y: auto;
      display: grid;
      gap: 10px;
      padding-right: 4px;
      padding-top: 4px;
    }}
    .session-list form {{
      display: block;
    }}
    .session-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: start;
    }}
    .session-link {{
      display: block;
      text-decoration: none;
      color: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.04);
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
      width: 100%;
      text-align: left;
    }}
    .session-delete-form {{
      display: flex;
    }}
    .session-delete {{
      min-width: 40px;
      min-height: 40px;
      padding: 0;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 255, 255, 0.04);
      color: rgba(255, 255, 255, 0.72);
      font-size: 1.1rem;
      line-height: 1;
    }}
    .session-delete:hover {{
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.24);
    }}
    .session-link:hover {{
      transform: translateY(-1px);
      border-color: rgba(255, 255, 255, 0.24);
      background: rgba(255, 255, 255, 0.08);
    }}
    .session-link.active {{
      border-color: var(--accent);
      background: rgba(0, 113, 227, 0.18);
      box-shadow: none;
    }}
    .session-title {{
      font-size: 1.06rem;
      font-weight: 600;
      line-height: 1.24;
      letter-spacing: -0.37px;
      margin-bottom: 6px;
    }}
    .session-meta {{
      color: rgba(255, 255, 255, 0.64);
      font-size: 0.75rem;
      letter-spacing: -0.12px;
      line-height: 1.4;
    }}
    .chat-stream {{
      display: grid;
      gap: 14px;
      align-content: start;
      min-height: 0;
      max-height: calc(100vh - 280px);
      overflow-y: auto;
      padding: 24px;
      border-radius: 20px;
      border: 1px solid var(--border);
      background: var(--bg);
      box-shadow: none;
    }}
    .bubble {{
      max-width: min(860px, 100%);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: rgba(0, 0, 0, 0.08) 0px 8px 24px 0px;
      border: 0;
    }}
    .user-bubble {{
      justify-self: end;
      background: var(--accent);
      color: white;
    }}
    .assistant-bubble {{
      justify-self: start;
      background: var(--panel);
    }}
    .bubble-role {{
      margin-bottom: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: -0.12px;
      text-transform: none;
      opacity: 0.82;
    }}
    .bubble-time {{
      margin-bottom: 12px;
      font-size: 0.75rem;
      color: inherit;
      opacity: 0.7;
    }}
    .bubble-text {{
      white-space: pre-wrap;
      line-height: 1.47;
      letter-spacing: -0.37px;
    }}
    .thinking {{
      opacity: 0.82;
      font-style: italic;
      color: var(--muted);
    }}
    .markdown-body {{
      line-height: 1.65;
    }}
    .markdown-body p,
    .markdown-body ul,
    .markdown-body pre,
    .markdown-body h1,
    .markdown-body h2,
    .markdown-body h3,
    .markdown-body h4 {{
      margin: 0 0 12px;
    }}
    .markdown-body ul {{
      padding-left: 22px;
    }}
    .markdown-body code {{
      padding: 0.14rem 0.38rem;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.06);
      font-family: "SF Mono", SFMono-Regular, ui-monospace, Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.92em;
    }}
    .markdown-body pre {{
      padding: 14px;
      border-radius: 12px;
      background: var(--surface);
      overflow-x: auto;
    }}
    .markdown-body pre code {{
      padding: 0;
      background: transparent;
    }}
    .markdown-body a {{
      color: var(--accent);
    }}
    .composer {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
      padding: 16px;
      border-radius: 20px;
      background: var(--surface);
      border: 1px solid var(--border);
      box-shadow: none;
    }}
    .composer input[disabled],
    .composer textarea[disabled],
    .composer button[disabled] {{
      opacity: 0.7;
      cursor: not-allowed;
    }}
    .composer button.loading {{
      background: var(--ink);
    }}
    .composer-actions {{
      display: grid;
      gap: 10px;
      align-items: center;
      grid-auto-flow: column;
      justify-content: end;
    }}
    .clear-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 0 16px;
      border-radius: 999px;
      text-decoration: none;
      color: var(--muted);
      border: 1px solid var(--border);
      background: white;
      font-weight: 400;
      white-space: nowrap;
    }}
    .clear-link:hover {{
      color: var(--ink);
      border-color: rgba(0, 113, 227, 0.35);
    }}
    .empty-chat {{
      padding: 56px 28px;
      border: 1px dashed var(--border);
      border-radius: 20px;
      color: var(--muted);
      background: var(--surface);
      text-align: center;
    }}
    .details-panel {{
      margin-top: 14px;
    }}
    .details-panel summary {{
      cursor: pointer;
      font-weight: 600;
      color: var(--muted);
    }}
    .hit {{
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--border);
    }}
    .hit-meta {{
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.75rem;
      font-family: "SF Mono", SFMono-Regular, ui-monospace, Consolas, "Liberation Mono", Menlo, monospace;
    }}
    .stat-row {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .stat-chip {{
      display: inline-flex;
      align-items: baseline;
      gap: 8px;
      min-height: 36px;
      padding: 0 12px;
      border-radius: 999px;
      background: var(--surface);
      color: var(--ink);
      border: 1px solid var(--border);
    }}
    .stat-chip strong {{
      font-size: 0.88rem;
      font-weight: 600;
      letter-spacing: -0.22px;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 0.75rem;
      letter-spacing: -0.12px;
    }}
    .settings-status {{
      margin-top: 24px;
    }}
    .status,
    .error {{
      background: transparent;
      border-radius: 0;
      padding: 0;
      box-shadow: none;
    }}
    .dark-section label {{
      color: rgba(255, 255, 255, 0.72);
    }}
    .dark-section input,
    .dark-section textarea {{
      background: #2a2a2d;
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.12);
    }}
    .dark-section button {{
      background: var(--accent);
    }}
    .ghost-light {{
      background: transparent;
      border-color: rgba(255, 255, 255, 0.32);
      color: #ffffff;
    }}
    @media (max-width: 640px) {{
      .shell {{ padding: 20px 14px 36px; }}
      .card {{ border-radius: 14px; }}
      .topbar {{
        align-items: flex-start;
        flex-direction: column;
        border-radius: 18px;
        padding: 10px 14px;
      }}
      .composer {{ grid-template-columns: 1fr; }}
      .composer-actions {{ grid-template-columns: 1fr 1fr; grid-auto-flow: row; }}
      .workspace {{ grid-template-columns: 1fr; }}
      .settings-section {{ grid-template-columns: 1fr; padding: 24px; gap: 20px; }}
      .session-sidebar {{ position: static; max-height: none; }}
      .workspace.sidebar-collapsed {{ grid-template-columns: 1fr; }}
      .workspace.sidebar-collapsed .session-sidebar {{
        opacity: 1;
        pointer-events: auto;
        transform: none;
        padding: 18px;
        border-width: 1px;
        box-shadow: 0 12px 30px rgba(43, 38, 30, 0.08);
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">MemPalace Local QA</div>
      <nav class="nav">
        <a class="{chat_active}" href="/">Chat</a>
        <a class="{settings_active}" href="/settings-page">Settings</a>
      </nav>
    </header>

    {body}
  </main>
{script_block}
</body>
</html>"""


class MemPalaceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/":
            self._send_html(render_page("chat"))
            return
        if self.path.startswith("/session/"):
            session_id = self.path.rsplit("/", 1)[-1]
            if STATE.switch_session(session_id):
                STATE.last_status = f"Opened session: {STATE.get_current_session().get('title', session_id)}"
                STATE.clear_error()
                self._redirect("/")
                return
            self.send_error(404)
            return
        if self.path == "/settings-page":
            self._send_html(render_page("settings"))
            return
        if self.path == "/chat":
            self._redirect("/")
            return
        if self.path == "/settings":
            self._redirect("/settings-page")
            return
        if self.path == "/clear-chat":
            reset_chat_state()
            STATE.last_status = "Chat history cleared."
            STATE.clear_error()
            self._redirect("/")
            return
        self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8", errors="replace")
        form = parse_qs(raw)

        try:
            if self.path == "/api/ask":
                self._handle_api_ask(form)
                return
            if self.path == "/delete-session":
                session_id = (form.get("session_id") or [""])[0].strip()
                if not session_id:
                    raise ValueError("Missing session id.")
                if not STATE.delete_session(session_id):
                    raise ValueError("Session not found.")
                session = STATE.get_current_session()
                STATE.last_status = f"Deleted session. Current session: {session.get('title', 'New Session')}"
                STATE.last_logs = ""
                STATE.clear_error()
                self._redirect("/")
                return
            if self.path == "/switch-session":
                session_id = (form.get("session_id") or [""])[0].strip()
                if not session_id:
                    raise ValueError("Missing session id.")
                if not STATE.switch_session(session_id):
                    raise ValueError("Session not found.")
                session = STATE.get_current_session()
                STATE.last_status = f"Opened session: {session.get('title', session_id)}"
                STATE.last_logs = ""
                STATE.clear_error()
                self._redirect("/")
                return
            if self.path == "/new-session":
                session = STATE.create_session(directory=STATE.current_dir, wing=STATE.current_wing)
                STATE.last_status = f"Created new session: {session['title']}"
                STATE.clear_error()
                self._redirect("/")
                return
            if self.path == "/index":
                directory = (form.get("directory") or [""])[0].strip()
                if not directory:
                    raise ValueError("Please provide a directory path.")
                previous_wing = STATE.current_wing
                STATE.current_dir = directory
                STATE.current_wing = directory_to_wing(directory)
                if STATE.current_wing != previous_wing:
                    reset_chat_state()
                session = STATE.ensure_session()
                session["directory"] = STATE.current_dir
                session["wing"] = STATE.current_wing
                if session.get("title") == "New Session":
                    session["title"] = Path(STATE.current_dir).name or "New Session"
                session["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                STATE.save_chat_history()
                STATE.last_status = f"Indexed directory: {Path(directory).expanduser().resolve()}"
                STATE.last_logs = index_directory(directory)
                STATE.clear_error()
                self._redirect("/settings-page")
                return
            elif self.path == "/browse-directory":
                selected = choose_directory_dialog(STATE.current_dir)
                if selected:
                    previous_wing = STATE.current_wing
                    STATE.current_dir = selected
                    STATE.current_wing = directory_to_wing(selected)
                    if STATE.current_wing != previous_wing:
                        reset_chat_state()
                    session = STATE.ensure_session()
                    session["directory"] = STATE.current_dir
                    session["wing"] = STATE.current_wing
                    session["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if session.get("title") == "New Session":
                        session["title"] = Path(STATE.current_dir).name or "New Session"
                    STATE.save_chat_history()
                    STATE.last_status = f"Selected directory: {selected}"
                    STATE.clear_error()
                self._redirect("/settings-page")
                return
            elif self.path == "/settings":
                STATE.model_base_url = (form.get("base_url") or [STATE.model_base_url])[0].strip()
                STATE.model_api_key = (form.get("api_key") or [""])[0].strip()
                STATE.model_name = (form.get("model") or [""])[0].strip()
                retrieval_k_raw = (form.get("retrieval_k") or [str(STATE.retrieval_k)])[0].strip()
                STATE.retrieval_k = max(1, min(20, int(retrieval_k_raw or "5")))
                history_raw = (form.get("max_history_turns") or [str(STATE.max_history_turns)])[0].strip()
                STATE.max_history_turns = max(0, min(12, int(history_raw or "4")))
                STATE.save_settings()
                STATE.last_status = "Model settings saved."
                STATE.clear_error()
                self._redirect("/settings-page")
                return
            elif self.path == "/test-model":
                if not STATE.model_base_url:
                    raise ValueError("Configure the model base URL first.")
                models = list_models(
                    base_url=STATE.model_base_url,
                    api_key=STATE.model_api_key,
                )
                if STATE.model_name:
                    if STATE.model_name in models:
                        STATE.last_status = f"Model connection ok. Found configured model: {STATE.model_name}"
                    else:
                        preview = ", ".join(models[:10]) if models else "none"
                        STATE.last_status = (
                            f"Connection ok, but configured model '{STATE.model_name}' was not found. "
                            f"Available models: {preview}"
                        )
                else:
                    preview = ", ".join(models[:10]) if models else "none"
                    STATE.last_status = f"Connection ok. Available models: {preview}"
                STATE.last_logs = ""
                STATE.clear_error()
                self._redirect("/settings-page")
                return
            elif self.path == "/clear-index":
                if not STATE.current_wing:
                    raise ValueError("Choose a directory first.")
                deleted = clear_directory_index(STATE.current_wing)
                STATE.last_status = f"Cleared {deleted} drawers from wing: {STATE.current_wing}"
                STATE.last_logs = ""
                STATE.clear_error()
                self._redirect("/settings-page")
                return
            else:
                self.send_error(404)
                return
        except Exception as exc:
            STATE.set_error(str(exc))
            STATE.last_logs = traceback.format_exc(limit=3)
            if self.path == "/api/ask":
                self._send_json({"error": str(exc)}, status=500)
                return
            if self.path == "/ask":
                self._redirect("/")
                return
            self._redirect("/settings-page")
            return

    def _handle_api_ask(self, form):
        question = (form.get("question") or [""])[0].strip()
        if not question:
            self._send_json({"error": "Please provide a question."}, status=400)
            return
        if not STATE.current_dir:
            self._send_json({"error": "Index a directory first."}, status=400)
            return
        if not STATE.model_name:
            self._send_json({"error": "Configure a model first."}, status=400)
            return

        result = ask_memories(
            question,
            MempalaceConfig().palace_path,
            wing=STATE.current_wing or None,
            n_results=STATE.retrieval_k,
        )
        if "error" in result:
            self._send_json({"error": result["error"]}, status=400)
            return

        created_at = datetime.now().strftime("%H:%M:%S")
        hits = result.get("results", [])
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        if not result.get("answer"):
            answer_text = "没有找到足够相关的内容。可以换更贴近原文的关键词再试。"
            record = {
                "question": question,
                "answer": answer_text,
                "hits": [],
                "created_at": created_at,
            }
            STATE.append_message_to_current_session(record)
            self._write_stream_event(
                {
                    "type": "done",
                    "answer": answer_text,
                    "html": render_markdown(answer_text),
                    "hits_html": "",
                    "created_at": created_at,
                }
            )
            return

        messages = build_chat_messages(
            question=question,
            results=hits,
            history=STATE.chat_history,
            max_history_turns=STATE.max_history_turns,
        )
        answer_parts = []
        try:
            for chunk in stream_chat_completion(
                base_url=STATE.model_base_url,
                api_key=STATE.model_api_key,
                model=STATE.model_name,
                messages=messages,
            ):
                answer_parts.append(chunk)
                self._write_stream_event({"type": "delta", "content": chunk})
        except Exception as exc:
            self._write_stream_event({"error": str(exc)})
            return

        answer_text = "".join(answer_parts).strip() or "模型返回了空响应。"
        STATE.last_answer = answer_text
        STATE.last_sources = result.get("citations", [])
        STATE.last_hits = hits
        STATE.last_context = build_context_block(hits)
        record = {
            "question": question,
            "answer": answer_text,
            "hits": hits,
            "created_at": created_at,
        }
        STATE.append_message_to_current_session(record)
        STATE.last_status = f"Answered from palace: {MempalaceConfig().palace_path}"
        STATE.clear_error()
        self._write_stream_event(
            {
                "type": "done",
                "answer": answer_text,
                "html": render_markdown(answer_text),
                "hits_html": render_hits(hits),
                "created_at": created_at,
            }
        )

    def log_message(self, fmt, *args):
        return

    def _send_html(self, payload: str):
        encoded = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict, status: int = 200):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _redirect(self, location: str):
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _write_stream_event(self, payload: dict):
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.wfile.write(encoded)
        self.wfile.flush()


def serve_web(host: str = "127.0.0.1", port: int = 8765):
    server = ThreadingHTTPServer((host, port), MemPalaceHandler)
    print(f"MemPalace Web UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
