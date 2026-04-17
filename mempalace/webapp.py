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
from .miner import READABLE_EXTENSIONS, mine
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
            "title": Path(directory).name if directory else "新会话",
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
        if message.get("question") and session.get("title") in ("新会话", Path(session.get("directory", "")).name):
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

DISPLAYABLE_DIRECTORY_EXTENSIONS = (
    ".txt",
    ".md",
    ".doc",
    ".docx",
    ".pdf",
    ".eml",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".csv",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".sql",
    ".toml",
)
SUPPORTED_DIRECTORY_FORMATS = "、".join(
    ext.lstrip(".")
    for ext in DISPLAYABLE_DIRECTORY_EXTENSIONS
    if ext in READABLE_EXTENSIONS
)


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
        raise ValueError(f"未找到目录：{target}")

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
            title="选择知识库目录",
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
      <h2>运行状态</h2>
      <pre>{html.escape(status_text)}</pre>
      <h3>日志</h3>
      <pre>{html.escape(STATE.last_logs or "暂无日志")}</pre>
    </section>
    """


def _render_chat_history() -> str:
    if not STATE.chat_history:
        return """
        <section class="empty-chat">
          <span class="eyebrow">Ready To Ask</span>
          <h2>还没有消息</h2>
          <p>先到设置页绑定目录并完成模型配置，然后回到这里开始提问。</p>
          <div class="empty-chat-actions">
            <a class="nav-link active" href="/settings-page">前往设置</a>
            <span class="meta-pill subtle">建议先完成索引，再开始连续问答</span>
          </div>
        </section>
        """

    items = []
    for item in STATE.chat_history:
        user = f"""
        <article class="bubble user-bubble">
          <div class="bubble-role">你</div>
          <div class="bubble-time">{_escape(item.get('created_at', ''))}</div>
          <div class="bubble-text">{html.escape(item['question'])}</div>
        </article>
        """
        assistant = f"""
        <article class="bubble assistant-bubble">
          <div class="bubble-role">助手</div>
          <div class="bubble-time">{_escape(item.get('created_at', ''))}</div>
          <div class="markdown-body">{render_markdown(item['answer'])}</div>
          <details class="details-panel">
            <summary>检索结果</summary>
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
        title = session.get("title") or "未命名会话"
        subtitle = session.get("directory") or "未绑定目录"
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
                <button type="submit" class="session-delete" title="删除会话">×</button>
              </form>
            </div>
            """
        )
    return "\n".join(items)


def _render_chat_page() -> str:
    settings_hint = "已完成" if STATE.current_dir and STATE.model_name else "未完成"
    index_status = get_directory_index_status(STATE.current_wing)
    indexed_label = "已索引" if index_status["exists"] else "未索引"
    current_session = STATE.get_current_session() or {}
    return f"""
    <section class="chat-status-bar">
      <div class="chat-status-copy">
        <span class="eyebrow eyebrow-muted">MemPalace</span>
        <h1>聊天</h1>
      </div>
    </section>

    <section class="utility-strip compact">
      <div class="utility-item">
        <span>目录</span>
        <strong>{_escape(Path(STATE.current_dir).name if STATE.current_dir else "未设置")}</strong>
      </div>
      <div class="utility-item">
        <span>模型</span>
        <strong>{_escape(STATE.model_name or "未设置")}</strong>
      </div>
      <div class="utility-item">
        <span>索引</span>
        <strong>{_escape(indexed_label)}</strong>
      </div>
      <div class="utility-item">
        <span>分区</span>
        <strong>{_escape(STATE.current_wing or "未设置")}</strong>
      </div>
      <div class="utility-item">
        <span>配置</span>
        <strong>{_escape(settings_hint)}</strong>
      </div>
      <div class="utility-item">
        <span>检索</span>
        <strong>{_escape(str(STATE.retrieval_k))}</strong>
      </div>
      <div class="utility-item">
        <span>记忆</span>
        <strong>{_escape(str(STATE.max_history_turns))}</strong>
      </div>
    </section>

    <section class="workspace">
      <aside class="session-sidebar card" id="session-sidebar">
        <div class="session-sidebar-head">
          <div>
            <span class="eyebrow eyebrow-muted">Conversations</span>
            <h2>会话列表</h2>
          </div>
          <form method="post" action="/new-session">
            <button type="submit">新建</button>
          </form>
        </div>
        <div class="session-list">
          {_render_session_list()}
        </div>
      </aside>

      <section class="chat-shell">
        <div class="chat-topbar">
          <div class="chat-topbar-copy">
            <span class="eyebrow eyebrow-muted">Workspace</span>
            <h2>{_escape(current_session.get("title") or "新会话")}</h2>
          </div>
          <div class="chat-topbar-actions">
            <span class="meta-pill subtle">目录：{_escape(Path(STATE.current_dir).name if STATE.current_dir else "未绑定")}</span>
            <button type="button" class="sidebar-toggle" id="sidebar-toggle">会话</button>
          </div>
        </div>
        <section class="chat-stream" id="chat-stream">
          {_render_chat_history()}
        </section>

        <form class="composer" method="post" action="/api/ask" id="chat-form">
          <textarea
            name="question"
            id="chat-input"
            placeholder="输入一个关于当前目录的问题，Enter 发送，Shift + Enter 换行"
            autocomplete="off"
            rows="1"
          >{_escape(STATE.last_question)}</textarea>
          <div class="composer-actions">
            <a class="clear-link" href="/clear-chat">清空聊天</a>
            <button type="submit" id="send-button">发送</button>
          </div>
        </form>
      </section>
    </section>

    """


def _render_settings_page() -> str:
    index_status = get_directory_index_status(STATE.current_wing)
    indexed_label = "已索引" if index_status["exists"] else "未索引"
    return f"""
    <section class="hero hero-panel settings-hero light-hero">
      <div class="hero-copy">
        <span class="eyebrow">Configuration</span>
        <h1>设置</h1>
        <p>在这里配置目录、索引和模型。聊天页会直接继承这些设置，适合先建库，再进入连续问答。</p>
      </div>
      <div class="hero-summary">
        <div class="hero-summary-label">当前配置</div>
        <dl class="summary-list">
          <div class="summary-row">
            <dt>仓库</dt>
            <dd>{_escape(STATE.palace_path)}</dd>
          </div>
          <div class="summary-row">
            <dt>目录</dt>
            <dd>{_escape(STATE.current_dir or "未设置")}</dd>
          </div>
          <div class="summary-row">
            <dt>模型</dt>
            <dd>{_escape(STATE.model_name or "未设置")}</dd>
          </div>
          <div class="summary-row">
            <dt>索引</dt>
            <dd>{_escape(indexed_label)}</dd>
          </div>
        </dl>
      </div>
    </section>

    <section class="settings-stack">
      <section class="settings-section light-section">
        <div class="section-copy">
          <h2>目录</h2>
          <p>选择当前知识库目录，查看索引情况，并在需要时重新建库或清空索引。</p>
        </div>
        <div class="settings-panel">
          <div class="stat-row">
            <div class="stat-chip">
              <span class="stat-label">抽屉数</span>
              <strong>{_escape(str(index_status["drawers"]))}</strong>
            </div>
            <div class="stat-chip">
              <span class="stat-label">文件数</span>
              <strong>{_escape(str(index_status["files"]))}</strong>
            </div>
          </div>
          <form method="post" action="/browse-directory">
            <button type="submit">选择目录</button>
          </form>
          <form method="post" action="/index">
            <label>
              本地目录
              <input type="text" name="directory" value="{_escape(STATE.current_dir)}" placeholder="例如：E:\\docs 或 C:\\work\\project">
            </label>
            <p class="field-hint">支持解析的文件格式：{_escape(SUPPORTED_DIRECTORY_FORMATS)}</p>
            <button class="secondary" type="submit">初始化并导入</button>
          </form>
          <form method="post" action="/clear-index">
            <button type="submit">清空当前目录索引</button>
          </form>
        </div>
      </section>

      <section class="settings-section dark-section">
        <div class="section-copy">
          <h2>模型</h2>
          <p>支持任意 OpenAI-compatible 接口。这里只保留必要配置，问答时直接调用。</p>
        </div>
        <div class="settings-panel dark-panel">
          <form method="post" action="/settings">
            <label>
              接口地址
              <input type="text" name="base_url" value="{_escape(STATE.model_base_url)}" placeholder="http://127.0.0.1:11434">
            </label>
            <label>
              API Key
              <input type="text" name="api_key" value="{_escape(STATE.model_api_key)}" placeholder="本地模型可留空">
            </label>
            <label>
              模型名
              <input type="text" name="model" value="{_escape(STATE.model_name)}" placeholder="qwen2.5:7b-instruct or gpt-4o-mini">
            </label>
            <label>
              检索条数
              <input type="number" name="retrieval_k" min="1" max="50" value="{_escape(str(STATE.retrieval_k))}">
            </label>
            <label>
              记忆轮数
              <input type="number" name="max_history_turns" min="0" max="12" value="{_escape(str(STATE.max_history_turns))}">
            </label>
            <button type="submit">保存设置</button>
          </form>
          <form method="post" action="/test-model">
            <button class="ghost-light" type="submit">测试模型连接</button>
          </form>
        </div>
      </section>

      <section class="settings-section light-section">
        <div class="section-copy">
          <h2>状态</h2>
          <p>运行状态和日志都放在这里，聊天页保持简洁。</p>
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
              '<div class="bubble-role">你</div>' +
              '<div class="bubble-time">刚刚</div>' +
              '<div class="bubble-text"></div>';
            userBubble.querySelector(".bubble-text").textContent = question;
            stream.appendChild(userBubble);

            assistantBubble = document.createElement("article");
            assistantBubble.className = "bubble assistant-bubble";
            assistantBubble.innerHTML =
              '<div class="bubble-role">助手</div>' +
              '<div class="bubble-time">刚刚</div>' +
              '<div class="thinking">正在思考...</div>' +
              '<div class="markdown-body" style="display:none;"></div>';
            assistantContent = assistantBubble.querySelector(".markdown-body");
            stream.appendChild(assistantBubble);
            stream.scrollTop = stream.scrollHeight;
          }

          input.setAttribute("disabled", "disabled");
          button.setAttribute("disabled", "disabled");
          button.classList.add("loading");
          button.textContent = "请稍候...";
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
                    details.innerHTML = "<summary>检索结果</summary>" + payload.hits_html;
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
              assistantContent.textContent = "出错了：" + error.message;
            }
          } finally {
            input.removeAttribute("disabled");
            button.removeAttribute("disabled");
            button.classList.remove("loading");
            button.textContent = "发送";
            input.focus();
          }
        });
      }
    })();
  </script>"""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MemPalace 本地知识问答</title>
  <style>
    :root {{
      --bg: #f5f5f7;
      --bg-dark: #000000;
      --panel: #ffffff;
      --panel-dark: #1d1d1f;
      --surface: #fbfbfd;
      --surface-dark: #272729;
      --ink: #1d1d1f;
      --ink-dark: #ffffff;
      --muted: rgba(0, 0, 0, 0.56);
      --muted-dark: rgba(255, 255, 255, 0.8);
      --accent: #0071e3;
      --accent-link: #0066cc;
      --accent-dark-link: #2997ff;
      --accent-2: #1d1d1f;
      --border: rgba(0, 0, 0, 0.08);
      --border-dark: rgba(255, 255, 255, 0.16);
      --nav-bg: rgba(0, 0, 0, 0.8);
      --shadow-lg: rgba(0, 0, 0, 0.22) 3px 5px 30px 0px;
      --shadow-md: rgba(0, 0, 0, 0.12) 3px 5px 20px 0px;
      --error: #7e2f2f;
      --radius-xl: 12px;
      --radius-lg: 8px;
      --radius-md: 5px;
      --transition-fast: 180ms ease-out;
      --transition-medium: 260ms ease-out;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "SF Pro Text", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 24px;
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
      letter-spacing: -0.224px;
      transition:
        background var(--transition-fast),
        border-color var(--transition-fast),
        color var(--transition-fast),
        transform var(--transition-fast);
      cursor: pointer;
    }}
    .nav-link:hover {{
      color: #fff;
      border-color: rgba(255,255,255,0.32);
    }}
    .nav-link.active {{
      background: var(--accent);
      color: white;
      border-color: transparent;
    }}
    .hero {{
      display: grid;
      gap: 16px;
      margin-bottom: 24px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-family: "SF Pro Display", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: clamp(2.5rem, 4vw, 3.5rem);
      font-weight: 600;
      line-height: 1.07;
      letter-spacing: -0.28px;
      max-width: 14ch;
    }}
    h2 {{
      font-family: "SF Pro Display", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: 1.31rem;
      font-weight: 700;
      line-height: 1.19;
      letter-spacing: 0.231px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.47;
      letter-spacing: -0.374px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 28px;
      width: fit-content;
      padding: 0 11px;
      border-radius: 999px;
      border: 1px solid rgba(0, 113, 227, 0.18);
      background: rgba(0, 113, 227, 0.08);
      color: var(--accent-link);
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: -0.12px;
      text-transform: uppercase;
    }}
    .eyebrow-muted {{
      border-color: var(--border);
      background: transparent;
      color: var(--muted);
    }}
    .hero-panel {{
      grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr);
      gap: 40px;
      align-items: end;
      padding: 40px;
      border-radius: var(--radius-xl);
      background: var(--bg-dark);
      box-shadow: none;
    }}
    .hero-copy {{
      display: grid;
      gap: 14px;
    }}
    .hero-copy p {{
      max-width: 62ch;
    }}
    .chat-status-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }}
    .chat-status-copy {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}
    .chat-status-copy h1 {{
      font-size: 1.5rem;
      line-height: 1.14;
      letter-spacing: 0.196px;
      max-width: none;
    }}
    .dark-hero {{
      background: var(--bg-dark);
      color: var(--ink-dark);
    }}
    .dark-hero p {{
      color: rgba(255,255,255,0.8);
    }}
    .light-hero {{
      background: var(--bg);
      color: var(--ink);
    }}
    .light-hero .hero-summary {{
      border-left-color: rgba(0, 0, 0, 0.12);
    }}
    .light-hero .hero-summary-label,
    .light-hero .summary-row dt {{
      color: var(--muted);
    }}
    .light-hero .summary-row {{
      border-bottom-color: rgba(0, 0, 0, 0.08);
    }}
    .light-hero .pill-link {{
      color: var(--accent-link);
      border-color: var(--accent-link);
    }}
    .hero-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 8px;
    }}
    .hero-summary {{
      align-self: stretch;
      display: grid;
      align-content: start;
      gap: 18px;
      padding-left: 32px;
      border-left: 1px solid rgba(255, 255, 255, 0.18);
    }}
    .hero-summary-label {{
      font-size: 0.75rem;
      line-height: 1.33;
      letter-spacing: -0.12px;
      text-transform: uppercase;
      color: rgba(255, 255, 255, 0.72);
    }}
    .summary-list {{
      margin: 0;
      display: grid;
      gap: 14px;
    }}
    .summary-row {{
      display: grid;
      gap: 6px;
      padding-bottom: 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .summary-row dt {{
      color: rgba(255, 255, 255, 0.72);
      font-size: 0.75rem;
      line-height: 1.33;
      letter-spacing: -0.12px;
    }}
    .summary-row dd {{
      margin: 0;
      color: inherit;
      font-size: 1.06rem;
      line-height: 1.47;
      letter-spacing: -0.374px;
      word-break: break-word;
    }}
    .pill-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 18px;
      border-radius: 980px;
      border: 1px solid var(--accent-link);
      color: var(--accent-link);
      text-decoration: none;
      font-size: 0.88rem;
      line-height: 1.43;
      letter-spacing: -0.224px;
      transition: background var(--transition-fast), color var(--transition-fast), border-color var(--transition-fast);
    }}
    .pill-link:hover {{
      text-decoration: underline;
    }}
    .pill-link.filled {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      text-decoration: none;
    }}
    .dark-hero .pill-link {{
      color: var(--accent-dark-link);
      border-color: rgba(255, 255, 255, 0.32);
    }}
    .dark-hero .pill-link.filled {{
      color: white;
      border-color: var(--accent);
    }}
    .settings-stack {{
      display: grid;
      gap: 0;
      background: var(--panel);
      border-radius: var(--radius-xl);
      overflow: hidden;
      box-shadow: var(--shadow-lg);
    }}
    .settings-section {{
      display: grid;
      grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
      gap: 32px;
      align-items: start;
      padding: 40px;
      border-bottom: 1px solid var(--border);
    }}
    .settings-section:last-child {{
      border-bottom: 0;
    }}
    .light-section {{
      background: var(--panel);
      box-shadow: none;
    }}
    .dark-section {{
      background: var(--bg-dark);
      color: #ffffff;
      box-shadow: none;
    }}
    .section-copy {{
      display: grid;
      gap: 14px;
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
      padding: 0;
      border-radius: 0;
      background: transparent;
    }}
    .dark-panel {{
      background: transparent;
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
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-lg);
    }}
    form {{ display: grid; gap: 12px; }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 0.88rem;
      line-height: 1.5;
      letter-spacing: 0;
      color: var(--muted);
    }}
    input,
    textarea {{
      width: 100%;
      border-radius: 11px;
      border: 1px solid var(--border);
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: white;
      transition:
        border-color var(--transition-fast),
        box-shadow var(--transition-fast),
        background var(--transition-fast);
    }}
    input::placeholder,
    textarea::placeholder {{
      color: #7f8ea3;
    }}
    input:focus-visible,
    textarea:focus-visible,
    button:focus-visible,
    .nav-link:focus-visible,
    .clear-link:focus-visible,
    summary:focus-visible {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(0, 113, 227, 0.2);
    }}
    textarea {{
      resize: vertical;
      min-height: 52px;
      max-height: 180px;
    }}
    button {{
      border: 1px solid transparent;
      border-radius: 999px;
      padding: 8px 18px;
      font: inherit;
      font-size: 1.06rem;
      font-weight: 400;
      line-height: 2.41;
      color: white;
      background: var(--accent);
      cursor: pointer;
      transition:
        transform var(--transition-fast),
        box-shadow var(--transition-fast),
        background var(--transition-fast),
        border-color var(--transition-fast);
      box-shadow: none;
    }}
    button:hover {{
      background: #0077ed;
    }}
    button.secondary {{
      background: var(--accent-2);
      color: white;
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
    .utility-strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0;
      margin-bottom: 20px;
      background: white;
      border-radius: 8px;
      box-shadow: var(--shadow-md);
      overflow: hidden;
    }}
    .utility-strip.compact {{
      grid-template-columns: repeat(6, minmax(0, 1fr));
      margin-bottom: 16px;
      box-shadow: none;
      border: 1px solid var(--border);
    }}
    .utility-item {{
      display: grid;
      gap: 8px;
      padding: 18px 20px;
      border-right: 1px solid var(--border);
    }}
    .utility-item:last-child {{
      border-right: 0;
    }}
    .utility-item span {{
      color: var(--muted);
      font-size: 0.75rem;
      line-height: 1.33;
      letter-spacing: -0.12px;
      text-transform: uppercase;
    }}
    .utility-item strong {{
      color: var(--ink);
      font-size: 1rem;
      line-height: 1.19;
      letter-spacing: 0.231px;
    }}
    .meta-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #fafafc;
      color: rgba(0, 0, 0, 0.8);
    }}
    .meta-pill.subtle {{
      color: var(--muted);
      background: transparent;
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
      border-radius: var(--radius-xl);
      padding: 20px;
      box-shadow: none;
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
      box-shadow: var(--shadow-lg);
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
      padding-bottom: 10px;
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
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 8px 4px 18px;
      border-bottom: 1px solid var(--border);
    }}
    .chat-topbar-copy {{
      display: grid;
      gap: 8px;
    }}
    .chat-topbar-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .sidebar-toggle {{
      min-height: 42px;
      padding: 0 16px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--ink);
      font-weight: 400;
      box-shadow: none;
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
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease, box-shadow 120ms ease;
      width: 100%;
      text-align: left;
      box-shadow: none;
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
      box-shadow: none;
    }}
    .session-delete:hover {{
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.24);
      background: rgba(255, 255, 255, 0.08);
    }}
    .session-link:hover {{
      transform: none;
      border-color: rgba(255, 255, 255, 0.24);
      background: rgba(255, 255, 255, 0.08);
    }}
    .session-link.active {{
      border-color: var(--accent);
      background: rgba(0, 113, 227, 0.18);
    }}
    .session-title {{
      font-size: 1rem;
      font-weight: 600;
      line-height: 1.35;
      letter-spacing: -0.01em;
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
      gap: 16px;
      align-content: start;
      min-height: 0;
      max-height: calc(100vh - 280px);
      overflow-y: auto;
      padding: 8px 0;
      border-radius: var(--radius-lg);
      border: 0;
      background: transparent;
      box-shadow: none;
    }}
    .bubble {{
      max-width: min(860px, 100%);
      border-radius: 8px;
      padding: 16px 18px;
      box-shadow: var(--shadow-md);
      border: 1px solid transparent;
    }}
    .user-bubble {{
      justify-self: end;
      background: var(--accent);
      color: white;
    }}
    .assistant-bubble {{
      justify-self: start;
      background: white;
      border-color: transparent;
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
      color: var(--ink);
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
      color: var(--accent-link);
    }}
    .composer {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
      padding: 16px 0 0;
      border-radius: var(--radius-lg);
      background: transparent;
      border-top: 1px solid var(--border);
      border-left: 0;
      border-right: 0;
      border-bottom: 0;
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
      transition:
        border-color var(--transition-fast),
        color var(--transition-fast),
        background var(--transition-fast);
    }}
    .clear-link:hover {{
      color: var(--ink);
      border-color: rgba(0, 113, 227, 0.35);
    }}
    .empty-chat {{
      display: grid;
      justify-items: center;
      gap: 14px;
      padding: 56px 28px;
      border: 1px dashed var(--border);
      border-radius: 12px;
      color: var(--muted);
      background: var(--surface);
      text-align: center;
    }}
    .empty-chat h2 {{
      font-size: 1.6rem;
    }}
    .empty-chat-actions {{
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 10px;
    }}
    .details-panel {{
      margin-top: 14px;
    }}
    .field-hint {{
      margin: -4px 0 14px;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.5;
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
      letter-spacing: 0;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 0.75rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
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
      box-shadow: none;
    }}
    .ghost-light:hover {{
      background: rgba(255, 255, 255, 0.06);
    }}
    @media (prefers-reduced-motion: reduce) {{
      *,
      *::before,
      *::after {{
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
      }}
    }}
    @media (max-width: 640px) {{
      .shell {{ padding: 18px 14px 30px; }}
      .card {{ border-radius: 16px; }}
      .topbar {{
        align-items: flex-start;
        flex-direction: column;
        border-radius: 18px;
        padding: 10px 14px;
      }}
      .hero-panel {{ grid-template-columns: 1fr; padding: 22px; gap: 20px; }}
      .chat-status-bar {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .hero-summary {{
        padding-left: 0;
        padding-top: 12px;
        border-left: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.18);
      }}
      .light-hero .hero-summary {{
        border-top-color: rgba(0, 0, 0, 0.12);
      }}
      h1 {{ max-width: none; font-size: clamp(2rem, 10vw, 2.8rem); }}
      .utility-strip {{ grid-template-columns: 1fr 1fr; }}
      .utility-strip.compact {{ grid-template-columns: 1fr 1fr; }}
      .composer {{ grid-template-columns: 1fr; }}
      .composer-actions {{ grid-template-columns: 1fr; grid-auto-flow: row; justify-content: stretch; }}
      .chat-topbar {{ align-items: flex-start; flex-direction: column; }}
      .chat-topbar-actions {{ width: 100%; justify-content: space-between; }}
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
        box-shadow: var(--shadow-md);
      }}
      .chat-stream {{ max-height: none; min-height: 360px; padding: 18px; }}
      .meta-pill {{ width: 100%; justify-content: center; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">MemPalace Local QA</div>
      <nav class="nav">
        <a class="{chat_active}" href="/">聊天</a>
        <a class="{settings_active}" href="/settings-page">设置</a>
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
                STATE.last_status = f"已打开会话：{STATE.get_current_session().get('title', session_id)}"
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
            STATE.last_status = "聊天记录已清空。"
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
                    raise ValueError("缺少会话 ID。")
                if not STATE.delete_session(session_id):
                    raise ValueError("未找到会话。")
                session = STATE.get_current_session()
                STATE.last_status = f"会话已删除，当前会话：{session.get('title', '新会话')}"
                STATE.last_logs = ""
                STATE.clear_error()
                self._redirect("/")
                return
            if self.path == "/switch-session":
                session_id = (form.get("session_id") or [""])[0].strip()
                if not session_id:
                    raise ValueError("缺少会话 ID。")
                if not STATE.switch_session(session_id):
                    raise ValueError("未找到会话。")
                session = STATE.get_current_session()
                STATE.last_status = f"已打开会话：{session.get('title', session_id)}"
                STATE.last_logs = ""
                STATE.clear_error()
                self._redirect("/")
                return
            if self.path == "/new-session":
                session = STATE.create_session(directory=STATE.current_dir, wing=STATE.current_wing)
                STATE.last_status = f"已新建会话：{session['title']}"
                STATE.clear_error()
                self._redirect("/")
                return
            if self.path == "/index":
                directory = (form.get("directory") or [""])[0].strip()
                if not directory:
                    raise ValueError("请填写目录路径。")
                previous_wing = STATE.current_wing
                STATE.current_dir = directory
                STATE.current_wing = directory_to_wing(directory)
                if STATE.current_wing != previous_wing:
                    reset_chat_state()
                session = STATE.ensure_session()
                session["directory"] = STATE.current_dir
                session["wing"] = STATE.current_wing
                if session.get("title") == "新会话":
                    session["title"] = Path(STATE.current_dir).name or "新会话"
                session["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                STATE.save_chat_history()
                STATE.last_status = f"已完成索引：{Path(directory).expanduser().resolve()}"
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
                    if session.get("title") == "新会话":
                        session["title"] = Path(STATE.current_dir).name or "新会话"
                    STATE.save_chat_history()
                    STATE.last_status = f"已选择目录：{selected}"
                    STATE.clear_error()
                self._redirect("/settings-page")
                return
            elif self.path == "/settings":
                STATE.model_base_url = (form.get("base_url") or [STATE.model_base_url])[0].strip()
                STATE.model_api_key = (form.get("api_key") or [""])[0].strip()
                STATE.model_name = (form.get("model") or [""])[0].strip()
                retrieval_k_raw = (form.get("retrieval_k") or [str(STATE.retrieval_k)])[0].strip()
                STATE.retrieval_k = max(1, min(50, int(retrieval_k_raw or "5")))
                history_raw = (form.get("max_history_turns") or [str(STATE.max_history_turns)])[0].strip()
                STATE.max_history_turns = max(0, min(12, int(history_raw or "4")))
                STATE.save_settings()
                STATE.last_status = "模型设置已保存。"
                STATE.clear_error()
                self._redirect("/settings-page")
                return
            elif self.path == "/test-model":
                if not STATE.model_base_url:
                    raise ValueError("请先配置模型接口地址。")
                models = list_models(
                    base_url=STATE.model_base_url,
                    api_key=STATE.model_api_key,
                )
                if STATE.model_name:
                    if STATE.model_name in models:
                        STATE.last_status = f"模型连接正常，已找到配置的模型：{STATE.model_name}"
                    else:
                        preview = ", ".join(models[:10]) if models else "无"
                        STATE.last_status = (
                            f"连接正常，但未找到已配置的模型“{STATE.model_name}”。"
                            f"可用模型：{preview}"
                        )
                else:
                    preview = ", ".join(models[:10]) if models else "无"
                    STATE.last_status = f"连接正常。可用模型：{preview}"
                STATE.last_logs = ""
                STATE.clear_error()
                self._redirect("/settings-page")
                return
            elif self.path == "/clear-index":
                if not STATE.current_wing:
                    raise ValueError("请先选择目录。")
                deleted = clear_directory_index(STATE.current_wing)
                STATE.last_status = f"已从分区 {STATE.current_wing} 清空 {deleted} 个抽屉。"
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
            self._send_json({"error": "请输入问题。"}, status=400)
            return
        if not STATE.current_dir:
            self._send_json({"error": "请先为目录建立索引。"}, status=400)
            return
        if not STATE.model_name:
            self._send_json({"error": "请先配置模型。"}, status=400)
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
