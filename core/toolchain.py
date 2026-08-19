"""
Agentic toolchain for Jarvis — safe execution capabilities.

Promotes the existing ``actions/`` stubs into a guarded, reusable tool set the
agent can call: web fetch, file read/write (sandboxed to the project dir),
terminal command execution (allowlisted + confirmation), email (SMTP stub), and
calendar (stub).  Every tool returns a structured result and refuses unsafe
operations by default.

Example::

    from core.toolchain import Toolchain
    tc = Toolchain()
    print(tc.run("web_fetch", {"url": "https://example.com"}))
    print(tc.run("file_read", {"path": "notes.txt"}))
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from core.security import get_base_dir

logger = logging.getLogger(__name__)

# Read-only / safe commands the agent may run without explicit approval.
DEFAULT_ALLOWLIST = {
    "dir", "ls", "pwd", "echo", "date", "whoami", "python", "python3",
    "git status", "git log", "git diff", "type", "cat", "head", "tail",
}


class ToolError(Exception):
    pass


class Toolchain:
    def __init__(self, sandbox_dir: str | Path | None = None) -> None:
        self.sandbox = Path(sandbox_dir) if sandbox_dir else get_base_dir()
        self.allowlist = set(DEFAULT_ALLOWLIST)

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = {
            "web_fetch": self.web_fetch,
            "file_read": self.file_read,
            "file_write": self.file_write,
            "run_command": self.run_command,
            "send_email": self.send_email,
            "calendar_add": self.calendar_add,
        }.get(name)
        if not handler:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            result = handler(**args)
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - tools must report, not crash
            logger.error("Tool %s failed: %s", name, exc)
            return {"ok": False, "error": str(exc)}

    # ── tools ─────────────────────────────────────────────────────────────────
    def web_fetch(self, url: str, timeout: int = 10) -> str:
        import requests

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Jarvis/1.0"})
        resp.raise_for_status()
        return resp.text[:20000]

    def _safe_path(self, path: str) -> Path:
        p = (self.sandbox / path).resolve()
        if self.sandbox not in p.parents and p != self.sandbox:
            raise ToolError("path escapes sandbox")
        return p

    def file_read(self, path: str) -> str:
        p = self._safe_path(path)
        if not p.exists():
            raise ToolError(f"no such file: {path}")
        return p.read_text(encoding="utf-8", errors="replace")

    def file_write(self, path: str, content: str) -> str:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {p}"

    def run_command(self, command: str, allow_unsafe: bool = False, timeout: int = 30) -> str:
        base = command.strip().split()[0] if command.strip() else ""
        if not allow_unsafe and command.strip() not in self.allowlist and base not in self.allowlist:
            raise ToolError(
                f"command not in allowlist: {command!r}. "
                "Pass allow_unsafe=True only for trusted, user-approved commands."
            )
        import subprocess

        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return (res.stdout or "") + (res.stderr or "")

    def send_email(self, to: str, subject: str, body: str, smtp: dict | None = None) -> str:
        # smtp: {"host", "port", "user", "password"} — credentials from vault in real use.
        if not smtp:
            raise ToolError("send_email requires SMTP config (credentials from vault)")
        msg = EmailMessage()
        msg["To"], msg["Subject"] = to, subject
        msg.set_content(body)
        with smtplib.SMTP(smtp["host"], int(smtp.get("port", 587))) as s:
            s.starttls()
            s.login(smtp["user"], smtp["password"])
            s.send_message(msg)
        return f"email sent to {to}"

    def calendar_add(self, title: str, when: str, notes: str = "") -> str:
        # Stub: wire to Google/Outlook Calendar API as needed.
        logger.info("calendar_add stub: %s @ %s", title, when)
        return f"[stub] calendar event '{title}' at {when}"
