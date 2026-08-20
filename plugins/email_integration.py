"""
Jarvis Plugin — Email (SMTP send + IMAP read).

A real, working email integration that needs NO OAuth — just an account with an
"app password" (Gmail, Outlook, Yahoo all support this). Credentials live in
``config/email_config.json`` (created on first setup). Uses only the Python
standard library (smtplib / imaplib), so it works offline of any provider SDK.

Setup (one time, tell JARVIS): "set up email" — it will explain the steps, or
you can write config/email_config.json directly:
  {
    "email":        "you@gmail.com",
    "app_password": "16-char-app-password",
    "smtp_host":    "smtp.gmail.com",
    "smtp_port":    587,
    "imap_host":    "imap.gmail.com",
    "imap_port":    993
  }

Triggers (spoken): "send email", "read my email", "check inbox", "unread mail".

Args:
  action : send | inbox | unread | setup   (default: inbox)
  to     : recipient address (send)
  subject: subject (send)
  body   : message body (send)
  limit  : number of messages to fetch (inbox/unread)
"""

from __future__ import annotations

import email
import imaplib
import logging
import smtplib
import ssl
from email.header import decode_header
from email.mime.text import MIMEText
from pathlib import Path

from core.json_store import read_json

logger = logging.getLogger("jarvis.plugin.email")

PLUGIN = {
    "name": "email",
    "description": (
        "Send and read email via standard SMTP/IMAP using an app password (no OAuth). "
        "Use when the user says 'send an email', 'email X', 'read my email', "
        "'check my inbox', or 'any unread mail'."
    ),
    "triggers": ["send email", "read email", "check inbox", "unread mail", "my email"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action":  {"type": "STRING",
                        "description": "send | inbox | unread | setup (default: inbox)"},
            "to":      {"type": "STRING", "description": "Recipient address (send)."},
            "subject": {"type": "STRING", "description": "Subject line (send)."},
            "body":    {"type": "STRING", "description": "Message body (send)."},
            "limit":   {"type": "INTEGER", "description": "Messages to fetch (inbox/unread)."},
        },
        "required": [],
    },
}

_SETUP_HINT = (
    "To enable email I need an app password (NOT your normal password). "
    "Create config/email_config.json with: email, app_password, smtp_host, "
    "smtp_port, imap_host, imap_port. For Gmail: smtp.gmail.com:587, "
    "imap.gmail.com:993, and generate an app password at myaccount.google.com "
    "→ Security → App passwords. I'll keep it stored locally and encrypted-at-rest "
    "only by file permissions."
)


def _cfg_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "email_config.json"


def _load_cfg() -> dict:
    return read_json(_cfg_path(), {}) or {}


def _decode(value: str) -> str:
    try:
        out = []
        for b, enc in decode_header(value):
            if isinstance(b, bytes):
                out.append(b.decode(enc or "utf-8", "replace"))
            else:
                out.append(str(b))
        return "".join(out)
    except Exception:  # noqa: BLE001
        return str(value)


def _send(to: str, subject: str, body: str) -> str:
    cfg = _load_cfg()
    if not cfg.get("email") or not cfg.get("app_password"):
        return _SETUP_HINT
    if not to:
        return "Who should I send it to? Give me a recipient address."
    msg = MIMEText(body or "", "plain", "utf-8")
    msg["From"] = cfg["email"]
    msg["To"] = to
    msg["Subject"] = subject or "(no subject)"
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg.get("smtp_host", "smtp.gmail.com"),
                          int(cfg.get("smtp_port", 587))) as s:
            s.starttls(context=ctx)
            s.login(cfg["email"], cfg["app_password"])
            s.send_message(msg)
        return f"📤 Email sent to {to}."
    except Exception as e:  # noqa: BLE001
        logger.error("email send failed: %s", e)
        return f"Failed to send email: {e}"


def _fetch(unread_only: bool, limit: int) -> str:
    cfg = _load_cfg()
    if not cfg.get("email") or not cfg.get("app_password"):
        return _SETUP_HINT
    limit = max(1, min(int(limit or 5), 20))
    try:
        mail = imaplib.IMAP4_SSL(
            cfg.get("imap_host", "imap.gmail.com"), int(cfg.get("imap_port", 993))
        )
        mail.login(cfg["email"], cfg["app_password"])
        mail.select("INBOX")
        crit = "(UNSEEN)" if unread_only else "ALL"
        _, data = mail.search(None, crit)
        ids = (data[0] or b"").split()
        if not ids:
            return "No matching messages." if unread_only else "Your inbox is empty."
        ids = ids[-limit:]
        out = []
        for num in ids:
            _, d = mail.fetch(num, "(RFC822)")
            raw = d[0][1]
            if isinstance(raw, bytes):
                m = email.message_from_bytes(raw)
            else:
                m = email.message_from_string(str(raw))
            out.append(
                f"• From: {_decode(m.get('From',''))}\n"
                f"  Subject: {_decode(m.get('Subject',''))}"
            )
        mail.logout()
        label = "Unread" if unread_only else "Latest"
        return f"📬 {label} messages:\n" + "\n".join(out)
    except Exception as e:  # noqa: BLE001
        logger.error("email fetch failed: %s", e)
        return f"Failed to read email: {e}"


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    action = (args.get("action") or "inbox").lower().strip()

    if action == "setup":
        return _SETUP_HINT
    if action == "send":
        return _send(args.get("to"), args.get("subject"), args.get("body"))
    if action == "unread":
        return _fetch(True, args.get("limit"))
    # default: inbox
    return _fetch(False, args.get("limit"))
