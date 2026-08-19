"""
Quick personal notes for Jarvis.

A lightweight, always-available scratchpad so the user can capture thoughts,
to-do items, or anything worth keeping without it polluting long-term memory.
Stored as JSON at ``<base>/memory/notes.json``.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()

CATEGORIES = ("inbox", "todo", "ideas", "people", "other")


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_NOTES_PATH = _base_dir() / "memory" / "notes.json"


def _read() -> list[dict[str, Any]]:
    if not _NOTES_PATH.exists():
        return []
    try:
        data = json.loads(_NOTES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _write(items: list[dict[str, Any]]) -> None:
    _NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _NOTES_PATH.write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def notes(parameters: dict, player: Any = None) -> str:
    """Tool handler for the ``notes`` function declaration."""
    params = parameters or {}
    action = str(params.get("action", "add")).lower().strip() or "add"

    if action == "add":
        text = (params.get("text") or "").strip()
        if not text:
            return "I need some text to save a note."
        category = str(params.get("category", "inbox")).lower().strip()
        if category not in CATEGORIES:
            category = "other"
        with _lock:
            items = _read()
            items.append({"text": text, "category": category})
            _write(items)
        return f"Saved to {category}: \"{text[:60]}\"."

    if action == "list":
        category = (params.get("category") or "").lower().strip()
        with _lock:
            items = _read()
        if category:
            items = [i for i in items if i.get("category") == category]
        if not items:
            return "No notes saved yet."
        lines = []
        for idx, item in enumerate(items, 1):
            cat = item.get("category", "other")
            lines.append(f"{idx}. [{cat}] {item.get('text', '')}")
        return "Your notes:\n" + "\n".join(lines)

    if action == "search":
        query = (params.get("query") or "").strip().lower()
        if not query:
            return "Specify a search term."
        with _lock:
            items = _read()
        hits = [i for i in items if query in (i.get("text", "") + i.get("category", "")).lower()]
        if not hits:
            return f"No notes matching '{query}'."
        lines = [f"{idx}. [{i.get('category','other')}] {i.get('text','')}"
                 for idx, i in enumerate(hits, 1)]
        return f"Notes matching '{query}':\n" + "\n".join(lines)

    if action == "delete":
        raw = str(params.get("index") or params.get("id") or "").strip()
        if not raw.isdigit():
            return "Specify the note number to delete (see the list)."
        idx = int(raw) - 1
        with _lock:
            items = _read()
            if 0 <= idx < len(items):
                removed = items.pop(idx)
                _write(items)
                return f"Deleted: \"{removed.get('text', '')[:60]}\"."
            return "No note at that number."

    return "Unknown notes action. Use add, list, search, or delete."
