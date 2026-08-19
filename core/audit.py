"""
Audit log + undo for JARVIS tool calls.

Every tool invocation is recorded to ``logs/audit.jsonl`` (one JSON object per
line) with enough context to (a) review what the assistant has done and (b)
*undo* the most recent destructive action.  Destructive tools (file delete, file
move, system-settings changes, reminders, memory edits) attach an ``undo``
payload describing how to revert.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_AUDIT_PATH = None


def _path() -> Path:
    global _AUDIT_PATH
    if _AUDIT_PATH is None:
        try:
            import sys as _sys
            if getattr(_sys, "frozen", False):
                base = Path(_sys.executable).parent
            else:
                base = Path(__file__).resolve().parents[1]
            _AUDIT_PATH = base / "logs" / "audit.jsonl"
        except Exception:
            _AUDIT_PATH = Path("logs/audit.jsonl")
    return _AUDIT_PATH


def record(
    tool: str,
    args: dict[str, Any] | None = None,
    result: str = "",
    ok: bool = True,
    error: str = "",
    *,
    correlation_id: str = "",
    undo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an audit entry; returns the written record (incl. its ``id``)."""
    entry = {
        "id": f"{int(time.time()*1000)}-{id(tool) % 100000:05d}",
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": tool,
        "args": _redact(args or {}),
        "result": (result or "")[:500],
        "ok": ok,
        "error": error[:500] if error else "",
        "correlation_id": correlation_id,
        "undo": undo,
        "destructive": undo is not None,
    }
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover - logging edge case
        logger.warning("audit write failed: %s", e)
    return entry


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Best-effort redaction of obviously sensitive fields."""
    out = {}
    for k, v in args.items():
        if any(s in k.lower() for s in ("password", "token", "key", "secret", "pin")):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def last_undoable() -> dict[str, Any] | None:
    """Return the most recent entry that carries an undo payload, or None."""
    try:
        p = _path()
        if not p.exists():
            return None
        last = None
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("undo"):
                    last = e
        return last
    except Exception:
        return None


def mark_undone(entry_id: str) -> None:
    """Stamp the given audit entry as undone (best-effort, for the log)."""
    p = _path()
    if not p.exists():
        return
    tmp = p.with_suffix(".tmp")
    try:
        with p.open("r", encoding="utf-8") as f, tmp.open("w", encoding="utf-8") as out:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    out.write(line)
                    continue
                if e.get("id") == entry_id:
                    e["undone"] = True
                out.write(json.dumps(e, ensure_ascii=False) + "\n")
        tmp.replace(p)
    except Exception:
        pass


def recent(limit: int = 25, only_destructive: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        p = _path()
        if not p.exists():
            return out
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if only_destructive and not e.get("destructive"):
                    continue
                out.append(e)
        return out[-limit:]
    except Exception:
        return out
