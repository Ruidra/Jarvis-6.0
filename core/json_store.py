"""
Atomic, thread-safe JSON storage for Jarvis.

Consolidates the duplicated "load file -> mutate one field -> write file"
pattern that Code Review item #3 flagged in ``config_manager.py``,
``actions/reminder.py``, ``actions/background_monitor.py`` and parts of
``dashboard/server.py``.  Every read/write goes through one lock and writes
to a temp file then ``os.replace`` so a crash mid-write can never corrupt
the live file.

Example::

    store = JsonStore("config/api_keys.json")
    data = store.read({}) or {}
    data["gemini_api_key"] = "..."
    store.write(data)

    # or, merge a few fields without a manual read:
    store.merge({"os_system": "windows"})
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file. Returns ``default`` if missing or corrupt."""
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("read_json failed for %s: %s", p, exc)
        return default


def atomic_write_json(path: str | Path, data: Any) -> bool:
    """Write JSON atomically via a temp file + os.replace. Returns success."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, p)
        return True
    except OSError as exc:
        logger.error("atomic_write_json failed for %s: %s", p, exc)
        return False


class JsonStore:
    """A small object wrapper around :func:`read_json` / :func:`atomic_write_json`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def read(self, default: Any = None) -> Any:
        if not self.path.exists():
            return default
        with self._lock:
            return read_json(self.path, default)

    def write(self, data: Any) -> bool:
        with self._lock:
            return atomic_write_json(self.path, data)

    def merge(self, mapping: dict[str, Any]) -> bool:
        """Merge ``mapping`` into the existing dict (creating it if absent)."""
        with self._lock:
            data = read_json(self.path, {}) or {}
            if not isinstance(data, dict):
                data = {}
            data.update(mapping)
            return atomic_write_json(self.path, data)

    def update(self, **fields: Any) -> bool:
        """Convenience form of :meth:`merge` taking keyword fields."""
        return self.merge(dict(fields))
