"""
JARVIS Goals — persistent objectives that resurface naturally.

Goals are longer-lived than a todo. They're stored in memory and:
  * surfaced in the morning briefing / day check-in ("remember you wanted to X"),
  * recalled by meaning,
  * marked done when the user says so.

Storage: memory/goals.json (atomic), plus a lightweight mirror in long-term
memory via the existing memory manager when useful.

Example::

    from core.goals import goals
    goals.add("Finish the Jarvis voice upgrade", due="Friday")
    goals.list()
    goals.complete(0)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from core.security import get_base_dir

logger = logging.getLogger(__name__)


class Goals:
    def __init__(self, store_path: str | Path | None = None) -> None:
        self.path = Path(store_path) if store_path else (
            get_base_dir() / "memory" / "goals.json"
        )
        self._lock = __import__("threading").Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            if self.path.exists():
                return json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("goals load failed, starting fresh")
        return {"goals": []}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            import os
            os.replace(tmp, self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("goals save failed: %s", exc)

    def add(self, text: str, due: str = "", category: str = "goal") -> int:
        text = (text or "").strip()
        if not text:
            return -1
        with self._lock:
            item = {
                "id": len(self._data["goals"]) + 1,
                "text": text, "due": due, "category": category,
                "done": False, "created": time.strftime("%Y-%m-%d"),
            }
            self._data["goals"].append(item)
            self._save()
            return item["id"]

    def list(self, only_open: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            items = self._data.get("goals", [])
            return [g for g in items if (not only_open or not g.get("done"))]

    def complete(self, goal_id: int) -> bool:
        with self._lock:
            for g in self._data.get("goals", []):
                if g.get("id") == goal_id:
                    g["done"] = True
                    self._save()
                    return True
        return False

    def by_text(self, query: str) -> list[dict[str, Any]]:
        q = (query or "").lower()
        with self._lock:
            return [g for g in self._data.get("goals", [])
                    if q in g.get("text", "").lower()]

    def summary(self, limit: int = 6) -> str:
        open_ = self.list(only_open=True)
        if not open_:
            return "You have no open goals right now. Want to set one?"
        lines = [f"• {g['text']}" + (f" (due {g['due']})" if g.get("due") else "")
                 for g in open_[:limit]]
        return "Your open goals:\n" + "\n".join(lines)


# Process-wide instance.
goals = Goals()
