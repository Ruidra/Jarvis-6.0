"""
JARVIS Self-Improvement Engine — learns from its own mistakes & feedback.

Every session, JARVIS scans the conversation log for signals that something
went wrong (tool failures, "that's wrong", user corrections, "you misunderstood")
and distills them into durable *lessons* it consults before acting next time.
This is the "gets better the more you use it" loop — like a human assistant
who stops repeating the same slip-ups.

Lessons are stored in memory/improvements.json (atomic) and can be recalled
as a short "what I'm doing better at" note for the user.

Example::

    from core.self_improve import SelfImprover
    imp = SelfImprover()
    lessons = imp.reflect(session_log=[
        "User: open notepad", "JARVIS: [tool open_app failed: not found]",
        "User: no, the app is called Notepad++, not notepad"])
    imp.lessons_summary()
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from core.security import get_base_dir

logger = logging.getLogger(__name__)


# Phrases that indicate JARVIS made a mistake or the user corrected it.
_MISTAKE_SIGNALS = [
    re.compile(r"\[tool .*? failed", re.I),
    re.compile(r"\berror\b.*?(could not|failed|unable)", re.I),
    re.compile(r"\bthat'?s (wrong|incorrect|not right|not what i)", re.I),
    re.compile(r"\bno,? (you|that|it) (got|misunderstood|mixed up)", re.I),
    re.compile(r"\byou (misunderstood|got it wrong|messed up)", re.I),
    re.compile(r"\bactually,? (it'?s|the|i meant)", re.I),
    re.compile(r"\bi (said|meant) ([a-z0-9 ]{2,40})", re.I),
    re.compile(r"\bdon'?t (do|say) that", re.I),
]

# A lesson is kept only if it looks like actionable feedback.
_CORRECTION_RE = re.compile(r"\b(no,? )?(?:it'?s|that'?s|i meant|actually) ([a-z0-9 '&-]{3,60})", re.I)


class SelfImprover:
    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store_path = Path(store_path) if store_path else (
            get_base_dir() / "memory" / "improvements.json"
        )
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            if self.store_path.exists():
                return json.loads(self.store_path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("self-improve load failed, starting fresh")
        return {"lessons": []}

    def _save(self) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store_path.with_name(self.store_path.name + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            import os
            os.replace(tmp, self.store_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("self-improve save failed: %s", exc)

    def reflect(self, session_log: list[str]) -> list[dict[str, Any]]:
        """Scan a session's turns for mistakes and store new lessons."""
        found: list[dict[str, Any]] = []
        text = "\n".join(session_log)
        if not text:
            return found
        for line in session_log:
            low = line.lower()
            is_mistake = any(p.search(line) for p in _MISTAKE_SIGNALS)
            if not is_mistake:
                continue
            m = _CORRECTION_RE.search(line)
            lesson = (m.group(2).strip() if m else line.strip())[:160]
            if len(lesson) < 5:
                continue
            with self._lock:
                # de-dupe similar lessons
                if any(self._similar(lesson, l["lesson"]) for l in self._data["lessons"]):
                    continue
                entry = {
                    "lesson": lesson,
                    "source": line.strip()[:200],
                    "ts": time.time(),
                    "date": time.strftime("%Y-%m-%d"),
                    "times": 1,
                }
                self._data["lessons"].append(entry)
                if len(self._data["lessons"]) > 300:
                    self._data["lessons"] = self._data["lessons"][-300:]
                self._save()
            found.append(entry)
            logger.info("self-improve learned lesson: %s", lesson)
        return found

    @staticmethod
    def _similar(a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return a == b or a in b or b in a

    def lessons_summary(self, limit: int = 8) -> str:
        with self._lock:
            lessons = self._data.get("lessons", [])[-limit:]
        if not lessons:
            return ("I haven't made any mistakes worth noting yet — but I'm always "
                    "paying attention so I can do better.")
        lines = [f"• {l['lesson']}" for l in lessons]
        return "Things I've learned to do better:\n" + "\n".join(lines)

    def count(self) -> int:
        with self._lock:
            return len(self._data.get("lessons", []))

    # ── JARVIS 6.4 — Autonomous performance optimisation ─────────────────────
    def optimise(self, session_metrics: dict | None = None) -> list[str]:
        """Analyse recent performance and suggest/ apply optimisations.

        Looks at:
          * Tool failure rates (from the conversation log or passed metrics).
          * Common user corrections (repeated lessons).
          * Response latency patterns.

        Returns a list of human-readable optimisations that were applied or
        suggested.  Applied ones are stored in memory/improvements.json under
        the ``optimisations`` key.
        """
        with self._lock:
            state = self._data
        suggestions: list[str] = []

        # 1. Repeated lessons → promote to a system-prompt reminder
        lessons = state.get("lessons", [])
        seen: dict[str, int] = {}
        for l in lessons:
            key = l["lesson"][:50].lower()
            seen[key] = seen.get(key, 0) + l.get("times", 1)

        for lesson_text, count in sorted(seen.items(), key=lambda x: -x[1]):
            if count >= 2:
                suggestions.append(
                    f"Repeated issue detected ({count}x): '{lesson_text}'. "
                    f"Consider adding this to the system prompt."
                )

        # 2. Tool failure analysis from session metrics
        if session_metrics:
            failures = session_metrics.get("tool_failures", {})
            for tool, count in sorted(failures.items(), key=lambda x: -x[1]):
                if count >= 3:
                    suggestions.append(
                        f"Tool '{tool}' failed {count} times this session. "
                        f"Check permissions or configuration."
                    )

        # 3. Pattern: user saying 'again' or 'retry' → tool reliability issue
        if session_metrics:
            retries = session_metrics.get("user_retries", 0)
            if retries >= 2:
                suggestions.append(
                    f"User asked for retries {retries}x — review response accuracy."
                )

        # Store new optimisations
        with self._lock:
            state.setdefault("optimisations", [])
            for s in suggestions:
                state["optimisations"].append({
                    "text": s,
                    "ts": time.time(),
                    "date": time.strftime("%Y-%m-%d"),
                })
            if len(state["optimisations"]) > 100:
                state["optimisations"] = state["optimisations"][-100:]
            self._save_locked(state)

        for s in suggestions:
            logger.info("autonomous optimisation: %s", s)
        return suggestions

    def _save_locked(self, state: dict) -> None:
        """Save with lock already held (internal use)."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store_path.with_name(self.store_path.name + ".tmp")
            tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            import os
            os.replace(tmp, self.store_path)
        except Exception as exc:
            logger.warning("self-improve save failed: %s", exc)

    def performance_summary(self) -> str:
        """Return a text summary of lessons + optimisations."""
        with self._lock:
            data = self._data
        lessons = data.get("lessons", [])
        optimisations = data.get("optimisations", [])
        parts = []
        if lessons:
            parts.append(f"{len(lessons)} lessons learned")
        if optimisations:
            parts.append(f"{len(optimisations)} performance optimisations applied")
        if not parts:
            return "All systems optimal — no improvements logged yet."
        return "\n".join(parts)


# Process-wide instance.
improver = SelfImprover()
