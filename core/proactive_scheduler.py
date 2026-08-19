"""
Proactive reasoning loop for Jarvis — autonomous, scheduled initiative.

Instead of only responding, Jarvis can *initiate* tasks on a schedule: a morning
briefing, a daily summary, meeting prep, background-topic checks, etc.  Each
task is a named callable registered with an interval; a background thread runs
them and emits results on the event bus (so the HUD / Web UI / TTS can react).

Example::

    from core.proactive_scheduler import ProactiveScheduler
    sched = ProactiveScheduler()
    sched.add_task("daily_summary", interval_s=86400, fn=summarize_day)
    sched.start()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from core.event_bus import bus

logger = logging.getLogger(__name__)


class ProactiveScheduler:
    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def add_task(self, name: str, fn: Callable[[], Any], interval_s: float = 3600.0) -> None:
        with self._lock:
            self._tasks[name] = {"fn": fn, "interval": interval_s, "last": 0.0}

    def remove_task(self, name: str) -> None:
        with self._lock:
            self._tasks.pop(name, None)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            with self._lock:
                due = {
                    n: t
                    for n, t in self._tasks.items()
                    if now - t["last"] >= t["interval"]
                }
            for name, t in due.items():
                try:
                    result = t["fn"]()
                    bus.emit("proactive.task", {"name": name, "result": result}, source="proactive")
                    logger.info("Proactive task '%s' ran.", name)
                except Exception as exc:  # noqa: BLE001 - a bad task must not kill the loop
                    logger.error("Proactive task '%s' failed: %s", name, exc)
                    bus.emit("error", {"where": f"proactive.{name}", "msg": str(exc)}, source="proactive")
                t["last"] = now
            self._stop.wait(1.0)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="proactive-scheduler")
        self._thread.start()
        logger.info("Proactive scheduler started with %d task(s).", len(self._tasks))

    def stop(self) -> None:
        self._stop.set()


# ── Default autonomous tasks (wire to memory/LLM as desired) ──────────────────
def _default_daily_summary() -> str:
    try:
        from memory.memory_manager import load_memory
        mem = load_memory()
        projects = mem.get("projects", {})
        return f"Projects tracked: {', '.join(projects.keys()) or 'none'}"
    except Exception as exc:  # noqa: BLE001
        return f"(summary unavailable: {exc})"


def _default_meeting_prep() -> str:
    return "No calendar linked yet — connect a calendar tool to enable meeting prep."


def build_default_scheduler() -> ProactiveScheduler:
    """Return a scheduler pre-loaded with safe, offline-capable proactive tasks."""
    s = ProactiveScheduler()
    s.add_task("daily_summary", _default_daily_summary, interval_s=86400)
    s.add_task("meeting_prep", _default_meeting_prep, interval_s=21600)
    return s
