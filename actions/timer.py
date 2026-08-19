"""
In-process timers, countdown alarms, and a stopwatch for Jarvis.

These live only for the lifetime of the current process (unlike ``reminder.py``,
which schedules OS tasks). They are useful for "set a 10 minute timer",
"wake me in 5 minutes", or "start a stopwatch". When a timer fires, the
background monitor in ``main.py`` injects a ``[TIMER_ALERT]`` into the live
session and plays a short tone (Windows) so the user is actually alerted.

State is held in a module-level singleton (``get_manager``) so the tool
handler and the background poller share the same data without global coupling.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from typing import Any


_DURATION_RE = re.compile(
    r"(?:(\d+)\s*h(?:ours?|rs?)?)?\s*"
    r"(?:(\d+)\s*m(?:in(?:ute)?s?)?)?\s*"
    r"(?:(\d+)\s*s(?:ec(?:ond)?s?)?)?",
    re.IGNORECASE,
)


def parse_duration(text: str | float | int) -> float | None:
    """Parse a duration into seconds.

    Accepts numbers (treated as minutes), or strings like "5m", "10 minutes",
    "1h30m", "90s", "1 hour 30 minutes". Returns ``None`` when nothing parses.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        val = float(text)
        return val * 60.0 if val > 0 else None

    text = str(text).strip().lower()
    if not text:
        return None

    # Pure number => minutes
    if re.fullmatch(r"\d+(\.\d+)?", text):
        val = float(text)
        return val * 60.0 if val > 0 else None

    m = _DURATION_RE.search(text)
    if not m:
        return None
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    total = h * 3600 + mi * 60 + s
    return total if total > 0 else None


def play_alert_tone() -> None:
    """Play a short three-beep alert. Best-effort; silent on failure."""
    try:
        import platform

        if platform.system() == "Windows":
            import winsound

            for freq in (880, 1100, 1320):
                winsound.Beep(freq, 220)
                time.sleep(0.09)
    except Exception:
        pass


class TimerEntry:
    __slots__ = ("id", "label", "ends_at", "duration")

    def __init__(self, tid: int, label: str, duration: float, ends_at: float):
        self.id = tid
        self.label = label
        self.duration = duration
        self.ends_at = ends_at

    def remaining(self) -> float:
        return max(0.0, self.ends_at - time.monotonic())


class TimerManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timers: dict[int, TimerEntry] = {}
        self._seq = 0
        self._stopwatch_start: float | None = None

    # ── Timers ────────────────────────────────────────────────────────────

    def start_timer(self, duration_seconds: float, label: str = "Timer") -> str:
        label = (label or "Timer").strip() or "Timer"
        with self._lock:
            self._seq += 1
            tid = self._seq
            ends = time.monotonic() + duration_seconds
            self._timers[tid] = TimerEntry(tid, label, duration_seconds, ends)
        mm = int(duration_seconds // 60)
        ss = int(duration_seconds % 60)
        dur = f"{mm} min {ss:02d}s" if mm else f"{ss}s"
        return f"Timer '{label}' set for {dur} (id {tid})."

    def cancel_timer(self, identifier: str) -> str:
        identifier = str(identifier or "").strip().lower()
        with self._lock:
            # Match by id or by label (case-insensitive prefix)
            matches = [
                t for t in self._timers.values()
                if str(t.id) == identifier or t.label.lower() == identifier
                or t.label.lower().startswith(identifier)
            ]
            if not matches:
                return "No matching timer found to cancel."
            for t in matches:
                del self._timers[t.id]
        names = ", ".join(f"{t.label} (id {t.id})" for t in matches)
        return f"Cancelled timer(s): {names}."

    def list_timers(self) -> str:
        with self._lock:
            if not self._timers:
                return "No active timers."
            lines = []
            for t in sorted(self._timers.values(), key=lambda x: x.ends_at):
                rem = int(t.remaining())
                mm, ss = divmod(rem, 60)
                lines.append(f"- {t.label} (id {t.id}): {mm}m {ss:02d}s left")
            return "Active timers:\n" + "\n".join(lines)

    def check_due(self) -> list[str]:
        """Return labels of timers that have elapsed, and remove them."""
        now = time.monotonic()
        due: list[str] = []
        with self._lock:
            for tid in [t.id for t in self._timers.values() if t.ends_at <= now]:
                due.append(self._timers.pop(tid).label)
        return due

    # ── Stopwatch ────────────────────────────────────────────────────────

    def start_stopwatch(self) -> str:
        with self._lock:
            if self._stopwatch_start is not None:
                return "Stopwatch is already running."
            self._stopwatch_start = time.monotonic()
        return "Stopwatch started."

    def stop_stopwatch(self) -> str:
        with self._lock:
            if self._stopwatch_start is None:
                return "Stopwatch is not running."
            elapsed = time.monotonic() - self._stopwatch_start
            self._stopwatch_start = None
        mm, ss = divmod(int(elapsed), 60)
        hh, mm = divmod(mm, 60)
        if hh:
            return f"Stopwatch stopped at {hh}h {mm}m {ss:02d}s."
        return f"Stopwatch stopped at {mm}m {ss:02d}s."


_manager: TimerManager | None = None


def get_manager() -> TimerManager:
    global _manager
    if _manager is None:
        _manager = TimerManager()
    return _manager


def timer(parameters: dict, player: Any = None) -> str:
    """Tool handler for the ``timer`` function declaration."""
    params = parameters or {}
    action = str(params.get("action", "start")).lower().strip() or "start"
    mgr = get_manager()

    if action == "list":
        return mgr.list_timers()

    if action == "cancel":
        ident = params.get("label") or params.get("id") or ""
        if not ident:
            return "Specify a timer label or id to cancel."
        return mgr.cancel_timer(ident)

    if action == "stopwatch":
        return mgr.start_stopwatch() if params.get("command") != "stop" else mgr.stop_stopwatch()

    if action in ("start", "set"):
        seconds = parse_duration(
            params.get("duration_minutes") if params.get("duration_minutes") is not None
            else params.get("duration")
        )
        if seconds is None:
            seconds = parse_duration(params.get("label"))
        if seconds is None:
            return "I need a duration — e.g. '5 minutes', '30s', or '1h'."
        if seconds > 24 * 3600:
            return "That's over 24 hours — please use a reminder instead."
        return mgr.start_timer(seconds, params.get("label", "Timer"))

    if action == "stopwatch_stop":
        return mgr.stop_stopwatch()

    return "Unknown timer action. Use start, cancel, list, or stopwatch."
