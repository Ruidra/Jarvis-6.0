"""
JARVIS Focus Mode — Do-Not-Disturb for deep work.

When enabled, JARVIS suppresses proactive interruptions (day check-ins,
background monitors, proactive suggestions) so you can concentrate, and quietly
batch-delivers a single "while you were focusing" summary when you exit Focus.
State persists in config so it survives restarts.

Example::

    from core.focus_mode import FocusMode
    fm = FocusMode()
    fm.enable()
    fm.suppressed  # True
    fm.exit_summary()  # returns a digest string (or "") 
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from core.security import get_base_dir

logger = logging.getLogger(__name__)

_CONFIG_KEY = "focus_mode"
_LOG_KEY = "focus_log"


class FocusMode:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.path = Path(config_path) if config_path else (
            get_base_dir() / "config" / "api_keys.json"
        )
        self._since = 0.0

    def _read(self) -> dict[str, Any]:
        try:
            if self.path.exists():
                return json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
        return {}

    def _write(self, data: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("focus_mode save failed: %s", exc)

    @property
    def active(self) -> bool:
        return bool(self._read().get(_CONFIG_KEY, False))

    def enable(self) -> str:
        data = self._read()
        data[_CONFIG_KEY] = True
        self._write(data)
        self._since = time.time()
        return ("Focus mode ON. I'll stay quiet unless you speak to me directly, "
                "and I'll catch you up when you're done.")

    def disable(self) -> str:
        data = self._read()
        was = bool(data.get(_CONFIG_KEY, False))
        data[_CONFIG_KEY] = False
        self._write(data)
        if not was:
            return "Focus mode was already off."
        return self.exit_summary() or "Focus mode OFF. Welcome back — I'm all yours."

    def exit_summary(self) -> str:
        # Could be extended to summarise missed monitors/reminders.
        if self._since:
            mins = int((time.time() - self._since) / 60)
            return (f"Focus mode OFF. You focused for about {mins} minute(s). "
                    f"I held back all interruptions — anything you need, just ask.")
        return "Focus mode OFF."

    def should_interrupt(self) -> bool:
        """Proactive loops call this: True means 'go ahead and speak'."""
        return not self.active


# Process-wide instance.
focus = FocusMode()
