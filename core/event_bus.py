"""
Event-driven core for Jarvis.

A tiny, thread-safe publish/subscribe bus that replaces ad-hoc polling.  Any
subsystem can ``emit`` an event and any other can ``subscribe``.  This is the
backbone for the proactive loop, toolchain, observability, and the Web UI —
everything communicates through typed events instead of shared globals.

Example::

    from core.event_bus import bus

    def on_speech(evt):
        print("heard:", evt.data["text"])

    bus.subscribe("speech.transcript", on_speech)
    bus.emit("speech.transcript", {"text": "jarvis, play music"})
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

EventHandler = Callable[["Event"], None]


class Event:
    __slots__ = ("type", "data", "source", "ts")

    def __init__(self, type_: str, data: Any = None, source: str = "") -> None:
        self.type = type_
        self.data = data
        self.source = source
        import time
        self.ts = time.time()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Event({self.type!r}, source={self.source!r})"


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[EventHandler]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        """Subscribe ``handler`` to ``event_type``. Returns an unsubscribe fn."""
        with self._lock:
            self._subs.setdefault(event_type, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subs.get(event_type, []).remove(handler)
                except ValueError:
                    pass

        return unsubscribe

    def emit(self, event_type: str, data: Any = None, source: str = "") -> None:
        evt = Event(event_type, data, source)
        with self._lock:
            handlers = list(self._subs.get(event_type, []))
        for h in handlers:
            try:
                h(evt)
            except Exception as exc:  # noqa: BLE001 - one bad listener must not break the bus
                logger.error("Event handler for %s failed: %s", event_type, exc)

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()


# Process-wide singleton bus.
bus = EventBus()
