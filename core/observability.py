"""
Observability for Jarvis — lightweight metrics + dashboard route.

A process-wide metrics registry (counters/gauges) fed by the event bus and the
core loop, plus an optional FastAPI ``/metrics`` route that attaches to the
existing dashboard app.  Decouples telemetry from the rotating logs added in
``core/logging_setup.py``.

Example::

    from core.observability import metrics
    metrics.inc("requests")
    metrics.set("latency_ms", 42)
    print(metrics.snapshot())
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()
        self.started = time.time()

    def inc(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + by

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    # ── Cost / token tracking ───────────────────────────────────────────────
    def add_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0,
                   cost_usd: float = 0.0, model: str = "") -> None:
        """Accumulate LLM token usage + estimated cost for the cost dashboard."""
        with self._lock:
            self._counters["llm_prompt_tokens"] = (
                self._counters.get("llm_prompt_tokens", 0) + prompt_tokens)
            self._counters["llm_completion_tokens"] = (
                self._counters.get("llm_completion_tokens", 0) + completion_tokens)
            self._counters["llm_calls"] = self._counters.get("llm_calls", 0) + 1
            prev_cost = self._gauges.get("est_cost_usd", 0.0)
            self._gauges["est_cost_usd"] = round(prev_cost + cost_usd, 6)
            if model:
                self._gauges["llm_model"] = model

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }


metrics = Metrics()


def attach_to_app(app, path: str = "/metrics") -> bool:
    """Add a ``GET {path}`` JSON route to a FastAPI/Starlette ``app``.

    Returns False if the framework isn't available or the route exists.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except Exception as exc:  # noqa: BLE001
        logger.warning("observability: FastAPI not available (%s)", exc)
        return False
    if not isinstance(app, FastAPI) or any(getattr(r, "path", "") == path for r in app.routes):
        return False

    @app.get(path)
    def _metrics():
        return JSONResponse(metrics.snapshot())

    logger.info("observability: attached %s route", path)
    return True


# Auto-feed a few core events into metrics when the bus is used.
def _feed(event) -> None:  # pragma: no cover - wired at runtime
    if event.type.startswith("speech"):
        metrics.inc("speech_events")
    elif event.type.startswith("tool"):
        metrics.inc("tool_calls")
    elif event.type.startswith("error"):
        metrics.inc("errors")


try:
    from core.event_bus import bus

    bus.subscribe("speech.transcript", _feed)
    bus.subscribe("tool.call", _feed)
    bus.subscribe("error", _feed)
except Exception:  # noqa: BLE001 - bus optional at import time
    pass
