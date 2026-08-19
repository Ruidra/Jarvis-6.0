"""
Optional "god-level" wiring for Jarvis.

Call :func:`init_god_features` once at startup (e.g. from ``main.py`` after the
core session is up) to: discover plugins, start the proactive scheduler, attach
observability to the dashboard, and connect the event bus to the logs.  Every
piece is defensive — if an optional dependency is missing, the rest still loads.

This is intentionally separate from ``main.py`` so the base app boots unchanged
and god features opt-in.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def init_god_features(dashboard_app=None) -> dict:
    """Initialise the god-level subsystems. Returns a status dict."""
    status: dict[str, str] = {}

    # 1) Plugin registry + hot reload
    try:
        from core.plugin_registry import PluginRegistry

        registry = PluginRegistry()
        names = registry.discover()
        registry.start_watching()
        status["plugins"] = f"{len(names)} discovered, hot-reload on"
    except Exception as exc:  # noqa: BLE001
        status["plugins"] = f"error: {exc}"

    # 2) Proactive scheduler (autonomous tasks)
    try:
        from core.proactive_scheduler import build_default_scheduler

        sched = build_default_scheduler()
        sched.start()
        status["proactive"] = "started"
        # Watchdog: auto-restart the scheduler thread if it ever dies.
        try:
            import threading

            def _watchdog():
                while True:
                    import time as _t
                    _t.sleep(60)
                    try:
                        th = getattr(sched, "_thread", None)
                        if th is None or not th.is_alive():
                            logger.warning("Proactive scheduler down — auto-restarting")
                            sched.start()
                    except Exception:
                        pass
            _wd = threading.Thread(target=_watchdog, daemon=True, name="proactive-watchdog")
            _wd.start()
            status["proactive"] = "started + watchdog"
        except Exception as exc:  # noqa: BLE001
            status["proactive"] = f"started (no watchdog: {exc})"
    except Exception as exc:  # noqa: BLE001
        status["proactive"] = f"error: {exc}"

    # 3) Observability route
    try:
        from core.observability import attach_to_app

        if dashboard_app is not None and attach_to_app(dashboard_app):
            status["observability"] = "attached"
        else:
            status["observability"] = "metrics only"
    except Exception as exc:  # noqa: BLE001
        status["observability"] = f"error: {exc}"

    # 4) Event bus -> logs bridge
    try:
        from core.event_bus import bus

        bus.subscribe("error", lambda e: logger.error("BUS error: %s", e.data))
        status["event_bus"] = "bridged"
    except Exception as exc:  # noqa: BLE001
        status["event_bus"] = f"error: {exc}"

    logger.info("God features initialised: %s", status)
    return status
