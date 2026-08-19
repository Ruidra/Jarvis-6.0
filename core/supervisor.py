"""
Crash-resilient supervisor for Jarvis.

Wraps the main run loop (or any long-lived ``target``) so that if it crashes,
the supervisor restarts it — up to ``max_restarts`` times within
``window_seconds`` — after a backoff delay.  Includes a heartbeat health check:
if the target hasn't reported alive for ``health_timeout`` seconds, it is
considered stuck and restarted.

Example::

    from core.supervisor import Supervisor
    sup = Supervisor(target=main_loop, max_restarts=5)
    sup.start()          # blocks, runs target in a managed thread
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(
        self,
        target: Callable[[], None],
        max_restarts: int = 5,
        window_seconds: float = 300.0,
        restart_delay: float = 2.0,
        health_timeout: float = 0.0,
    ) -> None:
        self.target = target
        self.max_restarts = max_restarts
        self.window_seconds = window_seconds
        self.restart_delay = restart_delay
        self.health_timeout = health_timeout  # 0 = disabled
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._heartbeat = time.time()
        self._lock = threading.Lock()
        self.restart_count = 0

    def heartbeat(self) -> None:
        with self._lock:
            self._heartbeat = time.time()

    def _health_stale(self) -> bool:
        if not self.health_timeout:
            return False
        with self._lock:
            return time.time() - self._heartbeat > self.health_timeout

    def _run_once(self) -> None:
        try:
            self.target()
        except Exception as exc:  # noqa: BLE001 - supervisor's whole job is to catch this
            logger.error("Supervised target crashed: %s", exc)

    def _monitor(self) -> None:
        restarts: list[float] = []
        while not self._stop.is_set():
            if self._health_stale():
                logger.warning("Health check stale — restarting target.")
                # the thread is daemonic; start a fresh one
            t = threading.Thread(target=self._run_once, daemon=True, name="supervised")
            t.start()
            t.join()
            if self._stop.is_set():
                break
            now = time.time()
            restarts = [r for r in restarts if now - r < self.window_seconds]
            restarts.append(now)
            self.restart_count = len(restarts)
            if len(restarts) > self.max_restarts:
                logger.error("Max restarts (%d) exceeded — giving up.", self.max_restarts)
                break
            logger.info("Restarting supervised target in %.1fs (count=%d)", self.restart_delay, len(restarts))
            self._stop.wait(self.restart_delay)

    def start(self, blocking: bool = True) -> None:
        if blocking:
            self._monitor()
        else:
            self._thread = threading.Thread(target=self._monitor, daemon=True, name="supervisor")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
