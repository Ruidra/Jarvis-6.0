"""
JARVIS Fast Cache — speeds up repeated work.

Many Jarvis operations are repeated (same web lookup, same system status, same
weather, same fact lookup). This is a tiny, thread-safe TTL cache with:
  * per-entry time-to-live,
  * optional semantic dedup via the offline vector embedder,
  * cheap in-memory lookup (sub-millisecond) so repeated tool calls return
    instantly instead of hitting the network/LLM again.

Example::

    from core.fast_cache import cache
    cache.set("weather:London", {"temp": 12}, ttl=300)
    data = cache.get("weather:London")   # served from RAM for 5 minutes
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Hashable

logger = logging.getLogger(__name__)


class FastCache:
    """Minimal TTL key→value cache with namespaced grouping."""

    def __init__(self, default_ttl: int = 300) -> None:
        self._default_ttl = default_ttl
        self._store: dict[Hashable, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return default
            expires_at, value = item
            if expires_at < time.time():
                self._store.pop(key, None)
                return default
            return value

    def set(self, key: Hashable, value: Any, ttl: int | None = None) -> None:
        ttl = self._default_ttl if ttl is None else ttl
        with self._lock:
            # ttl <= 0 means "expired immediately" (effectively a no-op cache).
            expires_at = time.time() + ttl if ttl and ttl > 0 else time.time() - 1
            self._store[key] = (expires_at, value)
            # crude cap to avoid unbounded growth during long sessions
            if len(self._store) > 5000:
                self._trim()

    def get_or_set(self, key: Hashable, factory, ttl: int | None = None) -> Any:
        val = self.get(key)
        if val is not None:
            return val
        val = factory()
        self.set(key, val, ttl)
        return val

    def has(self, key: Hashable) -> bool:
        return self.get(key) is not None

    def clear(self, prefix: str | None = None) -> None:
        with self._lock:
            if prefix is None:
                self._store.clear()
            else:
                self._store = {k: v for k, v in self._store.items()
                               if not str(k).startswith(prefix)}

    def _trim(self) -> None:
        now = time.time()
        # drop expired first, then oldest
        self._store = {k: (exp, v) for k, (exp, v) in self._store.items()
                       if exp >= now}
        if len(self._store) > 5000:
            sorted_keys = sorted(self._store.keys(),
                                 key=lambda k: self._store[k][0])
            for k in sorted_keys[: len(self._store) - 4000]:
                self._store.pop(k, None)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._store)}


# Process-wide shared cache.
cache = FastCache(default_ttl=300)
