"""
Retry decorator with exponential backoff and jitter.

Addresses the "Error Handling & Retries" gap: network/API calls currently
crash or swallow exceptions. Wrap a fragile call so transient failures
(timeouts, 5xx, connection resets) are retried with increasing delays
instead of surfacing to the user.

Example::

    from core.retry import retry

    @retry(on_exceptions=(requests.RequestException,), tries=4, delay=0.5, backoff=2.0)
    def fetch(url):
        return requests.get(url, timeout=10).json()
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Type, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def retry(
    on_exceptions: tuple[type[BaseException], ...] = (Exception,),
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
    log: logging.Logger | None = None,
) -> Callable[[F], F]:
    """Retry ``fn`` up to ``tries`` times on ``on_exceptions``.

    Waits ``delay * backoff ** (attempt-1)`` seconds between attempts, plus
    uniform random ``jitter`` to avoid thundering-herd retries.  The final
    failure (after all tries) is re-raised so callers can fall back.
    """
    if tries < 1:
        raise ValueError("tries must be >= 1")
    log = log or logger

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except on_exceptions as exc:  # noqa: PERF203 - intentional linear retries
                    attempt += 1
                    if attempt >= tries:
                        log.error(
                            "retry: %s failed after %d attempt(s): %s",
                            fn.__name__, attempt, exc,
                        )
                        raise
                    wait = delay * (backoff ** (attempt - 1)) + random.uniform(0, jitter)
                    log.warning(
                        "retry: %s attempt %d/%d failed (%s); retrying in %.2fs",
                        fn.__name__, attempt, tries, exc, wait,
                    )
                    time.sleep(wait)

        return wrapper  # type: ignore[return-value]

    return decorator
