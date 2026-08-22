"""
Retry decorator with exponential backoff and jitter.

Addresses the "Error Handling & Retries" gap: network/API calls currently
crash or swallow exceptions. Wrap a fragile call so transient failures
(timeouts, 5xx, connection resets) are retried with increasing delays
instead of surfacing to the user.

Retrying is only useful for *transient* failures. An exhausted API quota, a
revoked key, or a malformed request will fail identically on every attempt, so
retrying just multiplies the latency and the log noise —
``logs/jarvis.log`` recorded 26 such wasted attempts against
``_gemini_search`` for a quota error that could never have succeeded within the
0.5s/1.0s backoff window. :func:`is_transient` classifies these, and the
decorator fails fast on terminal errors.

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
import re
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

# Substrings that mark a failure as permanent for the lifetime of this call.
# Matched case-insensitively against str(exc), because the google-genai SDK
# reports these as generic ClientError/ServerError with the detail in the
# message rather than as distinct exception classes.
_TERMINAL_MARKERS = (
    "resource_exhausted",         # quota / rate limit ceiling reached
    "quota",
    "exceeded your current quota",
    "permission_denied",
    "unauthenticated",
    "api key not valid",
    "api_key_invalid",
    "invalid_argument",           # malformed request — identical every time
    "failed_precondition",
    "not_found",
    "unsupported",
)

# A server-supplied wait hint, e.g. "retryDelay": "27s" or Retry-After: 27.
_RETRY_HINT = re.compile(
    r"(?:retry[-_]?delay|retry[-_]?after)\D{0,8}?(\d+(?:\.\d+)?)\s*(m?s)?",
    re.IGNORECASE,
)


def is_transient(exc: BaseException) -> bool:
    """True if retrying ``exc`` could plausibly succeed.

    Conservative in the useful direction: anything unrecognised is treated as
    transient, so this never turns a currently-retried error into a hard
    failure. Only the explicitly terminal cases in :data:`_TERMINAL_MARKERS`
    are refused.
    """
    text = str(exc).lower()
    # 429 is ambiguous: a short-term rate limit is worth retrying, but a
    # daily-quota RESOURCE_EXHAUSTED is not. The marker check below decides.
    return not any(m in text for m in _TERMINAL_MARKERS)


def retry_after_hint(exc: BaseException) -> float | None:
    """Extract a server-suggested wait, in seconds, if the error carries one."""
    m = _RETRY_HINT.search(str(exc))
    if not m:
        return None
    try:
        value = float(m.group(1))
    except (TypeError, ValueError):
        return None
    if (m.group(2) or "").lower() == "ms":
        value /= 1000.0
    # Ignore absurd hints so a bad parse cannot stall Jarvis for an hour.
    return value if 0 < value <= 120 else None


def retry(
    on_exceptions: tuple[type[BaseException], ...] = (Exception,),
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
    log: logging.Logger | None = None,
    give_up: Callable[[BaseException], bool] | None = None,
    respect_retry_after: bool = True,
) -> Callable[[F], F]:
    """Retry ``fn`` up to ``tries`` times on ``on_exceptions``.

    Waits ``delay * backoff ** (attempt-1)`` seconds between attempts, plus
    uniform random ``jitter`` to avoid thundering-herd retries.  The final
    failure (after all tries) is re-raised so callers can fall back.

    Args:
        give_up: predicate returning True when an exception is terminal and
            must not be retried. Defaults to ``not is_transient(exc)``, so
            quota/auth errors fail on the first attempt.
        respect_retry_after: honor a server-supplied ``retryDelay`` /
            ``Retry-After`` hint in place of the computed backoff.
    """
    if tries < 1:
        raise ValueError("tries must be >= 1")
    log = log or logger
    _give_up = give_up if give_up is not None else (lambda e: not is_transient(e))

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except on_exceptions as exc:  # noqa: PERF203 - intentional linear retries
                    attempt += 1
                    if _give_up(exc):
                        log.warning(
                            "retry: %s failed with a non-retryable error on "
                            "attempt %d; not retrying: %s",
                            fn.__name__, attempt, exc,
                        )
                        raise
                    if attempt >= tries:
                        log.error(
                            "retry: %s failed after %d attempt(s): %s",
                            fn.__name__, attempt, exc,
                        )
                        raise
                    wait = delay * (backoff ** (attempt - 1)) + random.uniform(0, jitter)
                    if respect_retry_after:
                        hinted = retry_after_hint(exc)
                        if hinted is not None:
                            wait = hinted
                    log.warning(
                        "retry: %s attempt %d/%d failed (%s); retrying in %.2fs",
                        fn.__name__, attempt, tries, exc, wait,
                    )
                    time.sleep(wait)

        return wrapper  # type: ignore[return-value]

    return decorator
