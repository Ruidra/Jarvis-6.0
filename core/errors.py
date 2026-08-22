"""
Error handling helpers for Jarvis.

The codebase had ~436 ``except Exception: pass`` blocks. They keep Jarvis from
crashing, but they also mean a genuine bug (a ``TypeError`` from a refactor) is
indistinguishable from an expected condition (a missing optional dependency) —
both vanish without a trace. Three real bugs survived for months behind them:
the vector-memory cache never persisting, the mood journal never saving on
Windows, and ``browser_control`` being unreachable.

This module keeps the "never crash" property while making failures *visible*.

Usage — replace a silent swallow::

    # before
    try:
        risky()
    except Exception:
        pass

    # after
    with swallow("loading optional persona"):
        risky()

``swallow`` logs at DEBUG by default, so normal operation stays quiet but
``--log-level DEBUG`` reveals everything that was being hidden. Pass
``level=logging.WARNING`` when the failure is worth surfacing.

Call :func:`install_excepthook` once at startup so anything that escapes a
handler — including exceptions in threads and asyncio callbacks, which Python
otherwise prints to a stderr that may not exist in the windowed (pythonw/VBS)
launch — still reaches ``logs/jarvis.log``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import sys
import threading
from typing import Any, Callable, Iterator, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Exceptions that must never be swallowed: they signal "stop the program", not
# "this operation failed". Catching KeyboardInterrupt/SystemExit would make
# Ctrl-C and clean shutdown unreliable; catching asyncio.CancelledError (a
# BaseException since 3.8) would break task cancellation and TaskGroup unwinding.
_NEVER_SWALLOW: tuple[type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


@contextlib.contextmanager
def swallow(
    context: str = "",
    *,
    expect: tuple[type[BaseException], ...] = (Exception,),
    level: int = logging.DEBUG,
    log: logging.Logger | None = None,
    reraise: tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Suppress ``expect`` but log it, instead of discarding it silently.

    Args:
        context: short description of what was being attempted, used in the log
            message (e.g. ``"saving mood journal"``).
        expect: exception types to suppress. Narrow this when you know what can
            actually go wrong — a narrower tuple is a better bug filter.
        level: level to log at. DEBUG keeps normal runs quiet; use WARNING when
            the user would want to know.
        log: logger to use. Defaults to this module's logger; pass the calling
            module's logger so the log line names the right subsystem.
        reraise: exception types to let through even if they match ``expect``.

    ``KeyboardInterrupt``, ``SystemExit``, ``GeneratorExit`` and
    ``asyncio.CancelledError`` are always re-raised.
    """
    _log = log or logger
    try:
        yield
    except _NEVER_SWALLOW:
        raise
    except BaseException as exc:  # noqa: BLE001 - filtered below
        # CancelledError is a BaseException in 3.8+; swallowing it breaks
        # cooperative cancellation, so it always propagates.
        if isinstance(exc, asyncio.CancelledError):
            raise
        if reraise and isinstance(exc, reraise):
            raise
        if not isinstance(exc, expect):
            raise
        where = context or "operation"
        # exc_info at DEBUG only — a traceback per swallowed error at INFO
        # would flood the log for expected conditions like a missing optional
        # dependency.
        _log.log(
            level,
            "suppressed %s while %s: %s",
            type(exc).__name__,
            where,
            exc,
            exc_info=(level >= logging.WARNING),
        )


def log_exc(
    context: str = "",
    *,
    log: logging.Logger | None = None,
    level: int = logging.ERROR,
) -> None:
    """Log the exception currently being handled, with traceback.

    Call from inside an ``except`` block when you want the traceback recorded
    but intend to handle or re-raise the error yourself::

        except OSError:
            log_exc("writing cache", log=logger)
            return False
    """
    (log or logger).log(level, "error while %s", context or "operation", exc_info=True)


def guard(
    context: str = "",
    *,
    default: Any = None,
    expect: tuple[type[BaseException], ...] = (Exception,),
    level: int = logging.DEBUG,
    log: logging.Logger | None = None,
) -> Callable[[F], F]:
    """Decorator form of :func:`swallow` that returns ``default`` on failure.

    For optional/best-effort helpers whose failure must not propagate::

        @guard("reading persona", default="")
        def persona_fragment() -> str:
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _ctx = context or f"calling {fn.__name__}"
            with swallow(_ctx, expect=expect, level=level, log=log or logger):
                return fn(*args, **kwargs)
            return default

        return wrapper  # type: ignore[return-value]

    return decorator


_excepthook_installed = False


def install_excepthook(log: logging.Logger | None = None) -> None:
    """Route uncaught exceptions into the logging system. Idempotent.

    Covers three escape routes that otherwise print to a stderr which is
    ``None`` under the windowed launch (``Jarvis-Silent.vbs`` / pythonw), losing
    the traceback entirely:

      * ``sys.excepthook``        — the main thread
      * ``threading.excepthook``  — background threads (3.8+)
      * asyncio exception handler — failed tasks nobody awaited
    """
    global _excepthook_installed
    if _excepthook_installed:
        return

    _log = log or logging.getLogger("jarvis.crash")

    def _hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            # Preserve default Ctrl-C behaviour rather than logging a traceback.
            sys.__excepthook__(exc_type, exc, tb)
            return
        _log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook

    def _thread_hook(args: threading.ExceptHookArgs) -> None:  # type: ignore[name-defined]
        if issubclass(args.exc_type, SystemExit):
            return
        _log.critical(
            "Uncaught exception in thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_hook
    _excepthook_installed = True


def install_asyncio_handler(loop: Any, log: logging.Logger | None = None) -> None:
    """Route unhandled asyncio task exceptions into the log.

    Separate from :func:`install_excepthook` because it needs a running loop.
    Call once from ``JarvisLive.run`` after the loop exists.
    """
    _log = log or logging.getLogger("jarvis.crash")

    def _handler(_loop: Any, ctx: dict) -> None:
        exc = ctx.get("exception")
        msg = ctx.get("message") or "asyncio error"
        if exc is not None:
            if isinstance(exc, asyncio.CancelledError):
                return
            _log.error("asyncio: %s", msg, exc_info=exc)
        else:
            _log.error("asyncio: %s (%s)", msg, {k: v for k, v in ctx.items() if k != "message"})

    with swallow("installing asyncio exception handler", log=_log):
        loop.set_exception_handler(_handler)
