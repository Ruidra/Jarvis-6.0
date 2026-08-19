"""
Centralized logging for Jarvis.

Replaces the scattered ``print(f"[Tag] ...")`` calls with a single,
configurable logging setup so every subsystem gets timestamps, log levels,
and (optionally) a rotating on-disk log file for debugging deployed builds.

Usage (call once, early, from main.py)::

    from core.logging_setup import setup_logging
    setup_logging(log_dir="logs")

Then in any module::

    import logging
    logger = logging.getLogger("jarvis.vision")
    logger.info("Mic started")
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from pathlib import Path

# Correlation ID shared across HUD, dashboard, and background tasks for a single
# request/turn, so logs for one action can be tied together during debugging.
CORRELATION_ID: ContextVar[str] = ContextVar("correlation_id", default="-")

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | [%(cid)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.cid = CORRELATION_ID.get()
        return True


def set_correlation_id(cid: str) -> None:
    """Set the correlation id for the current execution context."""
    CORRELATION_ID.set(cid)


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT)


def setup_logging(
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
    to_console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger once. Idempotent unless ``force=True``.

    Args:
        level: minimum level to emit (logging.INFO by default).
        log_dir: if given, a rotating ``jarvis.log`` is written here.
        to_console: also emit to stderr.
        force: reconfigure even if already configured.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    if getattr(root, "_jarvis_logging_configured", False) and not force:
        return root

    root.setLevel(level)
    # Drop any handlers a naive basicConfig() may have added earlier.
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = _make_formatter()
    _cid_filter = _CorrelationFilter()

    if to_console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        stream.addFilter(_cid_filter)
        root.addHandler(stream)

    if log_dir is not None:
        log_dir = Path(log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_dir / "jarvis.log",
                maxBytes=5_000_000,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(_cid_filter)
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - filesystem edge case
            root.warning("Could not attach file log handler: %s", exc)

    root._jarvis_logging_configured = True  # type: ignore[attr-defined]
    logging.getLogger("jarvis").info("Logging initialised (level=%s)", logging.getLevelName(level))
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``get_logger("vision")`` -> ``jarvis.vision``."""
    return logging.getLogger(f"jarvis.{name}")
