"""
Atomic, thread-safe JSON storage for Jarvis.

Consolidates the duplicated "load file -> mutate one field -> write file"
pattern that Code Review item #3 flagged in ``config_manager.py``,
``actions/reminder.py``, ``actions/background_monitor.py`` and parts of
``dashboard/server.py``.  Every read/write goes through one lock and writes
to a temp file then ``os.replace`` so a crash mid-write can never corrupt
the live file.

Windows caveat that drove the hardening here: ``os.replace`` briefly makes the
destination unopenable, so a *concurrent reader* gets ``PermissionError`` even
though the file is perfectly intact. A 12-thread stress test saw 84 such reads
out of 480. Both sides therefore retry transient sharing violations, and
:meth:`JsonStore.merge` refuses to write when it could not read an existing
file — otherwise one unlucky read would return ``{}`` and the merge would
overwrite every other key in the file (silently wiping saved API keys).

Example::

    store = JsonStore("config/api_keys.json")
    data = store.read({}) or {}
    data["gemini_api_key"] = "..."
    store.write(data)

    # or, merge a few fields without a manual read:
    store.merge({"os_system": "windows"})
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# A sentinel distinct from None so callers can tell "file is absent" (a normal
# first-run condition) from "the file exists but could not be read right now"
# (transient, and unsafe to treat as empty).
class _Unreadable:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<unreadable>"


UNREADABLE = _Unreadable()

# os.replace() fails with PermissionError (WinError 5 / WinError 32) on Windows
# when the destination is momentarily held open by another process — antivirus,
# a file indexer, the dashboard reading the same JSON, or a second Jarvis
# instance. The handle is normally released within milliseconds, so a short
# bounded retry turns a lost write into a successful one. Observed in
# logs/jarvis.log as repeated "mood journal save failed: [WinError 5]".
_REPLACE_ATTEMPTS = 8
_REPLACE_BACKOFF = 0.02  # seconds; doubles each attempt (~2.5s worst case)

# Readers hit the same transient lock, from the other side.
_READ_ATTEMPTS = 8
_READ_BACKOFF = 0.02


def read_json_strict(path: str | Path) -> Any:
    """Read a JSON file, distinguishing "absent" from "temporarily unreadable".

    Returns the decoded value, ``None`` if the file does not exist, or
    :data:`UNREADABLE` if it exists but could not be read/parsed. Callers doing
    read-modify-write **must** check for :data:`UNREADABLE` and abort, rather
    than treating it as empty and clobbering the file.
    """
    p = Path(path)
    delay = _READ_BACKOFF
    last_exc: Exception | None = None

    for attempt in range(1, _READ_ATTEMPTS + 1):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except PermissionError as exc:
            # Almost always os.replace() on another thread/process holding the
            # destination for a few milliseconds. Worth retrying.
            last_exc = exc
            if attempt == _READ_ATTEMPTS:
                break
            time.sleep(delay)
            delay *= 2
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Genuinely malformed content — retrying cannot help.
            logger.warning("read_json: %s contains invalid JSON: %s", p, exc)
            return UNREADABLE
        except OSError as exc:
            last_exc = exc
            break

    logger.warning("read_json: could not read %s: %s", p, last_exc)
    return UNREADABLE


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file. Returns ``default`` if missing, corrupt, or locked.

    Convenience wrapper over :func:`read_json_strict` for callers that only
    display or read data. If you intend to write the result back, use
    :func:`read_json_strict` so a transient failure cannot be mistaken for an
    empty file.
    """
    value = read_json_strict(path)
    if value is None or value is UNREADABLE:
        return default
    return value


def atomic_write_json(path: str | Path, data: Any) -> bool:
    """Write JSON atomically via a temp file + ``os.replace``. Returns success.

    Safe against concurrent writers: the temp file name is unique per
    process/thread, so two writers can never scribble over each other's
    partially-written temp file. The final ``os.replace`` is retried with
    backoff to survive transient Windows sharing violations.
    """
    p = Path(path)
    tmp: Path | None = None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Serialise before creating the temp file so an un-encodable payload
        # cannot leave a stray temp file behind.
        payload = json.dumps(data, indent=2, ensure_ascii=False)

        # Unique temp name in the destination directory (same filesystem, so
        # os.replace stays atomic). A shared ".tmp" suffix was the original bug:
        # concurrent writers collided on one path.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent)
        )
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            # Durability: without fsync a crash right after replace() can leave
            # a zero-length file on some filesystems.
            os.fsync(fh.fileno())

        last_exc: OSError | None = None
        delay = _REPLACE_BACKOFF
        for attempt in range(1, _REPLACE_ATTEMPTS + 1):
            try:
                os.replace(tmp, p)
                tmp = None  # replace consumed it; nothing to clean up
                return True
            except PermissionError as exc:
                last_exc = exc
                if attempt == _REPLACE_ATTEMPTS:
                    break
                logger.debug(
                    "atomic_write_json: %s locked (attempt %d/%d), retrying in %.2fs",
                    p, attempt, _REPLACE_ATTEMPTS, delay,
                )
                time.sleep(delay)
                delay *= 2
            except OSError as exc:
                last_exc = exc
                break

        logger.error("atomic_write_json failed for %s: %s", p, last_exc)
        return False
    except (OSError, TypeError, ValueError) as exc:
        # TypeError/ValueError: the caller handed us data json cannot encode.
        # That is a real bug in the caller, so say so plainly rather than
        # reporting a generic write failure.
        logger.error("atomic_write_json failed for %s: %s", p, exc)
        return False
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                logger.debug("could not remove temp file %s", tmp)


class JsonStore:
    """A small object wrapper around :func:`read_json` / :func:`atomic_write_json`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def read(self, default: Any = None) -> Any:
        with self._lock:
            return read_json(self.path, default)

    def write(self, data: Any) -> bool:
        with self._lock:
            return atomic_write_json(self.path, data)

    def merge(self, mapping: dict[str, Any]) -> bool:
        """Merge ``mapping`` into the existing dict (creating it if absent).

        Aborts without writing if the file exists but could not be read. The
        alternative — treating an unreadable file as ``{}`` — would replace the
        whole file with just ``mapping``, destroying every other key. On a
        config file that means silently losing the user's saved API keys.
        """
        with self._lock:
            existing = read_json_strict(self.path)
            if existing is UNREADABLE:
                logger.error(
                    "merge into %s aborted: file exists but is unreadable; "
                    "refusing to overwrite it with partial data", self.path,
                )
                return False
            data = existing if isinstance(existing, dict) else {}
            if existing is not None and not isinstance(existing, dict):
                logger.warning(
                    "merge into %s: existing content is %s, not an object — replacing it",
                    self.path, type(existing).__name__,
                )
            data.update(mapping)
            return atomic_write_json(self.path, data)


    def update(self, **fields: Any) -> bool:
        """Convenience form of :meth:`merge` taking keyword fields."""
        return self.merge(dict(fields))
