"""
Persistent configuration storage for Jarvis L.

Reads and writes ``config/api_keys.json``, which holds the Gemini API key,
assistant/user display names, and small user preferences (e.g. whether the
morning briefing is enabled). All writes go through :func:`_update_config`,
which loads the current file, applies a partial update, and writes it back
under a lock, instead of every setter reimplementing its own
read-modify-write and error handling.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_lock = Lock()

DEFAULT_ASSISTANT_NAME = "JARVIS"
_MIN_VALID_KEY_LENGTH = 15


def get_base_dir() -> Path:
    """Return the project root, accounting for PyInstaller-frozen builds."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
CONFIG_DIR = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"


def ensure_config_dir() -> None:
    """Create the config directory if it doesn't already exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def config_exists() -> bool:
    """Return True if the config file has been created (first-run check)."""
    return CONFIG_FILE.exists()


def load_api_keys() -> dict[str, Any]:
    """Load the full config dict. Returns {} if missing or unreadable."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", CONFIG_FILE.name, e)
        return {}


def _update_config(**fields: Any) -> None:
    """Merge ``fields`` into the config file, creating it if needed."""
    ensure_config_dir()
    with _lock:
        data = load_api_keys()
        data.update(fields)
        try:
            CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            logger.error("Failed to write %s: %s", CONFIG_FILE.name, e)


def save_api_keys(gemini_api_key: str) -> None:
    """Persist the Gemini API key."""
    _update_config(gemini_api_key=gemini_api_key.strip())


def get_gemini_key() -> str | None:
    """Return the stored Gemini API key, or None if not set."""
    return load_api_keys().get("gemini_api_key")


def is_configured() -> bool:
    """Return True if an API key is present and looks plausible."""
    key = get_gemini_key()
    return bool(key and len(key) > _MIN_VALID_KEY_LENGTH)


def get_assistant_name() -> str:
    """Return the configured assistant name, defaulting to 'JARVIS'."""
    return load_api_keys().get("assistant_name") or DEFAULT_ASSISTANT_NAME


def get_user_name() -> str:
    """Return the configured user name used for addressing them."""
    return load_api_keys().get("user_name", "")


def save_assistant_config(assistant_name: str, user_name: str) -> None:
    """Persist the assistant's display name and the user's name."""
    _update_config(
        assistant_name=assistant_name.strip() or DEFAULT_ASSISTANT_NAME,
        user_name=user_name.strip(),
    )


def get_brief_enabled() -> bool:
    """Return whether the morning briefing feature is enabled (default True)."""
    return load_api_keys().get("morning_brief_enabled", True)


def save_brief_enabled(enabled: bool) -> None:
    """Persist whether the morning briefing feature is enabled."""
    _update_config(morning_brief_enabled=bool(enabled))


def get_god_mode() -> bool:
    """Return whether unrestricted 'God Mode' is enabled (default False)."""
    return bool(load_api_keys().get("god_mode", False))


def set_god_mode(enabled: bool) -> None:
    """Persist the 'God Mode' flag."""
    _update_config(god_mode=bool(enabled))
