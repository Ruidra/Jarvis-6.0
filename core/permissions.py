"""
Tool permission scopes for JARVIS.

Lets the assistant grant/maintain *granular* capability scopes instead of the
all-or-nothing ``god_mode`` flag.  Every tool declares which scope(s) it needs;
a call is only executed if the active scope set permits it.  Scopes are stored in
``config/api_keys.json`` under ``permission_scopes`` (default: all enabled).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Canonical scopes — a tool is gated by its required scope.
SCOPE_WEB          = "web"            # browsing, search, youtube, flights, web search
SCOPE_MESSAGING    = "messaging"      # send_message
SCOPE_FILES        = "files"          # file_controller, file_processor, uploads
SCOPE_SYSTEM       = "system"         # computer_settings, system_status, reminders
SCOPE_DESKTOP_CTRL = "desktop_control"  # mouse/keyboard/computer_control (powerful)
SCOPE_VISION       = "vision"         # screen/camera
SCOPE_CODE         = "code"           # code_helper, dev_agent
SCOPE_MEMORY       = "memory"         # save/recall/forget memory, notes
SCOPE_AGENT        = "agent"          # multi-step planning / autonomous actions

ALL_SCOPES = [
    SCOPE_WEB, SCOPE_MESSAGING, SCOPE_FILES, SCOPE_SYSTEM, SCOPE_DESKTOP_CTRL,
    SCOPE_VISION, SCOPE_CODE, SCOPE_MEMORY, SCOPE_AGENT,
]

# tool name -> required scope
_TOOL_SCOPE = {
    "open_app": SCOPE_DESKTOP_CTRL,
    "web_search": SCOPE_WEB,
    "research": SCOPE_WEB,
    "weather_report": SCOPE_WEB,
    "send_message": SCOPE_MESSAGING,
    "reminder": SCOPE_SYSTEM,
    "youtube_video": SCOPE_WEB,
    "screen_process": SCOPE_VISION,
    "close_camera": SCOPE_VISION,
    "face_recognize": SCOPE_VISION,
    "computer_settings": SCOPE_SYSTEM,
    "browser_control": SCOPE_WEB,
    "file_controller": SCOPE_FILES,
    "desktop_control": SCOPE_DESKTOP_CTRL,
    "code_helper": SCOPE_CODE,
    "dev_agent": SCOPE_CODE,
    "computer_control": SCOPE_DESKTOP_CTRL,
    "game_updater": SCOPE_SYSTEM,
    "flight_finder": SCOPE_WEB,
    "manage_monitor": SCOPE_AGENT,
    "file_processor": SCOPE_FILES,
    "save_memory": SCOPE_MEMORY,
    "recall_memory": SCOPE_MEMORY,
    "forget_memory": SCOPE_MEMORY,
    "audit_memory": SCOPE_MEMORY,
    "notes": SCOPE_MEMORY,
    "timer": SCOPE_SYSTEM,
    "image_gen": SCOPE_WEB,
    "undo_last": SCOPE_AGENT,
    "agent": SCOPE_AGENT,
    "run_command": SCOPE_SYSTEM,
    # JARVIS 6.1 — emotional intelligence + self-learning
    "emotion": SCOPE_MEMORY,
    "motivate": SCOPE_MEMORY,
    "learn": SCOPE_MEMORY,
    "persona": SCOPE_MEMORY,
    "focus": SCOPE_SYSTEM,
    "goals": SCOPE_MEMORY,
    "discover": SCOPE_MEMORY,
}

_CONFIG_PATH = None  # set lazily to BASE_DIR/config/api_keys.json


def _config_path() -> Path:
    global _CONFIG_PATH
    if _CONFIG_PATH is None:
        from pathlib import Path as _P
        try:
            import sys as _sys
            if getattr(_sys, "frozen", False):
                _CONFIG_PATH = Path(_sys.executable).parent / "config" / "api_keys.json"
            else:
                _CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "api_keys.json"
        except Exception:
            _CONFIG_PATH = Path("config/api_keys.json")
    return _CONFIG_PATH


def _load_scopes() -> set[str] | None:
    """Return the configured scope set, or None if not configured yet."""
    try:
        p = _config_path()
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        scopes = data.get("permission_scopes")
        if scopes is None:
            return None
        return set(scopes)
    except Exception:
        return None


def active_scopes() -> set[str]:
    """Effective scopes: configured set, or ALL_SCOPES if unset/first run."""
    s = _load_scopes()
    return s if s is not None else set(ALL_SCOPES)


def set_scopes(scopes: Iterable[str]) -> None:
    p = _config_path()
    data = {}
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data["permission_scopes"] = sorted(set(scopes) & set(ALL_SCOPES))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def is_allowed(tool_name: str) -> bool:
    """Whether the active scope set permits executing ``tool_name``."""
    scope = _TOOL_SCOPE.get(tool_name)
    if scope is None:
        return True  # unknown tools are not gated
    return scope in active_scopes()


def required_scope(tool_name: str) -> str | None:
    return _TOOL_SCOPE.get(tool_name)
