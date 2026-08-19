# config/__init__.py
import json, os, platform
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "api_keys.json"
_VAULT_PATH = Path(__file__).parent / "api_keys.enc"

def _platform_os() -> str:
    """Auto-detect OS when config file is absent."""
    return {"Windows": "windows", "Darwin": "mac", "Linux": "linux"}.get(
        platform.system(), "linux"
    )

def get_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_os() -> str:
    """Returns: 'windows' | 'mac' | 'linux'"""
    return get_config().get("os_system", _platform_os()).lower()

def is_windows() -> bool: return get_os() == "windows"
def is_mac()     -> bool: return get_os() == "mac"
def is_linux()   -> bool: return get_os() == "linux"


## ── Hand control + clap activation constants ──────────────────────────
_cfg = get_config()

CAMERA_INDEX: int = _cfg.get("camera_index", 0)
CAMERA_WIDTH: int = _cfg.get("camera_width", 1280)
CAMERA_HEIGHT: int = _cfg.get("camera_height", 720)
HAND_DETECTION_CONFIDENCE: float = _cfg.get("hand_detection_confidence", 0.65)
HAND_TRACKING_CONFIDENCE: float = _cfg.get("hand_tracking_confidence", 0.65)
CLAP_SENSITIVITY: float = _cfg.get("clap_sensitivity", 1.0)
CLAP_COOLDOWN: float = _cfg.get("clap_cooldown", 2.0)
WAKE_TIMEOUT: float = _cfg.get("wake_timeout", 10.0)
HAND_DEBUG: bool = _cfg.get("hand_debug", False)
WAKE_WORDS: list[str] = _cfg.get("wake_words", ["jarvis", "hey jarvis", "jarvis hey"])

## ── Gesture → action mapping (per-state override or global) ─────────────────
## Global: applies in any state.  Per-state: only applies when JARVIS is in that state.
## Actions: "wake" | "lock" | "interrupt" | "sleep"
GESTURE_ACTION_MAP: dict[str, dict[str, str]] = _cfg.get("gesture_action_map", {
    "OFFLINE": {"WAVE": "wake", "OPEN_PALM": "wake"},
    "LOCKED":  {"WAVE": "wake", "OPEN_PALM": "wake"},
    "READY":   {"POINT": "interrupt", "FIST": "sleep"},
    "LISTENING": {"FIST": "lock", "WAVE": "wake"},
})
## Direct gesture→action overrides (highest priority, any state):
GESTURE_DIRECT_MAP: dict[str, str] = _cfg.get("gesture_direct_map", {
    "CLAP": "wake",
})


def get_secret(key: str, default: str | None = None) -> str | None:
    """Return a secret (e.g. an API key) with transparent encryption support.

    Prefers the encrypted vault (``api_keys.enc``) when present, otherwise
    falls back to the plaintext ``api_keys.json`` so existing installs keep
    working.  Run ``core.security.migrate_plaintext_to_vault()`` to encrypt.
    """
    if _VAULT_PATH.exists():
        try:
            from core.security import SecretVault
            vault = SecretVault()
            data = vault.decrypt_dict(_VAULT_PATH.read_text(encoding="utf-8"))
            return data.get(key, default)
        except Exception:
            pass  # fall through to plaintext
    return get_config().get(key, default)
