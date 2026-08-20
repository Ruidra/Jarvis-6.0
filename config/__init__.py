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


## ── Camera (vision / face recognition only — no hand tracking) ─────────
_cfg = get_config()

CAMERA_INDEX: int = _cfg.get("camera_index", 0)
CAMERA_WIDTH: int = _cfg.get("camera_width", 1280)
CAMERA_HEIGHT: int = _cfg.get("camera_height", 720)

## ── Wake system: clap (microphone) + spoken wake phrase ────────────────
## Flow:  clap  ➜  JARVIS arms (beep)  ➜  say "wake up"  ➜  JARVIS listens.
## Everything is microphone-based; the camera is never used to wake JARVIS.
CLAP_ENABLED: bool = _cfg.get("clap_enabled", True)
## 1.0 = normal. Raise (1.4) if your claps are missed, lower (0.7) if it
## triggers on door slams / keyboard noise.
CLAP_SENSITIVITY: float = _cfg.get("clap_sensitivity", 1.0)
## How many claps are needed (2 = clap twice — far fewer false triggers).
CLAP_COUNT: int = int(_cfg.get("clap_count", 2))
## Max seconds between the first and last clap of the pattern.
CLAP_WINDOW: float = _cfg.get("clap_window", 1.2)
## Seconds of silence enforced after a successful clap pattern.
CLAP_COOLDOWN: float = _cfg.get("clap_cooldown", 1.5)

## Seconds JARVIS stays armed waiting for the wake phrase after a clap.
WAKE_TIMEOUT: float = _cfg.get("wake_timeout", 12.0)
## Phrases that finish the wake-up. "wake up" is the primary one.
WAKE_WORDS: list[str] = _cfg.get("wake_words", [
    "wake up", "jarvis", "hey jarvis", "wake up jarvis", "jarvis wake up",
])
## True  → clap first, then say the phrase (default, avoids random wake-ups).
## False → saying the phrase alone is enough, no clap needed.
WAKE_REQUIRE_CLAP: bool = _cfg.get("wake_require_clap", True)
## Play a short confirmation tone when the clap arms JARVIS.
WAKE_BEEP: bool = _cfg.get("wake_beep", True)


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
