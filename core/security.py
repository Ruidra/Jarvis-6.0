"""
Security utilities for Jarvis: secrets-at-rest encryption and password auth.

Addresses two gaps from the review:

* **Hardcoded / plaintext API keys** — :class:`SecretVault` encrypts the
  config file on disk with Fernet (AES-128 in CBC with HMAC).  The symmetric
  key lives in ``config/.vault_key`` (chmod 600) and is generated on first
  use; ``migrate_plaintext_to_vault()`` moves an existing plaintext
  ``api_keys.json`` into an encrypted store.

* **No user authentication** — ``hash_password`` / ``verify_password`` give
  PBKDF2-HMAC-SHA256 password verification so the dashboard can require a real
  passphrase instead of accepting any string as the session key.

Everything is optional and backwards-compatible: if no vault key exists, the
plaintext ``api_keys.json`` is still used.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def safe_read_config(filename: str = "api_keys.json") -> dict[str, Any]:
    """Safely read a config JSON file, returning {} if missing or corrupt.

    Replaces dozens of bare ``open(path)`` calls that crash when the file
    doesn't exist — a CRITICAL issue from the security audit.
    """
    try:
        config_path = get_base_dir() / "config" / filename
        if not config_path.exists():
            logger.warning("Config file %s not found — using defaults", filename)
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Config file %s is invalid JSON: %s", filename, exc)
        return {}
    except Exception as exc:
        logger.error("Failed to read %s: %s", filename, exc)
        return {}


def safe_write_config(data: dict[str, Any], filename: str = "api_keys.json") -> bool:
    """Safely write a config JSON file with atomic write."""
    import tempfile
    config_path = get_base_dir() / "config" / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp", dir=config_path.parent,
            delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_path = f.name
        os.replace(tmp_path, config_path)
        return True
    except Exception as exc:
        logger.error("Failed to write %s: %s", filename, exc)
        return False


class SecretVault:
    """Encrypt/decrypt JSON or string secrets with a machine-local Fernet key."""

    def __init__(self, key_path: str | Path | None = None) -> None:
        self.key_path = Path(key_path) if key_path else (get_base_dir() / "config" / ".vault_key")
        self._fernet: Fernet | None = None

    def _get_fernet(self) -> Fernet:
        if self._fernet is None:
            if self.key_path.exists():
                key = self.key_path.read_bytes().strip()
            else:
                key = Fernet.generate_key()
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                self.key_path.write_bytes(key)
                try:
                    os.chmod(self.key_path, 0o600)
                except OSError:  # pragma: no cover - non-POSIX
                    pass
                logger.info("Generated new vault key at %s", self.key_path)
            self._fernet = Fernet(key)
        return self._fernet

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        return self._get_fernet().encrypt(json.dumps(data, ensure_ascii=False).encode()).decode()

    def decrypt_dict(self, token: str) -> dict[str, Any]:
        try:
            raw = self._get_fernet().decrypt(token.encode())
        except InvalidToken as exc:
            raise ValueError("Vault token is invalid or was encrypted with a different key") from exc
        return json.loads(raw)

    def encrypt_value(self, value: str) -> str:
        return self._get_fernet().encrypt(value.encode()).decode()

    def decrypt_value(self, token: str) -> str:
        try:
            return self._get_fernet().decrypt(token.encode()).decode()
        except InvalidToken as exc:
            raise ValueError("Vault token is invalid") from exc


def migrate_plaintext_to_vault(
    plaintext_path: str | Path | None = None,
    vault: SecretVault | None = None,
) -> bool:
    """Encrypt a plaintext ``api_keys.json`` into ``api_keys.enc``.

    The original file is NOT deleted (safe, reversible).  Returns True if a
    migration was performed, False if there was nothing to do.
    """
    src = Path(plaintext_path) if plaintext_path else (get_base_dir() / "config" / "api_keys.json")
    dst = src.with_suffix(".enc")
    vault = vault or SecretVault()
    if not src.exists() or dst.exists():
        return False
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Cannot read plaintext keys for migration: %s", exc)
        return False
    dst.write_text(vault.encrypt_dict(data), encoding="utf-8")
    logger.info("Migrated plaintext keys -> %s (original left in place)", dst)
    return True


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Return ``(salt_b64, hash_b64)`` using PBKDF2-HMAC-SHA256 (200k iterations)."""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return base64.b64encode(salt).decode(), base64.b64encode(dk).decode()


def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    """Constant-time verification of ``password`` against stored ``(salt, hash)``."""
    try:
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except Exception:  # noqa: BLE001 - any decode error means invalid input
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(dk, expected)


class EncryptedJsonStore:
    """A :class:`~core.json_store.JsonStore` whose contents are Fernet-encrypted at rest.

    Used to encrypt ``memory/long_term.json`` and other sensitive state so the
    assistant's knowledge of the user is not stored in plaintext.
    """

    def __init__(self, path: str | Path | None = None, vault: SecretVault | None = None) -> None:
        self.path = Path(path) if path else (get_base_dir() / "memory" / "long_term.enc.json")
        self.vault = vault or SecretVault()

    def read(self, default: Any = None) -> Any:
        if not self.path.exists():
            return default
        try:
            token = self.path.read_text(encoding="utf-8")
            return self.vault.decrypt_dict(token)
        except Exception as exc:  # noqa: BLE001
            logger.error("EncryptedJsonStore read failed: %s", exc)
            return default

    def write(self, data: Any) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.vault.encrypt_dict(data), encoding="utf-8")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("EncryptedJsonStore write failed: %s", exc)
            return False

    def migrate_from_plaintext(self, plaintext_path: str | Path) -> bool:
        """Encrypt an existing plaintext JSON file into this store. Returns True if done."""
        src = Path(plaintext_path)
        if not src.exists() or self.path.exists():
            return False
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Cannot read plaintext for migration: %s", exc)
            return False
        ok = self.write(data)
        if ok:
            logger.info("Encrypted memory migrated -> %s (original left in place)", self.path)
        return ok
