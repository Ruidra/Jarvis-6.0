"""
Unit tests for Jarvis core infrastructure (logging, json_store, retry,
security, plugin_manager).

Run from the project root:
    python -m pytest tests/ -q
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── logging_setup ────────────────────────────────────────────────────────────
def test_setup_logging_is_idempotent(tmp_path):
    from core import logging_setup

    importlib.reload(logging_setup)
    log_dir = tmp_path / "logs"
    logging_setup.setup_logging(level=logging.DEBUG, log_dir=log_dir)
    first_handlers = len(logging.getLogger().handlers)

    # calling again must not duplicate handlers
    logging_setup.setup_logging(level=logging.DEBUG, log_dir=log_dir, force=False)
    assert len(logging.getLogger().handlers) == first_handlers

    assert (log_dir / "jarvis.log").exists()
    logging_setup.setup_logging(force=True)  # reset for other tests


def test_get_logger_namespacing():
    from core.logging_setup import get_logger

    assert get_logger("vision").name == "jarvis.vision"


# ── json_store ───────────────────────────────────────────────────────────────
def test_json_store_roundtrip_and_atomic_write(tmp_path):
    from core.json_store import JsonStore, atomic_write_json, read_json

    p = tmp_path / "cfg.json"
    store = JsonStore(p)
    assert store.read({"x": 1}) == {"x": 1}  # default when missing

    store.write({"a": 1})
    assert store.read() == {"a": 1}

    store.merge({"b": 2})
    assert store.read() == {"a": 1, "b": 2}

    store.update(c=3)
    assert store.read() == {"a": 1, "b": 2, "c": 3}

    # standalone helpers
    atomic_write_json(tmp_path / "x.json", {"k": "v"})
    assert read_json(tmp_path / "x.json") == {"k": "v"}


def test_json_store_corrupt_file_returns_default(tmp_path):
    from core.json_store import JsonStore

    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert JsonStore(p).read("fallback") == "fallback"


# ── retry ────────────────────────────────────────────────────────────────────
def test_retry_succeeds_after_transient_failures():
    from core.retry import retry

    calls = {"n": 0}

    @retry(on_exceptions=(ValueError,), tries=3, delay=0.01, backoff=1.0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exhausting_tries():
    from core.retry import retry

    @retry(on_exceptions=(ValueError,), tries=2, delay=0.01, backoff=1.0)
    def always_fails():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        always_fails()


def test_retry_only_catches_specified_exceptions():
    from core.retry import retry

    @retry(on_exceptions=(ValueError,), tries=3, delay=0.01)
    def raises_keyerror():
        raise KeyError("not retried")

    with pytest.raises(KeyError):
        raises_keyerror()


# ── security ─────────────────────────────────────────────────────────────────
def test_secret_vault_roundtrip(tmp_path):
    from core.security import SecretVault

    key = tmp_path / ".vault_key"
    vault = SecretVault(key_path=key)
    token = vault.encrypt_dict({"gemini_api_key": "ABC123", "n": 1})
    assert "ABC123" not in token  # ciphertext, not plaintext
    assert vault.decrypt_dict(token) == {"gemini_api_key": "ABC123", "n": 1}

    # value-level helpers
    vtok = vault.encrypt_value("secret")
    assert vault.decrypt_value(vtok) == "secret"


def test_secret_vault_key_is_reused(tmp_path):
    from core.security import SecretVault

    key = tmp_path / ".vault_key"
    a = SecretVault(key_path=key).encrypt_value("x")
    b = SecretVault(key_path=key).encrypt_value("y")
    # same key -> both decryptable from a fresh vault instance
    fresh = SecretVault(key_path=key)
    assert fresh.decrypt_value(a) == "x"
    assert fresh.decrypt_value(b) == "y"


def test_password_hash_and_verify():
    from core.security import hash_password, verify_password

    salt, h = hash_password("correct horse")
    assert verify_password("correct horse", salt, h) is True
    assert verify_password("wrong", salt, h) is False
    # different salts, same password -> still verifies per-salt
    salt2, h2 = hash_password("correct horse")
    assert verify_password("correct horse", salt2, h2) is True


def test_migrate_plaintext_to_vault(tmp_path):
    from core.security import SecretVault, migrate_plaintext_to_vault

    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "api_keys.json").write_text('{"gemini_api_key": "XYZ"}', encoding="utf-8")
    vault = SecretVault(key_path=cfg / ".vault_key")

    ok = migrate_plaintext_to_vault(plaintext_path=cfg / "api_keys.json", vault=vault)
    assert ok is True
    enc = (cfg / "api_keys.enc").read_text(encoding="utf-8")
    assert "XYZ" not in enc
    assert vault.decrypt_dict(enc) == {"gemini_api_key": "XYZ"}

    # idempotent: second call finds the .enc already exists
    assert migrate_plaintext_to_vault(plaintext_path=cfg / "api_keys.json", vault=vault) is False


# ── plugin_manager ───────────────────────────────────────────────────────────
def _write_plugin(directory: Path, name: str, body: str) -> None:
    (directory / f"{name}.py").write_text(body, encoding="utf-8")


def test_plugin_discover_and_dispatch(tmp_path):
    from core.plugin_manager import PluginManager

    plugdir = tmp_path / "plugins"
    plugdir.mkdir()
    _write_plugin(
        plugdir,
        "hello",
        'PLUGIN = {"name": "hello", "triggers": ["hello", "ping"], "handler": "handle"}\n'
        'def handle(intent, args, ctx):\n'
        '    return f"hi:{intent}"\n',
    )

    pm = PluginManager(plugin_dir=plugdir)
    assert pm.discover() == ["hello"]
    assert pm.dispatch("please say hello") == "hi:please say hello"
    assert pm.dispatch("ping me") == "hi:ping me"
    assert pm.dispatch("unrelated text") is None


def test_plugin_failure_is_isolated(tmp_path):
    from core.plugin_manager import PluginManager

    plugdir = tmp_path / "plugins"
    plugdir.mkdir()
    _write_plugin(
        plugdir,
        "boom",
        'PLUGIN = {"name": "boom", "triggers": ["boom"]}\n'
        'def handle(intent, args, ctx):\n'
        '    raise RuntimeError("kaboom")\n',
    )
    pm = PluginManager(plugin_dir=plugdir)
    pm.discover()
    assert pm.dispatch("boom now") is None  # no crash, just no result


def test_plugin_skips_modules_without_manifest(tmp_path):
    from core.plugin_manager import PluginManager

    plugdir = tmp_path / "plugins"
    plugdir.mkdir()
    _write_plugin(plugdir, "no_manifest", "def handle(): pass\n")
    pm = PluginManager(plugin_dir=plugdir)
    assert pm.discover() == []
