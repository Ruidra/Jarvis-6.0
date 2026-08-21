"""
Plugin marketplace foundation for Jarvis.

Extends :class:`core.plugin_manager.PluginManager` with a persistent manifest
registry (enable/disable, metadata) and hot-reload: drop a new ``*.py`` into
``plugins/`` (or edit one) and the registry picks it up within ``poll_seconds``
without restarting Jarvis.

Example::

    from core.plugin_registry import PluginRegistry
    reg = PluginRegistry()
    reg.discover()
    reg.enable("hello")
    reg.start_watching()   # background hot-reload
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from core.plugin_manager import PluginManager, get_base_dir

logger = logging.getLogger(__name__)


class PluginRegistry:
    def __init__(self, plugin_dir: str | Path | None = None) -> None:
        self.plugin_dir = Path(plugin_dir) if plugin_dir else (get_base_dir() / "plugins")
        self.manager = PluginManager(self.plugin_dir)
        self.registry_path = self.plugin_dir / "registry.json"
        self._state: dict[str, Any] = {"enabled": {}, "installed": {}, "broken": {}}
        # Names of built-in tools (set by main.py). A plugin may not shadow one —
        # collisions are flagged BROKEN instead of silently breaking tool dispatch.
        self.core_tool_names: set[str] = set()
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._trigger_cache: dict[str, str] = {}  # intent_lower → plugin_name

    # ── registry persistence ───────────────────────────────────────────────────
    def _load_state(self) -> None:
        if self.registry_path.exists():
            try:
                self._state = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._state = {"enabled": {}, "installed": {}, "broken": {}}
        self._state.setdefault("enabled", {})
        self._state.setdefault("installed", {})
        self._state.setdefault("broken", {})

    def _save_state(self) -> None:
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self.registry_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            logger.error("registry save failed: %s", exc)

    # ── discovery ──────────────────────────────────────────────────────────────
    def discover(self) -> list[str]:
        self._load_state()
        names = self.manager.discover()
        with self._lock:
            # Pull load/validation/name-collision errors from the manager so a
            # broken plugin is recorded (and can be shown as BROKEN) rather than
            # silently dropped.
            self._state["broken"] = dict(self.manager.list_broken())
            # A plugin must not shadow a built-in tool. Flag it BROKEN and keep it
            # out of dispatch / tool declarations. First loaded plugin wins.
            for name in list(self.manager.plugins.keys()):
                if name in self.core_tool_names:
                    self._state["broken"][name] = (
                        f"Name '{name}' collides with a built-in tool - rename the plugin."
                    )
                    self.manager.plugins.pop(name, None)
            for name in names:
                if name in self.core_tool_names:
                    continue
                meta = self.manager.get(name)["meta"]
                self._state["installed"][name] = {
                    "description": meta.get("description", ""),
                    "triggers": meta.get("triggers", []),
                }
                self._state["enabled"].setdefault(name, True)
            # Clear trigger cache on reload
            self._trigger_cache.clear()
            # prune removed plugins
            for name in list(self._state["installed"]):
                if name not in self.manager.plugins:
                    self._state["installed"].pop(name, None)
                    self._state["enabled"].pop(name, None)
            self._save_state()
        return names

    def broken(self) -> dict[str, str]:
        """Return {plugin_or_file: reason} for plugins that failed to load."""
        with self._lock:
            return dict(self._state.get("broken", {}))

    def is_enabled(self, name: str) -> bool:
        with self._lock:
            return bool(self._state["enabled"].get(name, False))

    def enable(self, name: str) -> None:
        with self._lock:
            self._state["enabled"][name] = True
            self._save_state()

    def disable(self, name: str) -> None:
        with self._lock:
            self._state["enabled"][name] = False
            self._save_state()

    def dispatch(self, intent: str, args: dict | None = None, ctx: dict | None = None) -> Any:
        """Dispatch only to enabled plugins."""
        if not self.manager.plugins:
            self.discover()
        intent_l = (intent or "").lower()

        # Check cache first
        cached = self._trigger_cache.get(intent_l)
        if cached is not None:
            if not cached:
                return None
            if cached in self.manager.plugins and self.is_enabled(cached):
                try:
                    return self.manager.plugins[cached]["handle"](intent, args or {}, ctx or {})
                except Exception as exc:
                    logger.error("Plugin '%s' failed: %s", cached, exc)
                    return None
            return None

        # Linear scan with caching
        for name, p in self.manager.plugins.items():
            if not self.is_enabled(name):
                continue
            triggers = [t.lower() for t in p["meta"].get("triggers", [])]
            if intent_l == name.lower() or any(t in intent_l for t in triggers):
                self._trigger_cache[intent_l] = name
                try:
                    return p["handle"](intent, args or {}, ctx or {})
                except Exception as exc:
                    logger.error("Plugin '%s' failed: %s", name, exc)
                    return None

        self._trigger_cache[intent_l] = ""  # cache miss
        return None

    # ── hot reload ─────────────────────────────────────────────────────────────
    def start_watching(self, poll_seconds: float = 2.0) -> None:
        if self._watcher and self._watcher.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            last_mtime = self._dir_snapshot()
            while not self._stop.is_set():
                time.sleep(poll_seconds)
                now = self._dir_snapshot()
                if now != last_mtime:
                    logger.info("Plugin directory changed — reloading.")
                    self.discover()
                    last_mtime = now

        self._watcher = threading.Thread(target=_loop, daemon=True, name="plugin-watcher")
        self._watcher.start()

    def stop_watching(self) -> None:
        self._stop.set()

    def _dir_snapshot(self) -> tuple:
        if not self.plugin_dir.exists():
            return ()
        return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in self.plugin_dir.glob("*.py")))
