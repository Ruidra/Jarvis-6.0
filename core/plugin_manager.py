"""
Plugin architecture for Jarvis.

Decouples "skills" (weather, music, smart home, custom commands) from the
core so new capabilities can be added without touching ``main.py``.  A plugin
is any ``*.py`` file in ``plugins/`` that exposes a module-level ``PLUGIN``
dict and a ``handle(intent, args, ctx)`` function.

PLUGIN schema::

    PLUGIN = {
        "name": "hello",                 # unique id
        "description": "Greets the user",
        "triggers": ["hello", "hi", "hey"],  # lowercase substrings matched against intent
        "handler": "handle",             # optional; defaults to "handle"
    }

    def handle(intent: str, args: dict, ctx: dict) -> str | dict | None:
        return "Hello there!"

Dispatch matches an incoming ``intent`` string against ``triggers`` (or the
plugin name) and returns the handler's result, or ``None`` if nothing matched.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def get_base_dir() -> Path:
    import sys

    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


class PluginManager:
    def __init__(self, plugin_dir: str | Path | None = None) -> None:
        self.plugin_dir = Path(plugin_dir) if plugin_dir else (get_base_dir() / "plugins")
        self.plugins: dict[str, dict[str, Any]] = {}
        self.broken: dict[str, str] = {}
        self._trigger_cache: dict[str, str] = {}  # intent_lower → plugin_name

    def discover(self) -> list[str]:
        """Load every valid plugin module from ``plugin_dir``. Returns plugin names.

        Broken plugins (import error, missing PLUGIN dict, missing handler, or a
        name collision with another plugin) are recorded in ``self.broken`` keyed
        by filename stem so the registry can surface them as BROKEN — they never
        crash JARVIS and never overwrite a working plugin.
        """
        self.plugins.clear()
        self.broken.clear()
        self._trigger_cache.clear()
        if not self.plugin_dir.exists():
            logger.warning("Plugin directory missing: %s", self.plugin_dir)
            return []
        found: list[str] = []
        seen_names: dict[str, str] = {}  # plugin name -> owning filename stem
        for path in sorted(self.plugin_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            stem = path.stem
            try:
                mod_name = f"jarvis_plugins.{stem}"
                # Evict any cached module AND stale .pyc bytecode so hot-reload
                # always re-executes the CURRENT file.  importlib's
                # SourceFileLoader uses mtime-based .pyc invalidation, which can
                # serve stale bytecode when a plugin file is rewritten within the
                # same clock tick (fast edits / tests) — so we exec the source
                # directly into a fresh module instead of trusting the cache.
                sys.modules.pop(mod_name, None)
                source = path.read_text(encoding="utf-8")
                module = types.ModuleType(mod_name)
                module.__file__ = str(path)
                sys.modules[mod_name] = module
                exec(compile(source, str(path), "exec"), module.__dict__)
            except Exception as exc:  # noqa: BLE001 - a bad plugin must not crash the app
                self.broken[stem] = f"Failed to load: {type(exc).__name__}: {exc}"
                logger.error("Failed to load plugin %s: %s", path.name, exc)
                sys.modules.pop(mod_name, None)
                continue

            meta = getattr(module, "PLUGIN", None)
            if not isinstance(meta, dict) or "name" not in meta:
                self.broken[stem] = "Missing or invalid PLUGIN dict (needs a 'name')."
                logger.warning("Plugin %s has no PLUGIN dict; skipping", path.name)
                continue

            name = meta["name"]
            if name in seen_names:
                self.broken[stem] = (
                    f"Name '{name}' collides with plugin '{seen_names[name]}'."
                )
                logger.warning("Plugin %s name collision; skipping", path.name)
                continue

            handler_name = meta.get("handler", "handle")
            handler: Callable | None = getattr(module, handler_name, None)
            if handler is None or not callable(handler):
                self.broken[stem] = f"Missing callable handler '{handler_name}'."
                logger.warning("Plugin %s missing callable '%s'; skipping", path.name, handler_name)
                continue

            self.plugins[name] = {"meta": meta, "module": module, "handle": handler}
            seen_names[name] = stem
            found.append(name)
        logger.info("Discovered %d plugin(s): %s", len(found), found)
        if self.broken:
            logger.warning("Broken plugin(s): %s", self.broken)
        return found

    def list_broken(self) -> dict[str, str]:
        """Return {filename_stem: reason} for plugins that failed to load."""
        return dict(self.broken)

    def get(self, name: str) -> dict[str, Any] | None:
        return self.plugins.get(name)

    def list_triggers(self) -> dict[str, list[str]]:
        return {name: p["meta"].get("triggers", []) for name, p in self.plugins.items()}

    def _match(self, intent_l: str) -> str | None:
        """Return the first plugin name whose triggers match *intent_l*, or None.

        Uses the trigger cache for O(1) lookup on repeated intents, falling back
        to linear scan for uncached intents.
        """
        cached = self._trigger_cache.get(intent_l)
        if cached is not None and cached in self.plugins:
            return cached
        for name, p in self.plugins.items():
            meta = p["meta"]
            if intent_l == name.lower():
                self._trigger_cache[intent_l] = name
                return name
            triggers = [t.lower() for t in meta.get("triggers", [])]
            if any(t in intent_l for t in triggers):
                self._trigger_cache[intent_l] = name
                return name
        self._trigger_cache[intent_l] = ""  # cache miss
        return None

    def dispatch(self, intent: str, args: dict | None = None, ctx: dict | None = None) -> Any:
        """Return the first matching plugin handler's result, else ``None``.

        Matching: plugin name equality, or any ``trigger`` substring present in
        the lowercased intent.
        """
        if not self.plugins:
            self.discover()
        intent_l = (intent or "").lower()
        name = self._match(intent_l)
        if name is None:
            return None
        p = self.plugins[name]
        try:
            return p["handle"](intent, args or {}, ctx or {})
        except Exception as exc:  # noqa: BLE001 - isolate plugin failures
            logger.error("Plugin '%s' handler failed: %s", name, exc)
            return None
        return None
