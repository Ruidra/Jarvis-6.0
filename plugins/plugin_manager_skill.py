"""
Jarvis Plugin — Plugin Manager.

Lets you control the plugin system by voice: list what's installed, and
enable / disable individual skills. Useful after dropping a new ``*.py`` into
``plugins/`` — you can toggle it without restarting Jarvis.

Triggers (spoken): "list plugins", "enable plugin X", "disable plugin X",
"what plugins do you have".

Args:
  action : list | enable | disable   (default: list)
  name   : plugin id to enable/disable (e.g. 'quiz', 'email')
"""

from __future__ import annotations

import logging

from core.plugin_registry import PluginRegistry

logger = logging.getLogger("jarvis.plugin.plugin_manager")

PLUGIN = {
    "name": "plugin_manager",
    "description": (
        "Manage Jarvis plugins. List installed plugins (including any BROKEN "
        "ones and why they failed), or enable/disable a specific plugin by "
        "name. Use when the user says 'list plugins', 'what plugins do you "
        "have', 'show broken plugins', 'enable plugin X', or 'disable plugin X'."
    ),
    "triggers": ["list plugins", "enable plugin", "disable plugin", "manage plugin", "broken plugins"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "description": "list | enable | disable | broken (default: list)"},
            "name":   {"type": "STRING", "description": "Plugin id to enable/disable, e.g. 'quiz'."},
        },
        "required": [],
    },
}


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    action = (args.get("action") or "list").lower().strip()
    reg: PluginRegistry | None = ctx.get("plugins")
    if not reg:
        return "Plugin registry isn't available right now."

    if action in ("broken", "errors", "status"):
        broken = reg.broken()
        if not broken:
            return "✅ No broken plugins — everything loaded cleanly."
        out = ["⚠️ Broken plugins (not loaded):"]
        for key, reason in broken.items():
            out.append(f"• {key}: {reason}")
        out.append("")
        out.append("Fix the file, then say 'list plugins' to reload.")
        return "\n".join(out)

    if action == "list":
        out = ["🧩 Installed plugins:"]
        for name, p in reg.manager.plugins.items():
            state = "ENABLED" if reg.is_enabled(name) else "disabled"
            out.append(f"• {name} [{state}] — {p['meta'].get('description', '')}")
        broken = reg.broken()
        if broken:
            out.append("")
            out.append("⚠️ Broken (not loaded):")
            for key, reason in broken.items():
                out.append(f"• {key}: {reason}")
        return "\n".join(out)

    name = (args.get("name") or "").strip().lower()
    if not name:
        return "Which plugin? Give me the plugin name, e.g. 'disable plugin quiz'."
    if name not in reg.manager.plugins:
        return (f"I don't have a plugin called '{name}'. "
                f"Installed: {', '.join(reg.manager.plugins)}.")

    if action == "enable":
        reg.enable(name)
        return f"✅ Enabled plugin '{name}'."
    if action == "disable":
        reg.disable(name)
        return f"⏸️ Disabled plugin '{name}'. Re-enable anytime with 'enable plugin {name}'."

    return f"Unknown plugin action '{action}'."
