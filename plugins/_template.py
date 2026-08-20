"""
================================================================================
 JARVIS PLUGIN TEMPLATE
================================================================================
Copy this file to a NEW name (e.g. `my_skill.py`) inside the `plugins/` folder,
fill in the PLUGIN dict and the `handle()` function, then restart JARVIS — or
just say "list plugins" to hot-reload it. No core files need to be touched.

IMPORTANT:
  * Files starting with "_" (like this one) are IGNORED by the plugin loader,
    so this template will never be loaded as a real plugin. Rename it first.
  * Your plugin is auto-discovered, given its own Gemini tool schema, and can
    be enabled/disabled at runtime via the plugin_manager skill.
  * A broken plugin is NEVER fatal: import errors, a missing PLUGIN dict, a
    missing handler, or a name that collides with a built-in tool or another
    plugin are reported as BROKEN (with the reason) and everything else keeps
    working. Pick a `name` that does not clash with an existing tool.

`ctx` is a shared dict containing at least:
    ctx["user_name"]      -> the user's name (or "sir")
    ctx["assistant_name"] -> the assistant's name (e.g. "JARVIS")
    ctx["ui"]             -> the UI/player object (for toasts, content panels)
    ctx["plugins"]        -> the PluginRegistry (to call other plugins)
================================================================================
"""

from __future__ import annotations


# ── 1) Declare your plugin ────────────────────────────────────────────────────
# `name`        : unique id (lowercase, no spaces). Used as the Gemini tool name.
# `description` : shown to the model so it knows when to call your plugin.
# `triggers`    : spoken phrases that route to this plugin via the dispatcher.
# `parameters`  : OPTIONAL Gemini function-call schema. If omitted, the model
#                 calls your plugin with a single JSON `args` string instead.
# `handler`     : name of your entry function (defaults to "handle").
PLUGIN = {
    "name": "my_skill",
    "description": (
        "One-line description of what this skill does and when to use it. "
        "Be specific so the model calls it for the right requests."
    ),
    "triggers": ["my skill", "do the thing"],
    "handler": "handle",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING", "description": "What the user asked the skill to do."},
        },
        "required": ["query"],
    },
}


# ── 2) Implement your handler ─────────────────────────────────────────────────
# Return a string (spoken reply) or a dict. Exceptions are caught by the
# dispatcher, so you generally don't need your own try/except here.
def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    query = (args.get("query") or intent or "").strip()
    user = (ctx.get("user_name") or "sir").title()

    # ... do the work here ...

    return f"Handled '{query}' for {user}. Replace this with your real logic."
