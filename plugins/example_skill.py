"""
Example Jarvis plugin — demonstrates the plugin architecture.

Drop this in ``plugins/`` (or keep it as a blueprint).  It responds to
"hello" / "ping" and shows how a plugin reads the shared context object
(which may carry the user's name, memory, or a reply callback).
"""

from __future__ import annotations

PLUGIN = {
    "name": "hello",
    "description": "A friendly greeting skill used to showcase the plugin system.",
    "triggers": ["hello", "hi", "hey", "ping"],
    "handler": "handle",
}


def handle(intent: str, args: dict, ctx: dict) -> str:
    user = (ctx.get("user_name") or "sir").title()
    return f"Hello {user}! The plugin system is working. (intent: '{intent}')"
