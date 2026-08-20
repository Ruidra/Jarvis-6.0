"""
Cross-device Web UI for Jarvis (Gradio).

A lightweight, phone/tablet-friendly web front-end that reuses the core engine:
chat with the local LLM, list/trigger plugins, and query vector memory.  Run it
with ``python web/app.py`` (or via Docker) and open the printed URL.

It is intentionally decoupled from the desktop HUD — same brain, different face.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_app():
    try:
        import gradio as gr
    except ImportError:  # pragma: no cover
        raise SystemExit("Gradio not installed. Run: pip install -r requirements-god.txt")

    from core.llm_client import call_llm_text
    from core.plugin_registry import PluginRegistry

    registry = PluginRegistry()
    registry.discover()

    def chat(message: str, history: list) -> Any:
        try:
            reply = call_llm_text(message, system="You are Jarvis, a helpful voice assistant.")
        except Exception as exc:  # noqa: BLE001
            reply = f"(LLM unavailable: {exc})"
        history = history or []
        history.append((message, reply))
        return history, ""

    def plugin_list() -> str:
        lines = []
        for name, p in registry.manager.plugins.items():
            state = "enabled" if registry.is_enabled(name) else "disabled"
            lines.append(f"- **{name}** ({state}): {p['meta'].get('description', '')}")
        return "\n".join(lines) or "_No plugins installed._"

    def run_plugin(name: str, arg: str) -> str:
        # Plugins read their own first-argument key (topic, coin, text, expr,
        # value, ...). Pass the user's argument under the common keys so the
        # Run button works for the majority of plugins.
        args = {
            "arg": arg, "text": arg, "query": arg, "topic": arg,
            "coin": arg, "expr": arg, "value": arg, "title": arg,
        }
        res = registry.dispatch(name, args, {"user_name": "user"})
        return str(res) if res is not None else f"No enabled plugin matched '{name}'."

    with gr.Blocks(title="Jarvis Web") as app:
        gr.Markdown("# 🛰️ Jarvis — Web Control")
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Talk to Jarvis…", label="Message")
            msg.submit(chat, [msg, chatbot], [chatbot, msg])
        with gr.Tab("Plugins"):
            gr.Markdown(plugin_list())
            pname = gr.Textbox(label="Plugin name")
            parg = gr.Textbox(label="Argument")
            pout = gr.Textbox(label="Result", lines=4)
            gr.Button("Run").click(run_plugin, [pname, parg], pout)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_app().launch(server_name="0.0.0.0", server_port=7860, share=False)
