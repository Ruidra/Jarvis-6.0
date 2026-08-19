"""
Jarvis Plugin — Translator.

Translate any text between languages using the local LLM (with Gemini fallback).
Fully voice-driven: "translate 'hello' to French", "say that in Spanish".

Args:
  text   : the text to translate
  target : target language (e.g. 'French', 'es', 'Japanese')
  source : optional source language (auto-detected if omitted)
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("jarvis.plugin.translator")

PLUGIN = {
    "name": "translator",
    "description": (
        "Translate text between languages. Use when the user says 'translate X to "
        "French', 'say that in Spanish', 'what is X in German', or wants any "
        "phrase converted to another language."
    ),
    "triggers": ["translate", "in french", "in spanish", "say that in", "what is", "language"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "text":   {"type": "STRING", "description": "The text to translate."},
            "target": {"type": "STRING", "description": "Target language, e.g. 'French', 'es', 'Japanese'."},
            "source": {"type": "STRING", "description": "Source language (optional; auto-detected)."},
        },
        "required": ["text", "target"],
    },
}


def _llm(prompt: str, system: str | None = None) -> str:
    try:
        from core.llm_client import call_llm_text
        return call_llm_text(prompt, system=system, timeout=60)
    except Exception:  # noqa: BLE001
        from core.gemini_text import generate
        return generate(prompt, system=system, timeout=60)


def _clean(text: str) -> str:
    # Strip quotes / leading labels the model sometimes adds.
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"^(translation|result)\s*[:\-]\s*", "", text, flags=re.I)
    return text.strip()


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    text = (args.get("text") or "").strip()
    target = (args.get("target") or "").strip()
    source = (args.get("source") or "").strip()

    if not text:
        return "What would you like me to translate?"
    if not target:
        return "Which language should I translate it into?"

    sys = ("You are a precise translator. Reply with ONLY the translation, "
           "no commentary, no quotes.")
    prompt = f"Translate the following{' from ' + source if source else ''} to {target}:\n{text}"
    try:
        out = _clean(_llm(prompt, system=sys))
    except Exception as e:  # noqa: BLE001
        return f"I couldn't translate that right now ({e})."
    if not out:
        return "I couldn't produce a translation."
    return f"🌐 {out}"
