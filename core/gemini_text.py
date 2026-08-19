"""
Jarvis core — Gemini-backed text generation.

A thin, robust text-generation helper that uses the SAME configured Gemini API
key the assistant already uses (so it works out of the box). It tries a fallback
chain of models and returns the first non-empty answer. Plugins (quiz, translator)
use this as a fallback when the local LLM (Ollama) is unavailable, so they keep
working with zero extra setup.
"""

from __future__ import annotations

import logging

try:
    from google import genai
except Exception:  # noqa: BLE001
    genai = None

logger = logging.getLogger("jarvis.gemini_text")

# Tried in order; the first that returns text wins. Guards against a 404 when the
# active API key lacks access to one specific model.
_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def _api_key() -> str | None:
    try:
        from config import get_secret
        return get_secret("gemini_api_key")
    except Exception:  # noqa: BLE001
        return None


def generate(prompt: str, system: str | None = None, timeout: int = 60) -> str:
    """Generate text with Gemini. Raises RuntimeError if all models fail."""
    if genai is None:
        raise RuntimeError("google.genai is not installed.")
    key = _api_key()
    if not key:
        raise RuntimeError("No Gemini API key configured.")

    client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
    full = (f"{system}\n\n" if system else "") + (prompt or "")
    contents = [{"role": "user", "parts": [{"text": full}]}]

    last_err: Exception | None = None
    for model in _MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                return text
            last_err = RuntimeError(f"{model}: empty response")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise last_err or RuntimeError("Gemini text generation failed.")
