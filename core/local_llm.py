"""
Local LLM fallback client (Ollama / llama.cpp compatible).

When Gemini Live is unreachable (offline, quota, or privacy-sensitive tasks), JARVIS
can route text completions to a local model served by Ollama at ``base_url``
(default http://localhost:11434).  This keeps basic chat/summarisation alive with no
cloud dependency.  Network/import failures degrade gracefully.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:latest"


def available(base_url: str = DEFAULT_BASE_URL) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def complete(prompt: str, *, model: str = DEFAULT_MODEL,
             base_url: str = DEFAULT_BASE_URL, temperature: float = 0.7,
             max_tokens: int = 1024) -> Optional[str]:
    """Return a completion from the local model, or None if unavailable."""
    try:
        import urllib.request

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("response")
    except Exception as e:  # noqa: BLE001
        logger.warning("local_llm fallback failed: %s", e)
        return None
