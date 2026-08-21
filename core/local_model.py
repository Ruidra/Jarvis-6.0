"""
JARVIS Local Model Fallback — JARVIS 7.0.

Runs a small local model (Llama / Qwen via Ollama) for low-stakes tasks,
reserving Gemini Live for complex reasoning and voice interaction.

Use cases:
  * Intent classification (route user command to the right tool)
  * Simple Q&A when offline
  * Keyword extraction for memory tagging
  * Fallback when Gemini API is unavailable

Example::

    from core.local_model import LocalModel

    lm = LocalModel()
    if lm.is_available():
        intent = lm.classify_intent("open notepad")
        # -> "open_app"
    else:
        # Fall back to keyword matching
        intent = "open_app"
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("jarvis.local_model")


class LocalModel:
    """Thin wrapper around Ollama for low-stakes local inference."""

    def __init__(self, model: str = "qwen2.5:latest") -> None:
        self.model = model
        self._client = None
        self._available = False
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            import ollama
            self._client = ollama
            # Quick health check
            list(self._client.list().get("models", []))
            self._available = True
            logger.info("Local model available: %s", self.model)
        except ImportError:
            logger.info("ollama package not installed — local model fallback disabled")
            self._available = False
        except Exception:
            logger.info("Ollama server not running — local model fallback disabled")
            self._available = False

    @property
    def is_available(self) -> bool:
        if not self._available:
            return False
        try:
            import ollama
            list(ollama.list().get("models", []))
            return True
        except Exception:
            self._available = False
            return False

    def simple_qa(self, question: str) -> str:
        """Answer a simple factual question locally."""
        if not self.is_available:
            return ""
        try:
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                options={"temperature": 0.3, "num_ctx": 512},
            )
            return resp.get("message", {}).get("content", "").strip()
        except Exception as exc:
            logger.debug("Local model QA failed: %s", exc)
            return ""

    def classify_intent(self, text: str) -> str:
        """Classify user intent into a tool name (keyword fallback included)."""
        if not text:
            return "unknown"
        tl = text.lower()

        # Fast keyword matching (no model needed)
        intent_map: dict[str, list[str]] = {
            "open_app": ["open", "launch", "start", "run"],
            "web_search": ["search", "find", "look up", "what is", "how to"],
            "vision": ["screenshot", "capture", "what's on screen", "what is on my screen"],
            "power_tools": ["shutdown", "restart", "sleep", "lock", "install"],
            "email": ["email", "send mail", "gmail"],
            "calendar": ["calendar", "schedule", "meeting", "event"],
            "goals": ["goal", "todo", "remind me", "task"],
            "emotion": ["mood", "feeling", "emotion", "how are you"],
            "learn": ["remember", "teach", "learn"],
            "autonomy": ["plan", "task", "do this", "multi-step"],
            "domain": ["smarthome", "smart home", "lights", "temperature"],
        }

        for intent, keywords in intent_map.items():
            if any(kw in tl for kw in keywords):
                return intent

        # Fall back to local model if available
        if self.is_available:
            prompt = (
                f"Classify this user request into one tool: {list(intent_map.keys())}. "
                f"Return ONLY the tool name, nothing else.\n\n"
                f"Request: '{text}'"
            )
            result = self.simple_qa(prompt).strip().lower()
            if result in intent_map:
                return result

        return "unknown"

    def extract_keywords(self, text: str, max_k: int = 5) -> list[str]:
        """Extract key terms from text (for memory tagging)."""
        import re
        # Simple keyword extraction: nouns, proper nouns (rough)
        words = re.findall(r"[a-zA-Z]+", text)
        # Filter stopwords
        stopwords = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
                     "be", "been", "to", "in", "on", "for", "with", "at", "by", "from"}
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 3]
        # Deduplicate, preserve order
        seen = set()
        result = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                result.append(kw)
            if len(result) >= max_k:
                break
        return result

    def summarize(self, text: str, max_words: int = 50) -> str:
        """Generate a short summary of *text* locally."""
        if not self.is_available:
            return text[:200]  # fallback: truncate

        prompt = f"Summarise this in {max_words} words or less:\n\n{text}"
        result = self.simple_qa(prompt)
        return result if result else text[:200]

    def get_config(self) -> dict[str, Any]:
        """Return the current configuration."""
        return {
            "model": self.model,
            "available": self.is_available,
            "backend": "ollama" if self.is_available else "disabled",
        }

    def status(self) -> str:
        if self.is_available:
            return f"Local model '{self.model}' is ready"
        return "Local model not available (install ollama and start the server)"


# Process-wide instance.
local_model = LocalModel()
