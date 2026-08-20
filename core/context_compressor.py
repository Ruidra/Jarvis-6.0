"""
Context Compressor for Jarvis.

JARVIS 6.3 — Sliding-window context compression for unlimited sessions.

As a conversation grows the token window fills up.  This module keeps the
conversation summary list within a bounded size by periodically collapsing
older turns into a single compressed summary (using an LLM when available).

If an LLM is not available a lightweight extractive summarizer (top-N
sentence scoring by TF-IDF-like word frequency) is used instead, so the
feature degrades gracefully offline.

Example::

    from core.context_compressor import ContextCompressor
    cc = ContextCompressor(max_chars=8000, compression_interval=20)
    cc.add("User: Hello there!")
    cc.add("Assistant: Hi! How can I help?")
    compressed = cc.maybe_compress(llm_client=my_llm)
    compressed    # a short summary paragraph
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

_TURN_SEPARATOR = "\n---\n"


class ContextCompressor:
    """Maintain a sliding-window conversation buffer that auto-summarises.

    Parameters
    ----------
    max_chars:
        Hard character cap on the raw buffer before compression triggers.
    compression_interval:
        Minimum number of turns between automatic compression runs.
    summary_ratio:
        Fraction of original content to keep in the compressed summary.
    """

    def __init__(
        self,
        max_chars: int = 8000,
        compression_interval: int = 20,
        summary_ratio: float = 0.3,
        max_summaries: int = 50,
    ):
        self._max_chars = max_chars
        self._compression_interval = max(1, compression_interval)
        self._summary_ratio = summary_ratio
        self._max_summaries = max_summaries

        self._buffer: list[str] = []       # raw turn texts
        self._summaries: list[str] = []   # compressed summaries
        self._turn_count = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add(self, text: str) -> None:
        """Append a conversation turn to the buffer."""
        if not text or not text.strip():
            return
        self._buffer.append(text.strip())
        self._turn_count += 1

    def maybe_compress(
        self,
        llm_client: object | None = None,
        force: bool = False,
    ) -> str | None:
        """Compress the buffer if it exceeds limits.

        Returns the compressed summary string, or ``None`` if no compression
        was needed.
        """
        total_chars = sum(len(t) for t in self._buffer)
        if not force and total_chars < self._max_chars:
            return None
        if not force and self._turn_count < self._compression_interval:
            return None

        summary = self._compress(llm_client)
        if summary:
            self._summaries.append(summary)
            # Keep only the most recent summaries
            if len(self._summaries) > self._max_summaries:
                excess = len(self._summaries) - self._max_summaries
                self._summaries = self._summaries[excess:]
        # Clear the compressed turns, keep the last few for context continuity
        keep = max(3, self._compression_interval // 2)
        self._buffer = self._buffer[-keep:] if keep < len(self._buffer) else self._buffer[:]
        self._turn_count = len(self._buffer)
        return summary

    async def maybe_compress_async(
        self,
        llm_client: object | None = None,
        force: bool = False,
    ) -> str | None:
        """Async-friendly wrapper for :meth:`maybe_compress`."""
        return self.maybe_compress(llm_client=llm_client, force=force)

    def get_context(self) -> str:
        """Return the current context: summaries + remaining raw turns."""
        parts: list[str] = []
        if self._summaries:
            parts.append("=== Conversation History ===\n" + "\n".join(self._summaries))
        if self._buffer:
            parts.append("=== Recent Turns ===")
            parts.extend(self._buffer)
        return "\n".join(parts)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def buffer_size(self) -> int:
        return sum(len(t) for t in self._buffer)

    def reset(self) -> None:
        """Clear all state (e.g. at session end)."""
        self._buffer = []
        self._summaries = []
        self._turn_count = 0

    # ------------------------------------------------------------------ #
    # Internal compression strategies
    # ------------------------------------------------------------------ #
    def _compress(self, llm_client: object | None) -> str:
        """Try LLM summarisation first, fall back to extractive."""
        text = _TURN_SEPARATOR.join(self._buffer)
        if not text.strip():
            return ""

        # Attempt LLM-based summarisation
        if llm_client is not None and hasattr(llm_client, "summarize"):
            try:
                result = llm_client.summarize(text, max_tokens=int(len(text) / 4))
                if result and len(result) < len(text) * 0.7:
                    return result.strip()
            except Exception as exc:
                logger.debug("LLM summarisation failed, falling back: %s", exc)

        # Extractive fallback
        return self._extractive_summary(text)

    def _extractive_summary(self, text: str) -> str:
        """TF-IDF-style extractive summary using top sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return text[:500]

        # Word frequency scoring
        word_freq: Counter = Counter()
        for sent in sentences:
            for word in re.findall(r"\b\w+\b", sent.lower()):
                word_freq[word] += 1

        # Score sentences
        scores: dict[str, float] = {}
        max_words = max(len(s.split()) for s in sentences) if sentences else 1
        for sent in sentences:
            words = re.findall(r"\b\w+\b", sent.lower())
            score = sum(word_freq.get(w, 0) for w in words) / max(1, len(words))
            # Prefer sentences at the beginning and end (context bookends)
            scores[sent] = score

        # Sort by score, keep top fraction
        keep = max(1, int(len(sentences) * self._summary_ratio))
        ranked = sorted(scores, key=lambda s: scores[s], reverse=True)[:keep]
        return " ".join(ranked)


__all__ = ["ContextCompressor"]
