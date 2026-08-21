"""
Proactive Audio — JARVIS 7.0.

Knows when you're not talking to it. While JARVIS is awake (LISTENING state),
it analyzes the incoming speech transcription. If the speech doesn't address
JARVIS (no "jarvis", "hey jarvis", "listen", etc.), the turn is suppressed
and JARVIS goes back to sleep — so TV dialogue, phone calls, and conversations
with other people don't trigger accidental replies.

Usage::

    from core.proactive_audio import ProactiveAudio

    guard = ProactiveAudio()
    if not guard.is_addressed("hey jarvis what's the weather"):
        # suppress — don't respond
        go_back_to_sleep()
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("jarvis.proactive_audio")

# Terms that indicate speech is addressed to JARVIS.
# Covers names, common nicknames, and contextual cues.
ADDRESS_TERMS: list[str] = [
    "jarvis",
    "hey jarvis",
    "hi jarvis",
    "hello jarvis",
    "listen",
    "hey",
    "ok jarvis",
    "okay jarvis",
    "computer",
    "sir",
    "boss",
]

# Build a single regex for fast matching.
# We match on word boundaries to avoid false positives like "jarvissession".
_ADDRESS_RE = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in ADDRESS_TERMS) + r")\b",
    re.IGNORECASE,
)


class ProactiveAudio:
    """Detect whether a speech transcript is addressed to JARVIS."""

    def __init__(self, terms: list[str] | None = None) -> None:
        self._terms = [t.lower() for t in (terms or ADDRESS_TERMS)]
        self._regex = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in self._terms) + r")\b",
            re.IGNORECASE,
        )

    @staticmethod
    def default_terms() -> list[str]:
        return list(ADDRESS_TERMS)

    def is_addressed(self, text: str) -> bool:
        """Return True if *text* appears to be addressed to JARVIS.

        Handles common prefixes where the address term appears at the start or
        within the first few words.  For short utterances (<4 words) the term
        must be present anywhere; for longer ones it must appear in the first
        ~5 words so mid-sentence "jarvis" in casual conversation still wakes
        the assistant when it's armed but suppresses background chatter.
        """
        if not text or not text.strip():
            return False
        text_l = text.strip().lower()

        # Quick substring check on the full text — cheap and catches most cases.
        if not any(term in text_l for term in self._terms):
            return False

        # Use word-boundary regex for a precise match.
        m = self._regex.search(text_l)
        if not m:
            return False

        # For longer utterances, require address near the beginning (first 5 words).
        # This prevents TV dialog mentioning JARVIS mid-sentence from waking us,
        # while still catching normal "hey jarvis" openers.
        words = text_l.split()
        if len(words) >= 4:
            # Check if the matched term is in the first 1/3 of the utterance
            first_segment = " ".join(words[:max(2, len(words) // 3)])
            if not self._regex.search(first_segment):
                return False

        return True

    def classify(self, text: str) -> dict:
        """Detailed classification of a transcript."""
        addressed = self.is_addressed(text)
        return {
            "addressed": addressed,
            "matched_term": self._regex.search(text.strip().lower()).group(1) if addressed else None,
        }
