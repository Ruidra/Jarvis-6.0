"""
Confidence / uncertainty signaling for JARVIS.

Several tools touch real files or system state.  Rather than silently acting, the
assistant should signal *how sure it is*.  This module provides a tiny vocabulary
the model can use in its replies and a wrapper that tags tool results with an
explicit confidence level so downstream UI/audio can phrase things accordingly.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Confidence(str, Enum):
    CERTAIN = "certain"          # verified, high-probability result
    LIKELY = "likely"            # plausible but not verified
    GUESS = "guess"             # best-effort, may be wrong
    UNKNOWN = "unknown"         # could not determine


# A short, natural-language prefix the assistant can prepend to a reply. Kept
# terse so it doesn't dominate the spoken response.
_PREFIX = {
    Confidence.CERTAIN: "",
    Confidence.LIKELY: "It looks like ",
    Confidence.GUESS: "I'm not certain, but my best guess is ",
    Confidence.UNKNOWN: "I couldn't verify this, so treat it as uncertain: ",
}


def phrase(text: str, level: Confidence) -> str:
    """Prepend an uncertainty cue to a reply when the level isn't CERTAIN."""
    pre = _PREFIX.get(level, "")
    if not pre:
        return text
    return pre + text


def tag(result: str, level: Confidence, detail: str = "") -> dict[str, Any]:
    """Wrap a tool result with an explicit confidence tag."""
    return {
        "result": result,
        "confidence": level.value,
        "detail": detail,
    }


# Heuristic helpers tools can use to self-assess.
def from_count(found: int, queried: bool = True) -> Confidence:
    if not queried:
        return Confidence.UNKNOWN
    if found > 0:
        return Confidence.CERTAIN
    return Confidence.GUESS


def from_error(err: Exception | str | None) -> Confidence:
    return Confidence.UNKNOWN if err else Confidence.CERTAIN
