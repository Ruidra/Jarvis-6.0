"""
Emotion recognition & expressive prosody for Jarvis.

This module now wraps the richer :mod:`core.emotion_engine` (multi-emotion
detection, intensity, empathy directives, persistent mood journal) while
keeping the original ``analyze() -> Emotion`` API so existing callers and
tests keep working unchanged.

Example::

    from core.emotion import analyze
    info = analyze("I'm so frustrated this keeps breaking!")
    info.label    # 'negative'
    info.prosody  # {"rate": 1.05, "pitch": 0.9, "style": "calm"}

For the full emotion model use::

    from core.emotion_engine import EmotionEngine
    eng = EmotionEngine()
    res = eng.analyze("...")
    res.empathy_directive
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.emotion_engine import Emotion, EmotionEngine, EmotionResult

logger = logging.getLogger(__name__)

# Shared engine instance (owns the persistent mood journal).
engine = EmotionEngine()


@dataclass
class EmotionLegacy:
    label: str
    score: float
    prosody: dict
    words: list


def analyze(text: str) -> Emotion:
    """Backwards-compatible entry point. Returns the minimal ``Emotion`` shape."""
    return engine.analyze(text).to_emotion()


__all__ = ["analyze", "Emotion", "EmotionEngine", "EmotionResult", "engine"]
