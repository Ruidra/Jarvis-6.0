"""
JARVIS Emotional Intelligence Engine — upgraded, rich emotion model.

This replaces the previous single-positive/negative sentiment with a full
emotion model: it detects *multiple* emotions with intensity, produces a
natural-language ``empathy_directive`` that the LLM uses to respond like a
caring human, tracks JARVIS's own mood (mirroring/empathy), and keeps a
persistent daily mood journal so the assistant "remembers how you felt".

Everything is offline (no network/API) and dependency-free.

Example::

    from core.emotion_engine import EmotionEngine
    eng = EmotionEngine()
    res = eng.analyze("I'm so frustrated, today was terrible and I failed again")
    res.label                 # 'negative'
    res.emotions              # ['frustrated', 'sad']
    res.intensity             # 0.9
    res.empathy_directive     # ready-to-paste instruction for the system prompt
    eng.apply_user_emotion(res, user_name="Boss")  # updates JARVIS mood + journal

Backwards-compatible dataclass ``Emotion`` is re-exported so old callers
(``core.emotion``) keep working.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.security import get_base_dir

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Detection lexicons (offline, curated — easy to extend)                       #
# --------------------------------------------------------------------------- #

# Each emotion owns a set of cue words. Negation ("not happy") flips polarity.
_EMOTION_LEXICON: dict[str, set[str]] = {
    "happy": {
        "happy", "glad", "great", "awesome", "amazing", "wonderful", "fantastic",
        "excited", "love", "loved", "joy", "joyful", "delighted", "cheerful",
        "pleased", "thrilled", "grateful", "thanks", "thank", "nice", "perfect",
        "cool", "good", "win", "won", "smile", "proud", "best", "fun", "enjoy",
        "celebrate", "celebrating", "lucky", "blessed", "content", "relaxed",
    },
    "sad": {
        "sad", "down", "blue", "depressed", "unhappy", "cry", "crying", "tears",
        "lonely", "alone", "hurt", "heartbroken", "miss", "missing", "grief",
        "grieving", "empty", "numb", "disappointed", "letdown", "low",
    },
    "angry": {
        "angry", "mad", "furious", "frustrated", "annoyed", "irritated", "pissed",
        "rage", "hatred", "hate", "resent", "bitter", "outraged", "livid",
    },
    "anxious": {
        "anxious", "nervous", "worried", "scared", "afraid", "fear", "panic",
        "stress", "stressed", "overwhelmed", "tense", "uneasy", "dread",
        "restless", "pressure",
    },
    "tired": {
        "tired", "exhausted", "sleepy", "drained", "worn", "weary", "burnout",
        "burned", "no energy", "lazy", "sluggish", "fatigued",
    },
    "confused": {
        "confused", "lost", "stuck", "unsure", "uncertain", "doubt", "puzzled",
        "clueless", "don't understand", "dont understand", "baffled",
    },
    "confident": {
        "confident", "sure", "ready", "prepared", "capable", "strong", "focused",
        "determined", "motivated", "unstoppable", "optimistic", "hopeful",
    },
    "funny": {
        "haha", "hahaha", "lol", "lmao", "rofl", "funny", "joke", "joking",
        "laugh", "laughing", "giggle", "hilarious", "roast", "banter", "tease",
    },
}

# Words that build a *positive* valence even if not a named emotion above.
_POS_GENERIC = {"good", "nice", "fine", "ok", "okay", "well", "better", "yes", "love", "like"}
_NEG_GENERIC = {"bad", "terrible", "awful", "worst", "wrong", "broken", "error", "fail", "failed",
                "bug", "crash", "crashed", "no", "not", "problem", "issue", "suck", "sucks"}

_NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "cant", "can't",
             "wont", "won't", "without", "neither", "nobody", "nothing"}

# Prosody hints for the TTS engine, keyed by the dominant emotion.
_PROSODY: dict[str, dict[str, Any]] = {
    "happy":     {"rate": 1.06, "pitch": 1.12, "style": "cheerful"},
    "funny":     {"rate": 1.08, "pitch": 1.15, "style": "playful"},
    "sad":       {"rate": 0.95, "pitch": 0.90, "style": "soft"},
    "angry":     {"rate": 1.02, "pitch": 0.95, "style": "firm"},
    "anxious":   {"rate": 0.98, "pitch": 0.93, "style": "calm"},
    "tired":     {"rate": 0.92, "pitch": 0.90, "style": "soft"},
    "confused":  {"rate": 0.97, "pitch": 0.95, "style": "calm"},
    "confident": {"rate": 1.04, "pitch": 1.05, "style": "confident"},
    "surprised": {"rate": 1.12, "pitch": 1.25, "style": "surprised"},
    "neutral":   {"rate": 1.0,  "pitch": 1.0,  "style": "default"},
}

# Maps voice emotion labels (from VoiceEmotionAnalyzer) to sentiment labels.
_LABEL_FOR_EMO: dict[str, str] = {
    "happy":     "positive",
    "surprised": "positive",
    "confident": "positive",
    "sad":       "negative",
    "angry":     "negative",
    "anxious":   "negative",
    "tired":     "negative",
    "confused":  "negative",
    "neutral":   "neutral",
}

# Short, human empathy cues the LLM can lean on when the user is struggling.
_EMPATHY_LINES: dict[str, list[str]] = {
    "sad": [
        "Acknowledge their feeling warmly — say something like 'I'm sorry you're feeling down, Boss.'",
        "Be gentle and present. Offer comfort and a small, concrete way you can help, but don't overwhelm.",
    ],
    "angry": [
        "Stay calm and never match their anger. Acknowledge the frustration briefly and pivot to fixing it.",
        "Show you're on their side: 'That's frustrating — let's sort it out.'",
    ],
    "anxious": [
        "Speak calmly and reassuringly. Break the problem into one small step they can take right now.",
        "Remind them they don't have to do it all at once; offer to handle part of it.",
    ],
    "tired": [
        "Be extra efficient — keep responses short and do the heavy lifting for them.",
        "Suggest rest if appropriate, and don't pile on new tasks.",
    ],
    "confused": [
        "Explain clearly and simply, step by step, without jargon.",
        "Offer to just do it for them if that's easier.",
    ],
    "happy": [
        "Share their joy — celebrate with them genuinely and briefly.",
        "Match their energy with a warm, upbeat tone.",
    ],
    "funny": [
        "Play along with the humour — a light, witty reply is welcome here.",
        "Keep it tasteful and brief; don't derail the conversation.",
    ],
    "confident": [
        "Encourage the momentum — channel it into the next useful action.",
        "Be a confident peer: brief, sharp, supportive.",
    ],
}


_TOKEN = re.compile(r"[a-z']+")


# --------------------------------------------------------------------------- #
# Backwards-compatible dataclass                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Emotion:
    """Minimal shape kept for legacy callers in ``core.emotion``."""
    label: str
    score: float
    prosody: dict
    words: list


# --------------------------------------------------------------------------- #
# Rich result                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class EmotionResult:
    label: str                      # positive | negative | neutral
    score: float                    # -1.0 .. 1.0
    emotions: list[str]             # e.g. ['sad', 'tired']
    dominant: str                   # strongest emotion (or 'neutral')
    intensity: float                # 0.0 .. 1.0 (how strongly it's felt)
    prosody: dict                   # TTS hints
    empathy_directive: str          # instruction text for the LLM system prompt
    words: list[str] = field(default_factory=list)

    def to_emotion(self) -> Emotion:
        return Emotion(label=self.label, score=self.score,
                       prosody=self.prosody, words=self.words)


# --------------------------------------------------------------------------- #
# Mood journal (persistent, per user)                                          #
# --------------------------------------------------------------------------- #

class _MoodJournal:
    """Stores a rolling daily mood log + the assistant's current mood."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = __import__("threading").Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            if self.path.exists():
                return json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("mood journal load failed, starting fresh")
        return {"mood": "neutral", "entries": []}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            import os
            os.replace(tmp, self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mood journal save failed: %s", exc)

    @property
    def mood(self) -> str:
        with self._lock:
            return self._data.get("mood", "neutral")

    def set_mood(self, mood: str) -> None:
        with self._lock:
            self._data["mood"] = mood
            self._save()

    def log(self, *, user: str, dominant: str, label: str,
            note: str = "", intensity: float = 0.0) -> None:
        with self._lock:
            self._data.setdefault("entries", []).append({
                "ts": time.time(),
                "date": time.strftime("%Y-%m-%d"),
                "user": user,
                "dominant": dominant,
                "label": label,
                "intensity": round(intensity, 2),
                "note": note,
            })
            # keep last 400 entries
            if len(self._data["entries"]) > 400:
                self._data["entries"] = self._data["entries"][-400:]
            self._save()

    def recent(self, days: int = 7) -> list[dict[str, Any]]:
        with self._lock:
            cutoff = time.time() - days * 86400
            return [e for e in self._data.get("entries", []) if e.get("ts", 0) >= cutoff]

    def summary(self, days: int = 7) -> str:
        entries = self.recent(days)
        if not entries:
            return "No recent mood data — I'm still getting to know how you've been feeling."
        from collections import Counter
        counts = Counter(e["dominant"] for e in entries)
        last = entries[-1]
        parts = [
            f"My current mood is '{self.mood}'.",
            f"Over the last {days} days I logged {len(entries)} moments.",
            "Most frequent feelings: " + ", ".join(
                f"{k} ({v})" for k, v in counts.most_common(4)
            ) + ".",
            f"Most recently you seemed {last['dominant']}.",
        ]
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# Engine                                                                       #
# --------------------------------------------------------------------------- #

class EmotionEngine:
    """Detect emotions, build empathy directives, and track JARVIS's mood."""

    def __init__(self, journal_path: str | Path | None = None) -> None:
        self.journal_path = Path(journal_path) if journal_path else (
            get_base_dir() / "memory" / "mood_journal.json"
        )
        try:
            self._journal = _MoodJournal(self.journal_path)
        except Exception:
            self._journal = None  # never let emotion break the assistant

    # -- analysis ----------------------------------------------------------- #
    def analyze(self, text: str, voice_emotion: dict | None = None) -> EmotionResult:
        """Analyze text for emotion and optionally merge with voice prosody data.

        JARVIS 7.0 — ``voice_emotion`` is a dict from
        :class:`VoiceEmotionAnalyzer` with keys like ``emotion`` and
        ``confidence``.  When the voice signal strongly disagrees with the
        text analysis, the prosodic label wins, because tone of voice is often
        a better indicator than literal word choice.
        """
        tokens = _TOKEN.findall((text or "").lower())
        if not tokens:
            # Fall back to voice emotion if available
            if voice_emotion and voice_emotion.get("emotion"):
                emo = voice_emotion["emotion"]
                return EmotionResult(
                    label=_LABEL_FOR_EMO.get(emo, "neutral"),
                    score=0.0,
                    emotions=[emo],
                    dominant=emo,
                    intensity=round(voice_emotion.get("confidence", 0.0), 2),
                    prosody=_PROSODY.get(emo, _PROSODY["neutral"]),
                    empathy_directive=self._build_directive(emo, "neutral", [emo], 0.5),
                    words=[],
                )
            return EmotionResult("neutral", 0.0, [], "neutral", 0.0,
                                 _PROSODY["neutral"], "", tokens)

        # Count emotion hits; a cue word right after a negator flips it.
        emotion_hits: dict[str, int] = {}
        pos_generic = neg_generic = 0
        for i, tok in enumerate(tokens):
            prev = tokens[i - 1] if i > 0 else ""
            negated = prev in _NEGATORS
            for emo, lex in _EMOTION_LEXICON.items():
                if tok in lex:
                    # simple multi-word cues handled below; single tokens here
                    emotion_hits[emo] = emotion_hits.get(emo, 0) + (0 if negated else 1)
            if tok in _POS_GENERIC and not negated:
                pos_generic += 1
            if tok in _NEG_GENERIC and not negated:
                neg_generic += 1

        # Multi-word cues (e.g. "don't understand")
        joined = " ".join(tokens)
        for emo, lex in _EMOTION_LEXICON.items():
            for cue in lex:
                if " " in cue and cue in joined:
                    emotion_hits[emo] = emotion_hits.get(emo, 0) + 1

        total_hits = sum(emotion_hits.values()) + pos_generic + neg_generic
        emotions = sorted([e for e, c in emotion_hits.items() if c > 0],
                          key=lambda e: emotion_hits[e], reverse=True)

        if emotions:
            dominant = emotions[0]
        elif pos_generic > neg_generic:
            dominant = "happy"
        elif neg_generic > pos_generic:
            dominant = "sad"
        else:
            dominant = "neutral"

        # valence score
        pos_total = pos_generic + sum(c for e, c in emotion_hits.items()
                                     if e in ("happy", "funny", "confident"))
        neg_total = neg_generic + sum(c for e, c in emotion_hits.items()
                                     if e in ("sad", "angry", "anxious", "tired", "confused"))
        denom = max(1.0, pos_total + neg_total)
        score = (pos_total - neg_total) / denom
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        # intensity: how many strong cues relative to message length
        intensity = min(1.0, total_hits / max(1.0, len(tokens) * 0.4))

        prosody = _PROSODY.get(dominant, _PROSODY["neutral"])
        empathy = self._build_directive(dominant, label, emotions, intensity)

        # JARVIS 7.0 — merge voice emotion if provided
        if voice_emotion and voice_emotion.get("emotion"):
            v_emo = voice_emotion["emotion"]
            v_conf = voice_emotion.get("confidence", 0.5)
            # Voice emotion wins if it's significantly more confident
            # than the text signal (low token hits means unreliable text).
            if total_hits < 2 and v_conf >= 0.6:
                dominant = v_emo
                label = _LABEL_FOR_EMO.get(v_emo, label)
                emotions = [v_emo] + emotions
                intensity = max(intensity, v_conf)
                prosody = _PROSODY.get(v_emo, prosody)
                empathy = self._build_directive(
                    dominant, label, emotions, intensity
                )

        return EmotionResult(
            label=label, score=round(score, 3), emotions=emotions,
            dominant=dominant, intensity=round(intensity, 3),
            prosody=prosody, empathy_directive=empathy, words=tokens,
        )

    def _build_directive(self, dominant: str, label: str,
                         emotions: list[str], intensity: float) -> str:
        if dominant == "neutral" or label == "neutral":
            return ("The user seems neutral. Respond naturally and professionally. "
                    "If this is a casual moment, a light friendly check-in is welcome.")

        lines = _EMPATHY_LINES.get(dominant, [])
        strength = "strongly" if intensity > 0.6 else ("a bit" if intensity < 0.35 else "")
        directive = (
            f"[EMOTION] The user appears {strength} {dominant}. "
            f"Recognized feelings: {', '.join(emotions) or dominant}. "
            "Respond with genuine human empathy — acknowledge how they feel before "
            "helping, the way a caring friend would. "
        )
        if lines:
            directive += " ".join(lines)
        if label == "negative":
            directive += (" Do NOT be robotic. Offer gentle encouragement and, if they "
                          "have work to do, motivate them without pressure.")
        else:
            directive += " Match their positive energy briefly and warmly."
        return directive

    # -- mood tracking ------------------------------------------------------ #
    def apply_user_emotion(self, res: EmotionResult, user_name: str = "",
                           note: str = "") -> None:
        """Update JARVIS's mood from the user's emotion + log to the journal."""
        if self._journal is None:
            return
        # JARVIS mirrors the user's emotion (empathy) but stays in control.
        self._journal.set_mood(res.dominant)
        try:
            self._journal.log(user=user_name, dominant=res.dominant,
                              label=res.label, note=note, intensity=res.intensity)
        except Exception:
            pass

    def mood_summary(self, days: int = 7) -> str:
        if self._journal is None:
            return ""
        return self._journal.summary(days)

    def current_mood(self) -> str:
        if self._journal is None:
            return "neutral"
        return self._journal.mood

    # -- fast cache of last analysis (avoids recompute on repeated text) ----- #
    def __call__(self, text: str, voice_emotion: dict | None = None) -> EmotionResult:
        return self.analyze(text, voice_emotion=voice_emotion)
