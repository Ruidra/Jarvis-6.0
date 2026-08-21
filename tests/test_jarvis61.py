"""Tests for JARVIS 7.0: emotional intelligence, self-learning, and fast cache."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tmp_journal():
    d = tempfile.mkdtemp()
    return Path(d) / "mood.json"


# --------------------------------------------------------------------------- #
# Emotion engine                                                               #
# --------------------------------------------------------------------------- #

def test_emotion_detects_sadness():
    from core.emotion_engine import EmotionEngine
    eng = EmotionEngine(journal_path=_tmp_journal())
    res = eng.analyze("I'm so sad and frustrated, today was terrible")
    assert res.label == "negative"
    assert "sad" in res.dominant or "frustrated" in res.emotions
    assert "[EMOTION]" in res.empathy_directive
    assert res.intensity > 0.0


def test_emotion_detects_happiness():
    from core.emotion_engine import EmotionEngine
    eng = EmotionEngine(journal_path=_tmp_journal())
    res = eng.analyze("I'm so happy and excited, this is amazing!")
    assert res.label == "positive"
    assert res.dominant in ("happy", "funny", "confident")


def test_emotion_negation():
    from core.emotion_engine import EmotionEngine
    eng = EmotionEngine(journal_path=_tmp_journal())
    res = eng.analyze("I'm not happy about this")
    # "not happy" should not be read as positive happiness
    assert res.dominant != "happy"


def test_emotion_mood_journal_persists():
    from core.emotion_engine import EmotionEngine
    p = _tmp_journal()
    eng = EmotionEngine(journal_path=p)
    res = eng.analyze("I'm tired and a bit down")
    eng.apply_user_emotion(res, user_name="Boss")
    assert p.exists()
    summary = eng.mood_summary(days=7)
    assert "tired" in summary or "down" in summary


def test_emotion_py_backcompat():
    from core.emotion import analyze, Emotion
    res = analyze("I love this, it's wonderful")
    assert isinstance(res, Emotion)
    assert res.label in ("positive", "neutral", "negative")
    assert "rate" in res.prosody


# --------------------------------------------------------------------------- #
# Learning                                                                     #
# --------------------------------------------------------------------------- #

def test_learner_extracts_facts():
    import shutil
    from core.learning import Learner
    from core.vector_memory import VectorMemory
    d = tempfile.mkdtemp()
    try:
        vm = VectorMemory(name="test_facts", persist_dir=Path(d) / "vec")
        l = Learner(store_path=Path(d) / "learned.json", vector=vm)
        learned = l.observe_user("my name is Alex and I love pizza")
        assert any("Alex" in x for x in learned)
        hits = l.recall("what is the user's name?")
        assert any("Alex" in h.get("text", "") for h in hits)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_learner_correction():
    from core.learning import Learner
    tmp = Path(tempfile.mkdtemp()) / "learned.json"
    l = Learner(store_path=tmp)
    l.observe_user("no, his name is Rexford, not Max")
    assert l._data["corrections"]
    assert any("Rexford" in c.get("text", "") for c in l._data["corrections"])


def test_learner_teach_and_recall():
    from core.learning import Learner
    tmp = Path(tempfile.mkdtemp()) / "learned.json"
    l = Learner(store_path=tmp)
    l.teach("User is allergic to peanuts", "preferences")
    hits = l.recall("what allergies does the user have?")
    assert any("peanut" in h.get("text", "").lower() for h in hits)


def test_learner_tool_habits():
    from core.learning import Learner
    tmp = Path(tempfile.mkdtemp()) / "learned.json"
    l = Learner(store_path=tmp)
    for _ in range(3):
        l.record_tool_use("web_search")
    l.record_tool_use("timer")
    assert "web_search" in l.top_habits(2)


# --------------------------------------------------------------------------- #
# Fast cache                                                                   #
# --------------------------------------------------------------------------- #

def test_cache_set_get():
    from core.fast_cache import FastCache
    c = FastCache(default_ttl=60)
    c.set("k", "v")
    assert c.get("k") == "v"


def test_cache_expiry():
    from core.fast_cache import FastCache
    c = FastCache(default_ttl=1)
    c.set("k", "v", ttl=0)  # expire immediately
    assert c.get("k") is None


def test_cache_get_or_set():
    from core.fast_cache import FastCache
    c = FastCache(default_ttl=60)
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return "made"

    assert c.get_or_set("x", factory) == "made"
    assert c.get_or_set("x", factory) == "made"
    assert calls["n"] == 1   # factory called only once
