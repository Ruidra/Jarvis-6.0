"""Smoke test: exercise JARVIS 7.0 live methods with a fake UI (no mic/network)."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeUI:
    def __init__(self):
        self.logs = []
        self.state = "OFFLINE"
        self.muted = False
    def write_log(self, *a, **k):
        self.logs.append(" ".join(str(x) for x in a))
    def set_state(self, s):
        self.state = s
    def show_content(self, *a, **k):
        pass


def _make_jarvis():
    import main
    j = main.JarvisLive(FakeUI())
    j._asst_name = "JARVIS"
    j._user_name = "Boss"
    j._turn_done_event = None
    j._last_user_speech = __import__("time").monotonic()
    j._loop = None
    j.session = None  # not needed for the pure-logic paths we test
    return j


def test_emotion_context_builds():
    j = _make_jarvis()
    j.analyze_and_apply_emotion("I'm so sad today, everything went wrong")
    ctx = j._build_emotion_context("Boss")
    assert "[EMOTIONAL STATE]" in ctx
    assert j._last_emotion.label == "negative"


def test_learn_and_recall_via_tools():
    import tempfile, shutil
    from core.learning import Learner
    from core.vector_memory import VectorMemory
    d = tempfile.mkdtemp()
    try:
        vm = VectorMemory(name="test_iso", persist_dir=Path(d) / "vec")
        l = Learner(store_path=Path(d) / "learned.json", vector=vm)
        assert l.teach("User is allergic to shellfish", "preferences")
        hits = l.recall("shellfish allergy")
        assert any("shellfish" in h.get("text", "").lower() for h in hits), hits
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_motivate_tool():
    j = _make_jarvis()
    out = j._tool_motivate({"topic": "my project", "tone": "gentle"})
    assert "project" in out.lower()


def test_emotion_tool_mood():
    j = _make_jarvis()
    j.analyze_and_apply_emotion("I'm excited about the weekend!")
    out = j._tool_emotion({"action": "mood"})
    assert "mood" in out.lower()


def test_proactive_day_checkin_does_not_crash():
    j = _make_jarvis()
    # should not raise even with no session
    import asyncio
    asyncio.get_event_loop().run_until_complete(j._send_day_checkin())


def test_briefing_checkin_clause_logic():
    j = _make_jarvis()
    # simulate recent sad feeling by writing to journal via engine API
    j.analyze_and_apply_emotion("I'm anxious about the presentation")
    # _build_emotion_context already exercised; just ensure mood persisted
    assert j._emotion_engine.current_mood() in ("anxious", "sad", "angry", "tired", "confused",
                                                  "happy", "funny", "confident", "neutral")
