"""Tests for JARVIS 7.0 extras: teach plugin, prosody speaker, day check-in."""

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# teach plugin (plain-English memory for non-devs)                             #
# --------------------------------------------------------------------------- #

def test_teach_plugin_parses_fact():
    import plugins.teach as t
    fact, cat = t._parse("teach my favorite color is blue")
    assert fact == "my favorite color is blue"
    assert cat == "preferences"


def test_teach_plugin_parses_relationship():
    import plugins.teach as t
    fact, cat = t._parse("remember that my dog's name is Max")
    assert "Max" in fact
    assert cat == "relationships"


def test_teach_plugin_recall_summary():
    from core.learning import learner
    learner.teach("User loves pizza", "preferences")
    import plugins.teach as t
    out = t.handle("what have you learned about me?", {}, {"user_name": "Boss"})
    assert "pizza" in out


def test_teach_plugin_empty_is_gentle():
    import plugins.teach as t
    out = t.handle("teach", {}, {})
    assert "tell me" in out.lower()


# --------------------------------------------------------------------------- #
# Prosody speaker (emotion-tuned local voice)                                  #
# --------------------------------------------------------------------------- #

def test_prosody_speaker_instantiates():
    from core.prosody_speaker import ProsodySpeaker
    spk = ProsodySpeaker()
    # available is True only if a TTS engine is installed; must not raise either way
    assert spk.available in (True, False)
    # speak must never raise even with no engine
    spk.speak("hello", {"rate": 1.0, "pitch": 1.0})


# --------------------------------------------------------------------------- #
# Day check-in wiring                                                          #
# --------------------------------------------------------------------------- #

def test_day_checkin_method_exists():
    import main  # noqa: F401  (may skip if heavy deps missing)
    assert hasattr(main.JarvisLive, "_send_day_checkin")
    assert hasattr(main.JarvisLive, "speak_with_emotion")
