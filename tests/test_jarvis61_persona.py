"""Tests for JARVIS 6.1 part 3: personas + self-improvement."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_personas_switch_and_persist():
    from core import personas
    original = personas.get_persona().name
    try:
        p = personas.set_persona("buddy")
        assert p.name == "buddy"
        assert "buddy" in personas.get_persona().system_fragment.lower() or personas.get_persona().name == "buddy"
        assert personas.get_persona().name == "buddy"
    finally:
        personas.set_persona(original)


def test_personas_list():
    from core import personas
    titles = personas.list_personas()
    assert any("JARVIS" in t for t in titles)
    assert len(titles) >= 4


def test_self_improve_learns_lessons():
    from core.self_improve import SelfImprover
    tmp = Path(__file__).resolve().parent.parent / "memory" / "_improve_test.json"
    imp = SelfImprover(store_path=tmp)
    log = [
        "User: open notepad",
        "JARVIS: [tool open_app failed: not found]",
        "User: no, the app is called Notepad++, not notepad",
    ]
    lessons = imp.reflect(log)
    assert lessons  # at least one lesson extracted
    summary = imp.lessons_summary()
    assert "better" in summary.lower()
    try:
        tmp.unlink()
    except Exception:
        pass


def test_self_improve_no_false_lessons_on_clean_log():
    from core.self_improve import SelfImprover
    tmp = Path(__file__).resolve().parent.parent / "memory" / "_improve_clean.json"
    imp = SelfImprover(store_path=tmp)
    log = ["User: hello", "JARVIS: hi there", "User: what time is it?"]
    lessons = imp.reflect(log)
    assert lessons == []
    try:
        tmp.unlink()
    except Exception:
        pass
