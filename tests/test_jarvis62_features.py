"""Tests for JARVIS 7.0 features: focus mode, goals, discovery."""

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tmp_cfg():
    d = Path(tempfile.mkdtemp())
    p = d / "api_keys.json"
    p.write_text("{}", encoding="utf-8")
    return p


def test_focus_mode_enable_disable():
    from core import focus_mode
    fm = focus_mode.FocusMode(config_path=_tmp_cfg())
    assert fm.active is False
    fm.enable()
    assert fm.active is True
    fm.disable()
    assert fm.active is False


def test_focus_blocks_interrupts():
    from core import focus_mode
    fm = focus_mode.FocusMode(config_path=_tmp_cfg())
    fm.enable()
    assert fm.should_interrupt() is False
    fm.disable()
    assert fm.should_interrupt() is True


def test_goals_add_list_complete():
    from core import goals
    tmp = Path(tempfile.mkdtemp()) / "goals.json"
    g = goals.Goals(store_path=tmp)
    gid = g.add("Ship the voice upgrade", due="Friday")
    assert gid > 0
    assert len(g.list()) == 1
    assert g.complete(gid) is True
    assert g.list() == []


def test_goals_summary():
    from core import goals
    tmp = Path(tempfile.mkdtemp()) / "goals.json"
    g = goals.Goals(store_path=tmp)
    g.add("Learn guitar")
    out = g.summary()
    assert "guitar" in out


def test_discover_tool_returns_capabilities():
    import main
    j = main.JarvisLive(_FakeUI())
    out = j._tool_discover()
    assert "Tools:" in out
    assert "emotion" in out


class _FakeUI:
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
