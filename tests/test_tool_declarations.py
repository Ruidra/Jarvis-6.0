"""
Tests for the LLM tool-calling schema.

The assistant's capabilities live in ``main.TOOL_DECLARATIONS``.  When that schema
drifts (missing description, malformed parameters), Gemini function-calling breaks
silently.  These tests fail fast on structural problems.
"""

from __future__ import annotations

import pytest


def _load_tools():
    try:
        import main  # type: ignore
        return getattr(main, "TOOL_DECLARATIONS", None)
    except Exception as e:  # pragma: no cover - import environment issue
        pytest.skip(f"main.py not importable in this env: {e}")


@pytest.fixture(scope="module")
def tools():
    return _load_tools()


def test_tools_present(tools):
    assert tools, "TOOL_DECLARATIONS is empty/missing"


def test_each_tool_well_formed(tools):
    required_top = {"name", "description", "parameters"}
    for t in tools:
        missing = required_top - set(t.keys())
        assert not missing, f"{t.get('name','?')} missing keys: {missing}"
        assert isinstance(t["name"], str) and t["name"]
        assert isinstance(t["description"], str) and len(t["description"]) > 5
        params = t["parameters"]
        assert params.get("type") == "OBJECT", f"{t['name']} parameters.type != OBJECT"
        props = params.get("properties", {})
        assert isinstance(props, dict), f"{t['name']} properties not a dict"


def test_required_is_list_of_known_props(tools):
    for t in tools:
        props = t["parameters"].get("properties", {})
        required = t["parameters"].get("required", [])
        for r in required:
            assert r in props, f"{t['name']} requires '{r}' but it's not in properties"


def test_no_duplicate_tool_names(tools):
    names = [t["name"] for t in tools]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"


def test_specific_tools_exist(tools):
    names = {t["name"] for t in tools}
    for expected in (
        "open_app", "web_search", "send_message", "file_processor",
        "image_gen", "forget_memory", "audit_memory", "undo_last",
    ):
        assert expected in names, f"expected tool '{expected}' not declared"
