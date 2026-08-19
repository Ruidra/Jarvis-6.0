"""
Unit tests for the "god-level" Jarvis modules that run offline (no LLM, audio,
or network).  Heavy optional backends (chromadb, porcupine, gradio, OCR) are
exercised only when installed; otherwise the offline fallback paths are tested.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── event_bus ──────────────────────────────────────────────────────────────────
def test_event_bus_pubsub():
    from core.event_bus import EventBus

    bus = EventBus()
    seen = []
    unsub = bus.subscribe("x", lambda e: seen.append(e.data))
    bus.emit("x", {"v": 1})
    assert seen == [{"v": 1}]
    unsub()
    bus.emit("x", {"v": 2})
    assert seen == [{"v": 1}]


# ── vector_memory (offline backend) ─────────────────────────────────────────────
def test_vector_memory_offline_recall(tmp_path, monkeypatch):
    from core import vector_memory as vm_mod

    # force offline backend regardless of whether chromadb is installed
    monkeypatch.setattr(vm_mod.VectorMemory, "_try_chroma", lambda self: None)
    mem = vm_mod.VectorMemory(persist_dir=tmp_path / "v")
    mem.add("user likes dark mode and minimal UI", {"kind": "pref"})
    mem.add("user is learning the Rust programming language", {"kind": "project"})
    mem.add("the cat sat on the mat", {"kind": "noise"})
    res = mem.query("what does the user prefer about interfaces?", top_k=2)
    assert res, "expected at least one recall"
    assert res[0]["metadata"]["kind"] in ("pref", "project")


def test_vector_memory_persist_dir_created(tmp_path):
    from core.vector_memory import VectorMemory

    mem = VectorMemory(persist_dir=tmp_path / "vectors")
    mem.add("hello world")
    assert mem.count() >= 1


# ── self_healing ───────────────────────────────────────────────────────────────
def test_self_healing_recovers_with_replanner():
    from core.self_healing import SelfHealingExecutor

    state = {"attempts": 0}

    def flaky(**kwargs):
        state["attempts"] += 1
        if state["attempts"] < 2:
            raise ValueError("boom")
        return "recovered"

    def replanner(err, step, history):
        return {"fn": step["fn"], "args": {}}

    ex = SelfHealingExecutor(replanner=replanner, max_iterations=3)
    results = ex.run([{"fn": flaky, "args": {}}])
    assert results[0]["ok"] is True
    assert results[0]["result"] == "recovered"


def test_self_healing_gives_up_after_max():
    from core.self_healing import SelfHealingExecutor

    def always_fail(**kwargs):
        raise RuntimeError("nope")

    ex = SelfHealingExecutor(replanner=lambda e, s, h: s, max_iterations=2)
    results = ex.run([{"fn": always_fail, "args": {}}])
    assert results[0]["ok"] is False


# ── multi_agent (fake llm) ───────────────────────────────────────────────────────
def test_multi_agent_pipeline_with_fake_llm():
    from core.multi_agent import MultiAgentTeam

    def fake_llm(prompt, system=None):
        if "PLANNER" in (system or ""):
            return "1. do the thing\n2. verify"
        if "CODER" in (system or ""):
            return "def f():\n    return 42"
        if "REVIEWER" in (system or ""):
            return '{"approved": true, "improved": "def f(): return 42"}'
        return "ok"

    team = MultiAgentTeam(llm=fake_llm, max_rounds=1)
    out = team.solve("write a function")
    assert out["final"] == "def f():\n    return 42"
    assert out["rounds"][0]["review"]["approved"] is True


# ── toolchain ────────────────────────────────────────────────────────────────────
def test_toolchain_file_read_write_sandbox(tmp_path):
    from core.toolchain import Toolchain, ToolError

    tc = Toolchain(sandbox_dir=tmp_path)
    assert tc.run("file_write", {"path": "a.txt", "content": "hi"})["ok"]
    r = tc.run("file_read", {"path": "a.txt"})
    assert r["ok"] and r["result"] == "hi"
    # escaping sandbox is refused
    bad = tc.run("file_read", {"path": "../escape.txt"})
    assert bad["ok"] is False and isinstance(bad["error"], str)


def test_toolchain_command_allowlist_enforced(tmp_path):
    from core.toolchain import Toolchain

    tc = Toolchain(sandbox_dir=tmp_path)
    echo = tc.run("run_command", {"command": "echo hello"})
    assert echo["ok"] and "hello" in echo["result"]
    unsafe = tc.run("run_command", {"command": "rm -rf /"})
    assert unsafe["ok"] is False


def test_toolchain_unknown_tool():
    from core.toolchain import Toolchain

    assert Toolchain().run("nope", {})["ok"] is False


# ── emotion ───────────────────────────────────────────────────────────────────────
def test_emotion_positive_negative_neutral():
    from core.emotion import analyze

    assert analyze("I love this, it's great!").label == "positive"
    assert analyze("this is broken and frustrating").label == "negative"
    assert analyze("the file is located at /tmp").label == "neutral"


# ── observability ────────────────────────────────────────────────────────────────
def test_observability_metrics():
    from core.observability import Metrics

    m = Metrics()
    m.inc("x")
    m.inc("x", 2)
    m.set("latency_ms", 10)
    snap = m.snapshot()
    assert snap["counters"]["x"] == 3
    assert snap["gauges"]["latency_ms"] == 10.0


def test_observability_attach_to_fastapi():
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception:  # noqa: BLE001 - fastapi optional
        pytest.skip("fastapi not installed")


    from core.observability import attach_to_app, metrics

    app = FastAPI()
    assert attach_to_app(app) is True
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "counters" in r.json()


# ── security: encrypted memory ───────────────────────────────────────────────────
def test_encrypted_json_store_roundtrip(tmp_path):
    from core.security import EncryptedJsonStore, SecretVault

    vault = SecretVault(key_path=tmp_path / ".vault_key")
    store = EncryptedJsonStore(path=tmp_path / "mem.enc.json", vault=vault)
    assert store.read({"default": 1}) == {"default": 1}
    store.write({"user": "Tony", "projects": {"jarvis": True}})
    # ciphertext on disk, not plaintext
    raw = (tmp_path / "mem.enc.json").read_text(encoding="utf-8")
    assert "Tony" not in raw
    assert store.read() == {"user": "Tony", "projects": {"jarvis": True}}


def test_encrypted_memory_migrate_from_plaintext(tmp_path):
    from core.security import EncryptedJsonStore, SecretVault

    (tmp_path / "long_term.json").write_text('{"identity": {"name": "Tony"}}', encoding="utf-8")
    vault = SecretVault(key_path=tmp_path / ".vault_key")
    store = EncryptedJsonStore(path=tmp_path / "long_term.enc.json", vault=vault)
    assert store.migrate_from_plaintext(tmp_path / "long_term.json") is True
    assert store.read() == {"identity": {"name": "Tony"}}


# ── plugin_registry ───────────────────────────────────────────────────────────────
def _write_plugin(directory, name, body):
    (directory / f"{name}.py").write_text(body, encoding="utf-8")


def test_plugin_registry_enable_disable_dispatch(tmp_path):
    from core.plugin_registry import PluginRegistry

    plugdir = tmp_path / "plugins"
    plugdir.mkdir()
    _write_plugin(
        plugdir,
        "greeter",
        'PLUGIN = {"name": "greeter", "triggers": ["hi"], "handler": "handle"}\n'
        'def handle(intent, args, ctx):\n'
        '    return "hi there"\n',
    )
    reg = PluginRegistry(plugin_dir=plugdir)
    reg.discover()
    assert reg.is_enabled("greeter") is True
    assert reg.dispatch("say hi") == "hi there"
    reg.disable("greeter")
    assert reg.dispatch("say hi") is None


def test_plugin_registry_hot_reload(tmp_path):
    from core.plugin_registry import PluginRegistry

    plugdir = tmp_path / "plugins"
    plugdir.mkdir()
    _write_plugin(plugdir, "a", 'PLUGIN={"name":"a","triggers":["a"]}\ndef handle(i,a,c): return "v1"\n')
    reg = PluginRegistry(plugin_dir=plugdir)
    reg.discover()
    assert reg.dispatch("a") == "v1"
    # modify the plugin file, then watch
    reg.start_watching(poll_seconds=0.1)
    time.sleep(0.2)
    _write_plugin(plugdir, "a", 'PLUGIN={"name":"a","triggers":["a"]}\ndef handle(i,a,c): return "v2"\n')
    # Poll (deterministic) until the watcher hot-reloads the new handler.
    deadline = time.time() + 5
    while time.time() < deadline:
        if reg.dispatch("a") == "v2":
            break
        time.sleep(0.05)
    reg.stop_watching()
    assert reg.dispatch("a") == "v2"


# ── supervisor ────────────────────────────────────────────────────────────────────
def test_supervisor_restarts_on_crash():
    from core.supervisor import Supervisor

    counter = {"n": 0}

    def flaky():
        counter["n"] += 1
        if counter["n"] < 3:
            raise RuntimeError("crash")
        # signal completion by stopping the supervisor
        sup.stop()

    sup = Supervisor(target=flaky, max_restarts=5, restart_delay=0.05)
    sup.start(blocking=True)
    assert counter["n"] == 3  # ran, crashed twice, succeeded on 3rd


# ── vad (energy) ──────────────────────────────────────────────────────────────────
def test_vad_energy_detects_speech():
    from core.vad import VADetector

    silent = np.zeros(16000, dtype=np.float32)
    speech = (np.sin(2 * np.pi * 200 * np.arange(16000) / 16000) * 0.5).astype(np.float32)
    vad = VADetector(threshold=0.015)
    assert vad.is_speech(speech) is True
    assert vad.is_speech(silent) is False


# ── proactive_scheduler ─────────────────────────────────────────────────────────────
def test_proactive_scheduler_runs_tasks():
    from core.proactive_scheduler import ProactiveScheduler

    hits = {"n": 0}

    def task():
        hits["n"] += 1
        return "done"

    sched = ProactiveScheduler()
    sched.add_task("t", task, interval_s=0.1)
    sched.start()
    time.sleep(0.5)
    sched.stop()
    assert hits["n"] >= 1
