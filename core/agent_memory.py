"""
core/agent_memory.py — what the specialist agents remember.

Every mission an agent runs is recorded here: the task, which agent handled it,
whether the manager approved it, how long it took and a short digest of the
result. Two things come out of that:

* **Recall** — before starting a new mission the orchestrator looks for similar
  past missions and hands the winning approach to the agent as context, so
  JARVIS gets better at repeated work instead of starting from zero each time.
* **Routing stats** — success rates per agent, used to break ties when the
  router is unsure and to report "which of my agents is actually good".

Storage is a single JSON file (``memory/agent_memory.json``), capped so it can
never grow without bound.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.agent_memory")

_BASE = Path(__file__).resolve().parent.parent
_STORE = _BASE / "memory" / "agent_memory.json"
_MAX_ENTRIES = 300
_lock = threading.Lock()

_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "to", "of", "in", "on", "my",
    "me", "please", "can", "you", "make", "create", "do", "that", "this", "it",
    "is", "are", "be", "have", "get", "give", "want", "need", "some", "then",
}


def _load() -> dict[str, Any]:
    try:
        data = json.loads(_STORE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("missions"), list):
            return data
    except Exception:
        pass
    return {"missions": [], "agents": {}}


def _save(data: dict[str, Any]) -> None:
    try:
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        missions = data.get("missions", [])
        if len(missions) > _MAX_ENTRIES:
            data["missions"] = missions[-_MAX_ENTRIES:]
        tmp = _STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_STORE)
    except Exception as exc:  # noqa: BLE001 — memory must never break a mission
        logger.warning("agent memory save failed: %s", exc)


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def record(
    task: str,
    agent: str | None,
    status: str,
    approved: bool,
    summary: str = "",
    result: str = "",
    elapsed: float = 0.0,
    revisions: int = 0,
) -> None:
    """Store one finished mission."""
    with _lock:
        data = _load()
        data["missions"].append({
            "ts": time.time(),
            "task": (task or "")[:400],
            "agent": agent or "fallback",
            "status": status,
            "approved": bool(approved),
            "summary": (summary or "")[:400],
            "digest": (result or "")[:800],
            "elapsed": round(float(elapsed), 1),
            "revisions": int(revisions),
        })
        agents = data.setdefault("agents", {})
        st = agents.setdefault(agent or "fallback", {"runs": 0, "approved": 0, "seconds": 0.0})
        st["runs"] += 1
        st["approved"] += 1 if approved else 0
        st["seconds"] = round(st["seconds"] + float(elapsed), 1)
        _save(data)


def similar(task: str, limit: int = 3, min_overlap: int = 2) -> list[dict[str, Any]]:
    """Past missions whose wording overlaps this task, best first."""
    want = _keywords(task)
    if not want:
        return []
    scored: list[tuple[float, dict]] = []
    for m in _load().get("missions", []):
        have = _keywords(m.get("task", ""))
        if not have:
            continue
        overlap = len(want & have)
        if overlap < min_overlap:
            continue
        score = overlap / max(1, len(want | have))
        if m.get("approved"):
            score += 0.15          # prefer approaches that actually worked
        scored.append((score, m))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return [m for _, m in scored[:limit]]


def context_for(task: str, limit: int = 2) -> str:
    """A short prompt fragment describing how similar work went before."""
    past = [m for m in similar(task, limit=limit) if m.get("approved")]
    if not past:
        return ""
    lines = ["[PAST EXPERIENCE — similar missions you completed successfully]"]
    for m in past:
        lines.append(
            f"- Task: {m['task'][:160]}\n"
            f"  Agent: {m['agent']}, took {m['elapsed']}s\n"
            f"  What worked: {(m.get('summary') or m.get('digest', ''))[:240]}"
        )
    lines.append("Reuse what worked; do not repeat mistakes.")
    return "\n".join(lines)


def agent_scores() -> dict[str, dict[str, Any]]:
    """Per-agent reliability stats."""
    out: dict[str, dict[str, Any]] = {}
    for name, st in _load().get("agents", {}).items():
        runs = max(1, int(st.get("runs", 0)))
        out[name] = {
            "runs": int(st.get("runs", 0)),
            "approved": int(st.get("approved", 0)),
            "success_rate": round(int(st.get("approved", 0)) / runs, 2),
            "avg_seconds": round(float(st.get("seconds", 0.0)) / runs, 1),
        }
    return out


def summary() -> str:
    """Human-readable report of the agent workforce."""
    scores = agent_scores()
    if not scores:
        return "No agent missions recorded yet."
    lines = ["Agent performance:"]
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]["runs"]):
        lines.append(
            f"  • {name}: {s['runs']} mission(s), "
            f"{int(s['success_rate'] * 100)}% approved, ~{s['avg_seconds']}s avg"
        )
    return "\n".join(lines)
