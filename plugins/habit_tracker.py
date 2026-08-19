"""
Jarvis Plugin — Habit & Health Tracker.

Track daily habits offline: water intake, calories (in/out), workouts, mood,
sleep, and custom notes. Persists per-day in ``memory/habits.json`` and can
report today's totals, streaks, and a weekly summary.

Triggers (spoken): "log water", "track calories", "did a workout", "my habits",
"how am I doing", "habit summary".

Args:
  action : log | summary | streak | reset   (default: summary)
  kind   : water | calories_in | calories_out | workout | mood | sleep | note
  amount : numeric value (water in glasses, calories as kcal, sleep in hours)
  note   : text for mood/note
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from core.json_store import JsonStore, read_json, atomic_write_json

logger = logging.getLogger("jarvis.plugin.habit")

PLUGIN = {
    "name": "habit",
    "description": (
        "Personal habit & health tracker (offline). Log water, calories, workouts, "
        "mood, and sleep; get today's totals, streaks, and weekly summaries. "
        "Use for 'log water', 'track calories', 'I worked out', 'my habit summary'."
    ),
    "triggers": ["log water", "track calories", "workout", "habit", "my habits"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING",
                       "description": "log | summary | streak | reset (default: summary)"},
            "kind":   {"type": "STRING",
                       "description": "water | calories_in | calories_out | workout | mood | sleep | note"},
            "amount": {"type": "NUMBER", "description": "Value: glasses, kcal, or hours."},
            "note":   {"type": "STRING", "description": "Text for mood/note kinds."},
        },
        "required": [],
    },
}

_UNITS = {
    "water": "glass(es)",
    "calories_in": "kcal",
    "calories_out": "kcal",
    "sleep": "h",
    "workout": "min",
}


def _store() -> JsonStore:
    base = Path(__file__).resolve().parent.parent
    return JsonStore(base / "memory" / "habits.json")


def _today() -> str:
    return date.today().isoformat()


def _log(kind: str, amount: float | None, note: str | None) -> str:
    kind = (kind or "").strip().lower()
    valid = {"water", "calories_in", "calories_out", "workout", "mood", "sleep", "note"}
    if kind not in valid:
        return "Log what kind? water, calories_in, calories_out, workout, mood, sleep, or note."
    store = _store()
    state = read_json(store.path, {}) or {}
    state.setdefault("days", {})
    day = state["days"].setdefault(_today(), {})
    day.setdefault(kind, [])
    if kind in ("mood", "note"):
        day[kind].append((note or "").strip())
    else:
        day[kind].append(float(amount or 0))
    atomic_write_json(store.path, state)

    if kind in ("mood", "note"):
        return f"📝 Logged {kind}: '{note}'."
    unit = _UNITS.get(kind, "")
    return f"✅ Logged {amount} {unit} of {kind} for today."


def _sum(day: dict, kind: str) -> float:
    vals = day.get(kind, [])
    if not vals:
        return 0.0
    if kind in ("mood", "note"):
        return float(len(vals))
    return sum(float(v) for v in vals)


def _summary() -> str:
    store = _store()
    state = read_json(store.path, {}) or {}
    day = state.get("days", {}).get(_today())
    user = "sir"
    if not day:
        return f"Nothing logged today, {user}. Try 'log 2 water' or 'I did a 30 min workout'."
    lines = ["📈 Today's log:"]
    for k in ("water", "calories_in", "calories_out", "workout", "sleep"):
        if day.get(k):
            lines.append(f"• {k}: {_sum(day, k):g} {_UNITS.get(k, '')}")
    net = _sum(day, "calories_in") - _sum(day, "calories_out")
    if day.get("calories_in") or day.get("calories_out"):
        lines.append(f"• net calories: {net:g} kcal")
    for k in ("mood", "note"):
        if day.get(k):
            lines.append(f"• {k}: " + "; ".join(str(x) for x in day[k]))
    return "\n".join(lines)


def _streak(kind: str) -> str:
    if not kind:
        return "Which habit's streak? e.g. streak for water."
    store = _store()
    state = read_json(store.path, {}) or {}
    days = state.get("days", {})
    streak = 0
    d = date.today().toordinal()
    while date.fromordinal(d).isoformat() in days and days[date.fromordinal(d).isoformat()].get(kind):
        streak += 1
        d -= 1
    return f"🔥 {kind} streak: {streak} day(s)."


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    action = (args.get("action") or "summary").lower().strip()
    user = (ctx.get("user_name") or "sir").title()

    if action == "log":
        return _log(args.get("kind"), args.get("amount"), args.get("note"))
    if action == "streak":
        return _streak((args.get("kind") or "").lower())
    if action == "reset":
        store = _store()
        state = read_json(store.path, {}) or {}
        state.get("days", {}).pop(_today(), None)
        atomic_write_json(store.path, state)
        return f"Reset today's log, {user}."
    return _summary()
