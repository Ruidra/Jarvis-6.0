"""
Jarvis Plugin — Local Calendar.

A fully working, dependency-free calendar. Events are stored atomically in
``memory/calendar.json`` (no Google/Microsoft OAuth needed). Supports adding,
listing, deleting, and showing today's / upcoming events.

Triggers (spoken): "add event", "what's on my calendar", "my schedule",
"calendar", "delete event".

Args schema (passed by the model):
  action : add | list | today | upcoming | delete   (default: list)
  title  : event title
  date   : YYYY-MM-DD (required for add)
  time   : HH:MM 24h (optional)
  note   : free-text notes
  id     : event id to delete
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, date
from pathlib import Path

from core.json_store import JsonStore, read_json, atomic_write_json

logger = logging.getLogger("jarvis.plugin.calendar")

PLUGIN = {
    "name": "calendar",
    "description": (
        "Local personal calendar. Add, list, and delete events; show today's or "
        "upcoming schedule. Works fully offline (no external account needed). "
        "Use when the user mentions a calendar, schedule, event, appointment, "
        "meeting, 'add to my calendar', or 'what's on my calendar'."
    ),
    "triggers": ["calendar", "schedule", "event", "appointment", "add event"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING",
                       "description": "add | list | today | upcoming | delete (default: list)"},
            "title":  {"type": "STRING", "description": "Event title (required for add)."},
            "date":   {"type": "STRING", "description": "Event date YYYY-MM-DD (required for add)."},
            "time":   {"type": "STRING", "description": "Event time HH:MM 24h (optional)."},
            "note":   {"type": "STRING", "description": "Optional notes / location."},
            "id":     {"type": "STRING", "description": "Event id for delete."},
        },
        "required": [],
    },
}


def _store() -> JsonStore:
    base = Path(__file__).resolve().parent.parent
    return JsonStore(base / "memory" / "calendar.json")


def _parse_date(s: str | None):
    if not s:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(s).strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _next_id(state: dict) -> str:
    n = int(state.get("_seq", 0)) + 1
    state["_seq"] = n
    return f"E{n:04d}"


def _add(title: str, date_s: str, time_s: str | None, note: str | None) -> str:
    if not title:
        return "I need an event title to add it to your calendar."
    d = _parse_date(date_s)
    if not d:
        return "Please give me a valid date in YYYY-MM-DD format."
    store = _store()
    state = read_json(store.path, {}) or {}
    state.setdefault("events", {})
    eid = _next_id(state)
    state["events"][eid] = {
        "id": eid,
        "title": title,
        "date": d.isoformat(),
        "time": (time_s or "").strip(),
        "note": (note or "").strip(),
        "created": time.time(),
    }
    atomic_write_json(store.path, state)
    when = d.strftime("%A, %B %d, %Y") + (f" at {time_s}" if time_s else "")
    return f"✅ Added to your calendar: '{title}' on {when} (id {eid})."


def _fmt(ev: dict) -> str:
    when = ev.get("date", "?")
    if ev.get("time"):
        when += f" {ev['time']}"
    extra = f" — {ev['note']}" if ev.get("note") else ""
    return f"• [{ev['id']}] {when}: {ev.get('title','')}{extra}"


def _list(filter_fn) -> str:
    store = _store()
    state = read_json(store.path, {}) or {}
    events = sorted(
        (e for e in state.get("events", {}).values() if filter_fn(e)),
        key=lambda e: (e.get("date", ""), e.get("time", "")),
    )
    if not events:
        return "Your calendar is empty."
    return "📅 " + "\n".join(_fmt(e) for e in events)


def _delete(eid: str) -> str:
    if not eid:
        return "Which event id should I delete? Use 'list my calendar' to see ids."
    store = _store()
    state = read_json(store.path, {}) or {}
    if eid in state.get("events", {}):
        title = state["events"].pop(eid).get("title", "")
        atomic_write_json(store.path, state)
        return f"🗑️ Deleted '{title}' (id {eid})."
    return f"I couldn't find an event with id {eid}."


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    action = (args.get("action") or "list").lower().strip()
    user = (ctx.get("user_name") or "sir").title()

    if action == "add":
        return _add(args.get("title"), args.get("date"),
                    args.get("time"), args.get("note"))
    if action == "delete":
        return _delete((args.get("id") or "").strip())
    if action == "today":
        today = date.today().isoformat()
        return _list(lambda e: e.get("date") == today) or f"Nothing on your calendar today, {user}."
    if action == "upcoming":
        today = date.today().isoformat()
        return _list(lambda e: e.get("date", "") >= today)
    # default: list everything
    return _list(lambda e: True)
