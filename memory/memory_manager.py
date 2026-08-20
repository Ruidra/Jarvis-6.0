"""
Long-term memory storage for Jarvis L.

Persists everything the assistant learns about the user to
``memory/long_term.json``: identity facts, preferences, projects,
relationships, wishes, free-form notes, a rolling window of recent session
summaries, and background-monitoring topics. All disk access goes through
:func:`_read_json` / :func:`_write_json` so locking and error handling live
in one place instead of being repeated in every public function.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_lock = Lock()

MAX_VALUE_LENGTH = 380      # a single remembered fact is truncated past this
MEMORY_MAX_CHARS = 2200     # total file size budget; oldest facts are trimmed first
SESSION_HISTORY_MAX = 3     # safety cap on stored session summaries (usually 0-1 after pop)

CATEGORIES = ("identity", "preferences", "projects", "relationships", "wishes", "notes")

# Category -> (display heading, max entries shown, whether keys are title-cased)
_PROMPT_SECTIONS: list[tuple[str, str, int, bool]] = [
    ("preferences", "Preferences", 15, True),
    ("projects", "Active Projects / Goals", 8, True),
    ("relationships", "People in their life", 10, True),
    ("wishes", "Wishes / Plans / Wants", 8, True),
    ("notes", "Other notes", 8, False),  # notes keep their raw key as-is
]

_IDENTITY_FIELD_ORDER = ["name", "age", "birthday", "city", "job", "language", "school", "nationality"]


def get_base_dir() -> Path:
    """Return the project root, accounting for PyInstaller-frozen builds."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
MEMORY_PATH = BASE_DIR / "memory" / "long_term.json"


def _empty_memory() -> dict[str, Any]:
    return {cat: {} for cat in CATEGORIES}


def _read_json() -> dict[str, Any]:
    """Read the memory file under lock. Returns {} if missing/corrupt."""
    if not MEMORY_PATH.exists():
        return {}
    with _lock:
        try:
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load memory file: %s", e)
            return {}
    return data if isinstance(data, dict) else {}


def _write_json(memory: dict[str, Any]) -> None:
    """Write the memory file under lock, creating the parent dir if needed."""
    with _lock:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            MEMORY_PATH.write_text(
                json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as e:
            logger.error("Failed to write memory file: %s", e)


def load_memory() -> dict[str, Any]:
    """Load memory, guaranteeing every known category key is present."""
    data = _read_json()
    if not data:
        return _empty_memory()
    for cat in CATEGORIES:
        data.setdefault(cat, {})
    return data


def _all_fact_entries(memory: dict[str, Any]) -> list[tuple[str, str, dict]]:
    """Flatten every {value, updated} fact across all categories."""
    entries = []
    for cat, items in memory.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if isinstance(entry, dict) and "value" in entry:
                entries.append((cat, key, entry))
    return entries


def _trim_to_limit(memory: dict[str, Any]) -> dict[str, Any]:
    """Drop the oldest facts until the serialized memory fits MEMORY_MAX_CHARS."""
    if len(json.dumps(memory, ensure_ascii=False)) <= MEMORY_MAX_CHARS:
        return memory
    entries = sorted(_all_fact_entries(memory), key=lambda t: t[2].get("updated", "0000-00-00"))
    for cat, key, _ in entries:
        if len(json.dumps(memory, ensure_ascii=False)) <= MEMORY_MAX_CHARS:
            break
        del memory[cat][key]
        logger.info("Trimmed oldest fact %s/%s to stay under size limit", cat, key)
    return memory


def save_memory(memory: dict[str, Any]) -> None:
    """Trim and persist the full memory dict."""
    if not isinstance(memory, dict):
        return
    _write_json(_trim_to_limit(memory))


def _truncate_value(value: str) -> str:
    if isinstance(value, str) and len(value) > MAX_VALUE_LENGTH:
        return value[:MAX_VALUE_LENGTH].rstrip() + "…"
    return value


def _recursive_update(target: dict[str, Any], updates: dict[str, Any]) -> bool:
    """Merge ``updates`` into ``target`` in place. Returns True if anything changed."""
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and "value" not in value:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
                changed = True
            changed = _recursive_update(target[key], value) or changed
            continue
        new_val = _truncate_value(str(value["value"] if isinstance(value, dict) else value))
        existing = target.get(key)
        if not isinstance(existing, dict) or existing.get("value") != new_val:
            target[key] = {"value": new_val, "updated": datetime.now().strftime("%Y-%m-%d")}
            changed = True
    return changed


def update_memory(memory_update: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update (e.g. {"projects": {"Jarvis_l": {"value": "..."}}})."""
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()
    memory = load_memory()
    if _recursive_update(memory, memory_update):
        save_memory(memory)
        logger.info("Memory updated: %s", list(memory_update.keys()))
    return memory


def _entry_value(entry: Any) -> Any:
    """A stored entry is either {"value": ..., "updated": ...} or a bare value."""
    return entry.get("value") if isinstance(entry, dict) else entry


def _format_section(memory: dict[str, Any], category: str, heading: str, limit: int, humanize_key: bool = True) -> list[str]:
    """Render one category as '  - Key: value' lines, or [] if the category is empty."""
    items = memory.get(category, {})
    if not items:
        return []
    lines = [f"", f"{heading}:"]
    for key, entry in list(items.items())[:limit]:
        val = _entry_value(entry)
        if not val:
            continue
        label = key.replace("_", " ").title() if humanize_key else key
        lines.append(f"  - {label}: {val}")
    return lines


def format_memory_for_prompt(memory: dict[str, Any] | None) -> str:
    """Render remembered facts as a compact block for the LLM system prompt."""
    if not memory:
        return ""

    lines: list[str] = []

    identity = memory.get("identity", {})
    for field in _IDENTITY_FIELD_ORDER:
        val = _entry_value(identity.get(field))
        if val:
            lines.append(f"{field.title()}: {val}")
    for key, entry in identity.items():
        if key in _IDENTITY_FIELD_ORDER:
            continue
        val = _entry_value(entry)
        if val:
            lines.append(f"{key.replace('_', ' ').title()}: {val}")

    for category, heading, limit, humanize_key in _PROMPT_SECTIONS:
        lines.extend(_format_section(memory, category, heading, limit, humanize_key))

    # Bridge the auto-learner (regex fact extraction) into the model's context.
    # Without this, facts JARVIS teaches *itself* during conversation are stored
    # in learned.json but never surfaced, so the assistant "forgets" them.
    try:
        from core.learning import learner as _learner
        _lf = _learner.learned_facts(limit=20)
        if _lf:
            lines.append("")
            lines.append("Auto-learned facts (from earlier conversations):")
            for _f in _lf:
                lines.append(f"  - {_f}")
    except Exception:
        pass

    if not lines:
        return ""

    header = "[WHAT YOU KNOW ABOUT THIS PERSON — use naturally, never recite like a list]\n"
    result = header + "\n".join(lines)
    if len(result) > 2000:
        result = result[:1997] + "…"
    return result + "\n"


def remember(key: str, value: str, category: str = "notes") -> str:
    """Store a single fact under a category, defaulting to 'notes' if invalid."""
    if category not in CATEGORIES:
        category = "notes"
    update_memory({category: {key: {"value": value}}})
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    """Delete a single remembered fact. Returns a human-readable status string."""
    memory = load_memory()
    cat = memory.get(category, {})
    if key not in cat:
        return f"Not found: {category}/{key}"
    del cat[key]
    memory[category] = cat
    save_memory(memory)
    return f"Forgotten: {category}/{key}"


forget_memory = forget  # backwards-compatible alias


# ── Relevance / decay ────────────────────────────────────────────────────────
def relevance_score(entry: Any, now: datetime | None = None) -> float:
    """Recency-weighted relevance in [0,1]. Facts updated today score ~1.0 and
    decay toward 0.2 over ~180 days. Older/unknown entries keep a floor so they
    aren't deleted, just de-prioritised."""
    if not isinstance(entry, dict):
        return 0.5
    updated = entry.get("updated")
    if not updated:
        return 0.4
    try:
        d = datetime.strptime(updated, "%Y-%m-%d")
    except Exception:
        return 0.4
    now = now or datetime.now()
    age_days = max(0, (now - d).days)
    return max(0.2, 1.0 - age_days / 225.0)


def audit_memory() -> str:
    """Return a human-readable, age-sorted audit of everything remembered."""
    memory = load_memory()
    rows = []
    for cat, items in memory.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if cat == "sessions":
                continue
            val = _entry_value(entry)
            age = entry.get("updated", "?") if isinstance(entry, dict) else "?"
            rel = relevance_score(entry)
            rows.append((rel, f"  [{cat}] {key} = {val}  (updated {age}, relevance {rel:.2f})"))
    if not rows:
        return "Memory audit: nothing stored yet."
    rows.sort(reverse=True)
    return "Memory audit (most relevant first):\n" + "\n".join(r[1] for r in rows)


def forget_category(category: str) -> str:
    """Forget every fact in a category (used by the 'forget this' command)."""
    if category not in CATEGORIES:
        return f"Unknown category: {category}"
    memory = load_memory()
    count = len(memory.get(category, {}))
    memory[category] = {}
    save_memory(memory)
    return f"Forgot {count} fact(s) in '{category}'."


def recall_goals(language: str = "") -> str:
    """Cross-session goal tracking: surface open projects/goals + last session
    summary so the assistant can pick up where it left off."""
    memory = load_memory()
    projects = memory.get("projects", {})
    wishes = memory.get("wishes", {})
    lines = []
    for cat, items in (("projects", projects), ("wishes", wishes)):
        for key, entry in items.items():
            if isinstance(entry, dict) and entry.get("value"):
                lines.append(f"  - {key.replace('_',' ').title()}: {entry['value']}")
    sessions = memory.get("sessions", [])
    last = sessions[-1] if isinstance(sessions, list) and sessions else None
    out = ["Active goals/projects from memory:"]
    out.append("\n".join(lines) if lines else "  (none tracked yet)")
    if last:
        out.append(f"\nLast session ({last.get('date','?')}): {last.get('summary','')}")
    return "\n".join(out)


# ── Session memory ──────────────────────────────────────────────────────
#
# A one-shot, "read once and delete" record of what the last session was
# about, so the morning briefing can mention it exactly once.

def save_session_summary(summary: str, language: str = "") -> None:
    """Append a 1-2 sentence session summary, keeping only the most recent few."""
    summary = (summary or "").strip()
    if not summary:
        return
    memory = load_memory()
    sessions = memory.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []
    entry: dict[str, Any] = {"date": datetime.now().strftime("%Y-%m-%d"), "summary": summary[:280]}
    if language:
        entry["language"] = language
    sessions.append(entry)
    memory["sessions"] = sessions[-SESSION_HISTORY_MAX:]
    _write_json(memory)
    logger.info("Session summary saved (%s): %s", entry["date"], summary[:60])


def pop_last_session() -> dict[str, Any] | None:
    """Return AND remove the most recent session entry.

    Consuming the entry here means it is never repeated in a future
    briefing — callers should treat this as "read once".
    """
    memory = _read_json()
    sessions = memory.get("sessions", [])
    if not isinstance(sessions, list) or not sessions:
        return None
    entry = sessions.pop()
    memory["sessions"] = sessions
    _write_json(memory)
    return entry
