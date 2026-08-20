"""
Memory recall for Jarvis.

Long-term memory is already injected into the system prompt, but that window is
small and truncated. This tool lets Jarvis actively query what it knows about
the user on demand — e.g. "what's my sister's name?" or "list my projects".
It searches the stored facts (not the truncated prompt block) and returns the
matching entries so the answer is accurate instead of guessed.
"""

from __future__ import annotations

from typing import Any

from memory.memory_manager import load_memory

_CATEGORY_LABELS = {
    "identity": "Identity",
    "preferences": "Preferences",
    "projects": "Projects / Goals",
    "relationships": "People",
    "wishes": "Wishes / Plans",
    "notes": "Notes",
}


def recall_memory(parameters: dict, player: Any = None) -> str:
    """Tool handler for the ``recall_memory`` function declaration."""
    params = parameters or {}
    query = (params.get("query") or "").strip()
    category = (params.get("category") or "").strip().lower()
    memory = load_memory()

    if category and category not in memory:
        return f"No memory category named '{category}'."

    if not query and not category:
        # No filter → summarise what is stored
        counts = []
        for cat, items in memory.items():
            if isinstance(items, dict) and items:
                counts.append(f"{_CATEGORY_LABELS.get(cat, cat)}: {len(items)}")
        if not counts:
            return "I haven't stored anything about you yet."
        return "What I know about you:\n" + "\n".join(f"- {c}" for c in counts)

    q = query.lower()
    results: list[str] = []
    cats = [category] if category else list(memory.keys())
    for cat in cats:
        items = memory.get(cat)
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            value = entry.get("value") if isinstance(entry, dict) else entry
            if not value:
                continue
            haystack = f"{key} {value}".lower()
            if q and (q in haystack or q in key.lower()):
                label = key.replace("_", " ").title()
                results.append(f"- {label}: {value}")

    if not results:
        scope = f" in {category}" if category else ""
        return f"I couldn't find anything about '{query}'{scope}."

    header = f"From memory (matching '{query}'):" if query else "Matching memories:"
    out = header + "\n" + "\n".join(results[:20])

    # Also consult the auto-learner's store so facts JARVIS taught *itself* are
    # reachable through this tool, not just facts saved via save_memory.
    try:
        from core.learning import learner as _learner
        _facts = _learner.learned_facts(limit=40)
        if query:
            _facts = [f for f in _facts if q in f.lower()]
        if _facts:
            _extra = "\n".join(f"- {f}" for f in _facts[:15])
            out += "\n\nAuto-learned facts:\n" + _extra
    except Exception:
        pass

    return out
