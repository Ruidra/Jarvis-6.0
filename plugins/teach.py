"""
teach — a natural-language memory plugin for JARVIS (no coding required).

Lets anyone teach JARVIS durable facts just by talking, e.g.:
    "teach my favorite color is blue"
    "teach I work at Stark Industries"
    "teach remember that I have a dentist appointment Friday"
    "teach my dog's name is Max"
    "what have you learned about me?"   (recall summary)

It parses the fact out of the sentence and stores it via the self-learning
system (core.learning), so it survives across sessions and is recalled by
meaning later. The model can also call it as a first-class tool.

Triggers: teach, remember that, learn that, what do you know, what have you learned
"""

from __future__ import annotations

import re

# Category hints: words that point at a memory category.
_CATEGORY_HINTS = {
    "preferences": ["favorite", "favourite", "like", "love", "hate", "dislike",
                    "prefer", "allergic", "allergy", "color", "colour", "food",
                    "movie", "song", "book", "game", "music", "band"],
    "relationships": ["wife", "husband", "partner", "girlfriend", "boyfriend",
                      "son", "daughter", "brother", "sister", "mother", "father",
                      "mom", "dad", "dog", "cat", "pet", "friend", "boss", "colleague"],
    "projects": ["working on", "building", "developing", "project", "startup",
                 "company", "work at", "work for", "business", "app", "website"],
    "identity": ["my name", "i am", "i'm", "i live", "i'm from", "from", "birthday"],
}

_PREFIX_RE = re.compile(
    r"^(?:teach|learn|remember|note|save|store|memorize)(?:\s+(?:that|me)?)?\s*:?\s*",
    re.I,
)


def _parse(intent: str) -> tuple[str, str]:
    """Return (fact_text, category). Empty fact → ('', '')."""
    text = intent.strip()
    m = _PREFIX_RE.match(text)
    fact = text[m.end():].strip() if m else text
    # strip a trailing question mark / period for cleanliness
    fact = fact.rstrip(".?! ").strip()
    if len(fact) < 3:
        return "", ""
    cat = "notes"
    low = fact.lower()
    for category, hints in _CATEGORY_HINTS.items():
        if any(h in low for h in hints):
            cat = category
            break
    return fact, cat


def handle(intent: str, args: dict, ctx: dict) -> str | None:
    # --- recall queries ---------------------------------------------------
    low = intent.lower()
    if any(k in low for k in ("what do you know", "what have you learned",
                              "what do you remember", "show me what you", "learned about me")):
        try:
            from core.learning import learner
            return learner.learned_summary(limit=15)
        except Exception as exc:
            return f"I couldn't recall my notes: {exc}"

    # --- teach / remember -------------------------------------------------
    fact, cat = _parse(intent)
    if not fact:
        return ("Tell me what to remember. For example: "
                "\"teach my favorite color is blue\" or "
                "\"remember that I have a meeting on Friday\".")
    try:
        from core.learning import learner
        ok = learner.teach(fact, cat)
        who = ctx.get("user_name") or "sir"
        if ok:
            return f"Got it, {who} — I'll remember that: {fact}."
        return f"I already knew that one, {who}: {fact}."
    except Exception as exc:
        return f"Sorry, I couldn't save that: {exc}"


PLUGIN = {
    "name": "teach",
    "description": (
        "Teach JARVIS durable facts in plain English (no coding). Also answers "
        "'what have you learned about me?'. Use for remembering preferences, "
        "people, projects, and notes."
    ),
    "triggers": [
        "teach", "remember that", "learn that", "note that", "save that",
        "what have you learned", "what do you know about me", "what do you remember",
    ],
    "handler": "handle",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "fact": {
                "type": "STRING",
                "description": "The fact to remember, e.g. 'my favorite color is blue'.",
            },
            "query": {
                "type": "STRING",
                "description": "When asking what JARVIS has learned, this can be omitted.",
            },
        },
        "required": [],
    },
}
