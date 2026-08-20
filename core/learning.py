"""
JARVIS Self-Learning System — learns continuously from every conversation,
like ChatGPT / Gemini "memory".

What it does (all offline, no extra dependencies):
  1. AUTO FACT EXTRACTION — scans user utterances for durable facts
     ("my name is X", "I work at Y", "I like Z", "remember that ...") and stores
     them into long-term memory + the semantic vector index for instant recall.
  2. CORRECTION LEARNING — detects when the user corrects Jarvis
     ("no, it's actually ...", "you're wrong", "that's not right") and stores the
     corrected fact so the same mistake is not repeated.
  3. SEMANTIC RECALL — searches everything Jarvis has learned + recent
     conversation history by *meaning* (not just keyword), via VectorMemory.
  4. TOOL-HABIT LEARNING — tracks which tools the user uses most, so Jarvis can
     proactively suggest the next helpful action and personalize.
  5. EXPLICIT TEACHING — a ``teach`` API the LLM/plugins can call to store a
     fact the user explicitly wants remembered.

Persistence:
  * facts / corrections / habits → memory/learned.json (atomic)
  * semantic index                  → memory/vectors (VectorMemory)

Example::

    from core.learning import learner
    learner.observe_user("my name is Alex and I love pizza")
    learner.observe_user("no, my dog's name is Rex, not Max")
    hits = learner.recall("what does the user like to eat?")
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from core.security import get_base_dir

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Fact-extraction patterns                                                    #
# --------------------------------------------------------------------------- #

# (regex, category, capture-group-name). Order matters. All are case-insensitive.
_FACT_PATTERNS: list[tuple[re.Pattern, str, int]] = [
    (re.compile(r"\bmy name is ([a-z0-9 ]{1,40})", re.I), "identity", 1),
    (re.compile(r"\bi am ([a-z0-9 ]{1,40})", re.I), "identity", 1),
    (re.compile(r"\bi'?m ([a-z0-9 ]{1,40})", re.I), "identity", 1),
    (re.compile(r"\bmy (?:wife|husband|partner|girlfriend|boyfriend) (?:is|name is) ([a-z0-9 ]{1,40})", re.I), "relationships", 1),
    (re.compile(r"\bmy (?:son|daughter|child|kid|brother|sister|mother|father|mom|dad) (?:is|name is) ([a-z0-9 ]{1,40})", re.I), "relationships", 1),
    (re.compile(r"\bi work (?:at|for) ([a-z0-9 &.]{1,40})", re.I), "projects", 1),
    (re.compile(r"\bi (?:am|'m) (?:a|an) ([a-z0-9 ]{1,40}) (?:developer|engineer|designer|student|teacher|writer)", re.I), "identity", 1),
    (re.compile(r"\bi (?:live|am) (?:in|from) ([a-z0-9 &.]{1,40})", re.I), "identity", 1),
    (re.compile(r"\bi (?:like|love|enjoy|prefer) ([a-z0-9 ]{1,50})", re.I), "preferences", 1),
    (re.compile(r"\bi (?:hate|dislike|don'?t like) ([a-z0-9 ]{1,50})", re.I), "preferences", 1),
    (re.compile(r"\bmy favourite (?:movie|film|song|book|game|food|color|colour) is ([a-z0-9 ]{1,50})", re.I), "preferences", 1),
    (re.compile(r"\bmy birthday is ([a-z0-9 .]{1,30})", re.I), "identity", 1),
    (re.compile(r"\bi'?m (?:building|working on|developing) ([a-z0-9 ]{1,60})", re.I), "projects", 1),
    (re.compile(r"\bremember (?:that )?(.{4,160}?)(?:\.|$|\n)", re.I), "notes", 1),
    (re.compile(r"\bmy goal is (?:to )?(.{4,160}?)(?:\.|$|\n)", re.I), "projects", 1),
]

_CORRECTION_PATTERNS = [
    re.compile(r"\bno,? (?:it'?s|that'?s|actually|i mean|my|the|his|her) ([a-z0-9 '&-]{1,60})", re.I),
    re.compile(r"\byou'?re (?:wrong|incorrect|mistaken)", re.I),
    re.compile(r"\bthat'?s (?:not|wrong|incorrect)", re.I),
    re.compile(r"\bactually,? ([a-z0-9 '&-]{1,60})", re.I),
    re.compile(r"\bi said ([a-z0-9 '&-]{1,60})", re.I),
    re.compile(r"\bit'?s actually ([a-z0-9 '&-]{1,60})", re.I),
    re.compile(r"\bnot ([a-z0-9 '&-]{1,40}),? (?:it'?s|that'?s|its)", re.I),
]
_CORRECTION_CLEAN = re.compile(
    r"^(it is|that is|the|my|i am|i'm|the name is|it's|that's|name is|"
    r"[a-z]+'s name is) ",
    re.I,
)


# --------------------------------------------------------------------------- #
# Learner                                                                      #
# --------------------------------------------------------------------------- #

class Learner:
    def __init__(self, store_path: str | Path | None = None,
                 vector: Any | None = None) -> None:
        self.store_path = Path(store_path) if store_path else (
            get_base_dir() / "memory" / "learned.json"
        )
        self._lock = threading.Lock()
        self._data = self._load()
        try:
            from core.vector_memory import VectorMemory
            self._vector = vector or VectorMemory(name="jarvis_learned")
        except Exception as exc:  # noqa: BLE001
            logger.warning("VectorMemory unavailable for learner: %s", exc)
            self._vector = None
        # seed the vector store with already-known facts
        if self._vector is not None:
            for f in self._data.get("facts", []):
                try:
                    self._vector.add(f["text"], {"kind": f.get("category", "fact"),
                                                 "source": "seed"})
                except Exception:
                    pass

    # -- persistence -------------------------------------------------------- #
    def _load(self) -> dict[str, Any]:
        try:
            if self.store_path.exists():
                return json.loads(self.store_path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("learner load failed, starting fresh")
        return {"facts": [], "corrections": [], "habits": {}, "history": []}

    def _save(self) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store_path.with_name(self.store_path.name + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            import os
            os.replace(tmp, self.store_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("learner save failed: %s", exc)

    # -- helpers ------------------------------------------------------------ #
    def _add_fact(self, text: str, category: str, confidence: float = 0.8,
                  source: str = "auto") -> bool:
        text = text.strip().strip(".").strip()
        if len(text) < 3:
            return False
        with self._lock:
            # de-duplicate near-identical facts
            norm = text.lower()
            for f in self._data["facts"]:
                if f["text"].lower() == norm:
                    f["confidence"] = max(f.get("confidence", 0.5), confidence)
                    f["updated"] = time.strftime("%Y-%m-%d")
                    self._save()
                    return False
            entry = {
                "text": text, "category": category, "confidence": confidence,
                "source": source, "created": time.strftime("%Y-%m-%d"),
                "updated": time.strftime("%Y-%m-%d"),
            }
            self._data["facts"].append(entry)
            # keep last 2000 facts
            if len(self._data["facts"]) > 2000:
                self._data["facts"] = self._data["facts"][-2000:]
            self._save()
        if self._vector is not None:
            try:
                self._vector.add(text, {"kind": category, "source": source})
            except Exception:
                pass
        logger.info("learned fact [%s]: %s", category, text)
        return True

    # -- public observation API -------------------------------------------- #
    def observe_user(self, text: str, remember_explicit: bool = True) -> list[str]:
        """Scan a user utterance for facts + corrections. Returns learned snippets."""
        if not text:
            return []
        learned: list[str] = []

        # explicit "remember that ..." already covered by facts; scan patterns
        for pat, cat, grp in _FACT_PATTERNS:
            m = pat.search(text)
            if m:
                val = m.group(grp).strip()
                if val:
                    if self._add_fact(f"{_cat_prefix(cat)}{val}", cat):
                        learned.append(val)

        # corrections
        for pat in _CORRECTION_PATTERNS:
            m = pat.search(text)
            if m and m.groups():
                corr = _CORRECTION_CLEAN.sub("", m.group(1)).strip()
                if len(corr) > 3:
                    with self._lock:
                        self._data.setdefault("corrections", []).append({
                            "text": corr, "raw": text.strip(),
                            "ts": time.time(),
                        })
                        if len(self._data["corrections"]) > 500:
                            self._data["corrections"] = self._data["corrections"][-500:]
                        self._save()
                    learned.append(f"correction: {corr}")
                    logger.info("learned correction: %s", corr)
                    break

        return learned

    def teach(self, fact: str, category: str = "notes",
              confidence: float = 0.95) -> bool:
        """Explicit teaching (called by the ``teach`` tool / plugins)."""
        return self._add_fact(fact, category, confidence, source="explicit")

    def record_tool_use(self, tool_name: str) -> None:
        with self._lock:
            h = self._data.setdefault("habits", {})
            h[tool_name] = h.get(tool_name, 0) + 1
            self._save()

    def top_habits(self, n: int = 5) -> list[str]:
        with self._lock:
            items = sorted(self._data.get("habits", {}).items(),
                           key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in items[:n]]

    def recall(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        """Semantic recall across learned facts + history."""
        if self._vector is None:
            # offline keyword fallback
            q = query.lower()
            with self._lock:
                return [
                    {"text": f["text"], "category": f["category"], "score": 1.0}
                    for f in self._data.get("facts", [])
                    if any(w in f["text"].lower() for w in q.split() if len(w) > 3)
                ][:top_k]
        try:
            hits = self._vector.query(query, top_k=top_k)
            # also fold in a few structured facts for completeness
            return hits
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic recall failed: %s", exc)
            return []

    def learned_summary(self, limit: int = 12) -> str:
        with self._lock:
            facts = self._data.get("facts", [])[-limit:]
        if not facts:
            return "I haven't learned anything durable yet — tell me about yourself!"
        lines = [f"• {f['text']}  ({f['category']})" for f in facts]
        return "Here's what I've learned so far:\n" + "\n".join(lines)

    def learned_facts(self, limit: int = 20) -> list[str]:
        """Return learned fact texts most-recent-first (for surfacing to the model)."""
        with self._lock:
            return [f["text"] for f in self._data.get("facts", [])[-limit:]]

    def count(self) -> int:
        with self._lock:
            return len(self._data.get("facts", []))


def _cat_prefix(cat: str) -> str:
    return {
        "identity": "User: ",
        "preferences": "User prefers ",
        "relationships": "Relationship: ",
        "projects": "Project: ",
        "notes": "",
    }.get(cat, "")


# Process-wide learner instance.
learner = Learner()
