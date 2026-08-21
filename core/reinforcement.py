"""
JARVIS Reinforcement Learning Engine — JARVIS 7.0.

Turns JARVIS's "lesson learning" into a **closed-loop reinforcement system**:

  * Every tool call is logged with success/failure + duration.
  * User follow-up sentiment is analysed (positive / neutral / correction).
  * A **policy table** maps "user says X → JARVIS should prefer Y" and is
    updated with exponential moving average on each interaction.
  * Periodically, preference weights stabilise → stored as a reusable
    behavioural policy.

This replaces the one-shot text-lesson approach in ``core/self_improve.py``
with actual preference learning over time.

Example::

    from core.reinforcement import rl_engine

    # After a tool call:
    rl_engine.log_tool_result(
        tool_name="web_search", success=True, duration_ms=1250,
        user_text="that's exactly what I meant",
    )

    # Get current best action for a user cue
    preferred = rl_engine.get_policy("open_app notepad")
    # -> "open_app"  (the user corrected JARVIS from browser to app last time)
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.json_store import JsonStore, read_json, atomic_write_json

logger = logging.getLogger("jarvis.rl")

_STORE = JsonStore(Path(__file__).resolve().parent.parent / "memory" / "rl_policy.json")

_ALPHA = 0.3  # learning rate — how fast new evidence overrides old
_DECAY = 0.95  # time decay per day for policy weights


def _sentiment(text: str) -> str:
    """Classify user follow-up sentiment: positive | neutral | correction."""
    if not text:
        return "neutral"
    tl = text.lower().strip()

    # Correction signals
    correction_patterns = [
        r"\bno,?\s*(you|that|it)\s+(got|misunderstood|mixed up|wrong)",
        r"\bactually,?\s*(it'?s|the|i meant)",
        r"\bi\s+(said|meant)\s+",
        r"\bdon'?t\s+(do|say)\s+that",
        r"\bthat'?s\s+(wrong|incorrect|not right)",
        r"\bnah\b",
    ]
    for pattern in correction_patterns:
        if re.search(pattern, tl):
            return "correction"

    # Negative signals
    negative_patterns = [
        r"\bnope\b", r"\bno\b", r"\buseless\b", r"\bwrong\b",
        r"\bnot what i\s+(want|meant)\b", r"\bunhelpful\b",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, tl):
            return "negative"

    # Positive signals
    positive_patterns = [
        r"\byes\b", r"\byeah\b", r"\bthanks?\b", r"\bperfect\b",
        r"\bawesome\b", r"\bbingo\b", r"\bexactly\b", r"\bthat'?s\s+it\b",
        r"\bright\b", r"\bcorrect\b", r"\bwell\s+done\b",
    ]
    for pattern in positive_patterns:
        if re.search(pattern, tl):
            return "positive"

    return "neutral"


_SENTIMENT_REWARD = {
    "positive": 1.0,
    "neutral": 0.0,
    "correction": -0.7,
    "negative": -0.5,
}


class ReinforcementEngine:
    """Closed-loop preference learning engine.

    Tracks per-(user_utterance, tool_action) pair success rates and updates
    a policy table using a simple Q-learning-like update rule.
    """

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store = JsonStore(store_path) if store_path else _STORE
        self._data = self._load()
        self._tool_logs: list[dict[str, Any]] = []
        # In-memory policy: (user_pattern, tool) -> avg_reward
        self._policy: dict[tuple[str, str], float] = defaultdict(float)
        # In-memory policy: user_pattern -> best_tool
        self._best_action: dict[str, str] = {}
        self._rebuild_policy()

    def _load(self) -> dict:
        data = read_json(self.store.path, {}) or {}
        data.setdefault("tool_logs", [])
        data.setdefault("policies", {})
        return data

    def _save(self) -> None:
        atomic_write_json(self.store.path, self._data)

    def _rebuild_policy(self) -> None:
        """Rebuild in-memory policy tables from persisted data."""
        policies = self._data.get("policies", {})
        for key, reward in policies.items():
            parts = key.rsplit("::", 1)
            if len(parts) == 2:
                self._policy[(parts[0], parts[1])] = reward
        # Find best action for each user pattern
        pattern_actions: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (user_pattern, tool), reward in self._policy.items():
            pattern_actions[user_pattern].append((tool, reward))
        for pattern, actions in pattern_actions.items():
            best = max(actions, key=lambda x: x[1])
            if best[1] > 0.1:  # only set if positive confidence
                self._best_action[pattern] = best[0]

    def log_tool_result(
        self,
        session_id: int | None,
        user_text: str,
        tool_name: str,
        success: bool,
        duration_ms: float,
        user_followup: str = "",
    ) -> None:
        """Log a tool call result + user sentiment for learning."""
        sentiment = _sentiment(user_followup)
        reward = _SENTIMENT_REWARD[sentiment]
        if success and sentiment != "correction":
            reward += 0.3  # direct success bonus
        elif not success:
            reward -= 0.5

        # Extract user intent pattern (roughly: first 5 words)
        intent = self._extract_intent(user_text)

        entry = {
            "session_id": session_id,
            "user_text": user_text[:200],
            "user_intent": intent,
            "tool": tool_name,
            "success": success,
            "sentiment": sentiment,
            "reward": round(reward, 3),
            "duration_ms": duration_ms,
            "ts": time.time(),
        }

        with self._tool_logs_lock():
            self._data["tool_logs"].append(entry)
            if len(self._data["tool_logs"]) > 1000:
                self._data["tool_logs"] = self._data["tool_logs"][-1000:]

        # Update policy table: (user_intent, tool) -> EMA of reward
        key = (intent, tool_name)
        old = self._policy.get(key, 0.0)
        new = (1 - _ALPHA) * old + _ALPHA * reward
        self._policy[key] = new

        policy_key = f"{intent}::{tool_name}"
        self._data["policies"][policy_key] = round(new, 3)

        # Update best action for this intent
        actions_for_intent = [
            (v, k[1]) for k, v in self._policy.items() if k[0] == intent
        ]
        if actions_for_intent:
            best_tool = max(actions_for_intent, key=lambda x: x[0])[1]
            if actions_for_intent[0][0] > 0.1:
                self._best_action[intent] = best_tool
            else:
                self._best_action.pop(intent, None)

        self._save()
        logger.debug("rl: intent='%s' tool='%s' reward=%.2f sentiment=%s",
                      intent, tool_name, reward, sentiment)

    @staticmethod
    def _extract_intent(text: str) -> str:
        """Extract a normalised intent pattern from user text."""
        # Remove punctuation and lowercase
        text = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
        # Take key words (skip common verbs)
        words = [w for w in text.split() if w not in ("the", "a", "an", "please", "can", "could", "would")]
        if len(words) >= 3:
            return " ".join(words[:3])
        return text[:50]

    def get_policy(self, user_text: str) -> str | None:
        """Return the best tool for this user intent, if we've learned one."""
        intent = self._extract_intent(user_text)
        return self._best_action.get(intent)

    def get_policy_confidence(self, user_text: str) -> float:
        """Return confidence (0-1) that the learned policy applies."""
        intent = self._extract_intent(user_text)
        # Look up any tool reward for this intent
        matching = [v for (i, t), v in self._policy.items() if i == intent]
        if matching:
            return max(abs(r) for r in matching)
        return 0.0

    def _tool_logs_lock(self):
        if not hasattr(self, "_lock"):
            import threading
            self._lock = threading.Lock()
        return self._lock

    def get_stats(self) -> dict[str, Any]:
        """Return summary stats for the RL engine."""
        return {
            "total_logs": len(self._data.get("tool_logs", [])),
            "policy_entries": len(self._data.get("policies", {})),
            "learned_intents": len(self._best_action),
            "top_policies": sorted(self._data.get("policies", {}).items(),
                                   key=lambda x: x[1], reverse=True)[:10],
        }

    def clear_old_logs(self, older_than_days: int = 30) -> int:
        """Remove tool logs older than *older_than_days*."""
        cutoff = time.time() - older_than_days * 86400
        with self._tool_logs_lock():
            original = len(self._data.get("tool_logs", []))
            self._data["tool_logs"] = [
                entry for entry in self._data.get("tool_logs", [])
                if entry.get("ts", 0) >= cutoff
            ]
            self._save()
            return original - len(self._data["tool_logs"])


# Process-wide instance.
rl_engine = ReinforcementEngine()
