"""
JARVIS Nightly Self-Review Engine — JARVIS 7.0.

Runs a daily (or on-demand) review pass that:

  1. Clusters recent tool failures and user corrections.
  2. Identifies recurring patterns (e.g. "web_search fails for price queries",
     "user always corrects app names after open_app").
  3. Generates candidate system-prompt rules and plugin tweaks.
  4. Saves them as *pending suggestions* for user approval rather than
     silently mutating behaviour.
  5. Logs a structured review to memory/reviews table (in SQLite) for
     analytics.

Example::

    from core.self_review import self_review

    report = self_review.run()
    # -> {'date': '2026-08-21', 'failures': 3, 'corrections': 2,
    #     'suggestions': [{'type': 'rule', 'text': '...', 'approved': False}]}
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core.json_store import read_json, atomic_write_json

logger = logging.getLogger("jarvis.self_review")

_SUGGESTION_STORE = Path(__file__).resolve().parent.parent / "memory" / "review_suggestions.json"


class SelfReview:
    """Nightly / on-demand self-review engine."""

    def run(self) -> dict[str, Any]:
        """Perform a full self-review pass and return the report."""
        report: dict[str, Any] = {
            "date": time.strftime("%Y-%m-%d"),
            "started_at": time.time(),
            "failures": 0,
            "corrections": 0,
            "suggestions": [],
            "summary": "",
        }

        # Gather data from multiple sources
        rl_data = read_json(_RL_PATH, {}) or {}
        imp_data = read_json(_IMPROVE_PATH, {}) or {}
        tool_logs: list[dict] = rl_data.get("tool_logs", [])

        # 1. Cluster tool failures
        failures = [log for log in tool_logs if not log.get("success")]
        report["failures"] = len(failures)

        failure_by_tool: dict[str, list[dict]] = defaultdict(list)
        for f in failures:
            failure_by_tool[f.get("tool", "unknown")].append(f)

        for tool, flist in failure_by_tool.items():
            if len(flist) >= 2:
                report["suggestions"].append({
                    "type": "plugin_tweak",
                    "target": tool,
                    "text": (
                        f"Tool '{tool}' failed {len(flist)} times in the last 24h. "
                        f"Review error patterns and add retry logic or input validation."
                    ),
                    "approved": False,
                })

        # 2. Analyse user corrections
        corrections = [log for log in tool_logs
                       if log.get("sentiment") == "correction"]
        report["corrections"] = len(corrections)

        correction_intents: Counter = Counter()
        for c in corrections:
            intent = c.get("user_intent", "unknown")
            correction_intents[intent] += 1

        for intent, count in correction_intents.most_common(5):
            if count >= 2:
                report["suggestions"].append({
                    "type": "rule",
                    "text": (
                        f"User corrected JARVIS {count}x for intent '{intent}'. "
                        f"Consider adding a clarification step: 'Did you mean X instead?'"
                    ),
                    "approved": False,
                })

        # 3. Check self-improvement lessons for recurring themes
        lessons = imp_data.get("lessons", [])
        recent_lessons = [l for l in lessons
                          if time.time() - l.get("ts", 0) < 86400 * 7]

        if len(recent_lessons) >= 3:
            report["suggestions"].append({
                "type": "system_prompt",
                "text": (
                    f"{len(recent_lessons)} recurring mistakes in the last week. "
                    f"Consider adding a global guideline to the system prompt."
                ),
                "approved": False,
            })

        # 4. Identify frequently-failing intent+tool pairs from RL data
        policies = rl_data.get("policies", {})
        weak_policies = [(k, v) for k, v in policies.items() if v < -0.3]
        for key, reward in weak_policies:
            intent, tool = key.rsplit("::", 1)
            report["suggestions"].append({
                "type": "policy_adjust",
                "text": (
                    f"Low-confidence policy: intent='{intent}' → tool='{tool}' "
                    f"(reward={reward:.2f}). Consider deprecating this mapping."
                ),
                "approved": False,
            })

        # 5. Generate summary
        report["summary"] = self._generate_summary(report)

        # Save suggestions for approval
        self._save_suggestions(report["suggestions"])

        # Save to SQLite reviews table if available
        try:
            from core.db import db
            db.save_review(
                date=report["date"],
                review_text=report["summary"],
                suggestions=report["suggestions"],
            )
        except Exception as exc:
            logger.debug("Could not save review to DB: %s", exc)

        report["completed_at"] = time.time()
        return report

    def _generate_summary(self, report: dict) -> str:
        """Generate a human-readable summary of the review."""
        parts = [f"Nightly Review — {report['date']}"]
        parts.append(f"  Tool failures: {report['failures']}")
        parts.append(f"  User corrections: {report['corrections']}")
        parts.append(f"  Suggestions generated: {len(report['suggestions'])}")
        if report["suggestions"]:
            parts.append("\nTop suggestions:")
            for s in report["suggestions"][:5]:
                parts.append(f"  • [{s['type']}] {s['text'][:100]}")
        return "\n".join(parts)

    def _save_suggestions(self, suggestions: list[dict]) -> None:
        """Save pending suggestions to the file store."""
        existing = read_json(_SUGGESTION_STORE, {"suggestions": []}) or {"suggestions": []}
        existing.setdefault("suggestions", []).extend(suggestions)
        # Deduplicate
        seen = set()
        unique = []
        for s in existing["suggestions"]:
            key = (s.get("type"), s.get("text", "")[:60])
            if key not in seen:
                seen.add(key)
                unique.append(s)
        existing["suggestions"] = unique[-100:]  # keep last 100
        existing["last_updated"] = time.time()
        atomic_write_json(_SUGGESTION_STORE, existing)

    def list_suggestions(self, approved_only: bool = False) -> list[dict]:
        """List pending or approved suggestions."""
        data = read_json(_SUGGESTION_STORE, {"suggestions": []}) or {"suggestions": []}
        suggestions = data.get("suggestions", [])
        if approved_only:
            return [s for s in suggestions if s.get("approved", False)]
        return suggestions

    def approve_suggestion(self, index: int) -> bool:
        """Approve a suggestion by index, returning success."""
        data = read_json(_SUGGESTION_STORE, {"suggestions": []}) or {"suggestions": []}
        suggestions = data.get("suggestions", [])
        if 0 <= index < len(suggestions):
            suggestions[index]["approved"] = True
            suggestions[index]["approved_at"] = time.time()
            atomic_write_json(_SUGGESTION_STORE, data)
            return True
        return False

    def run_async(self) -> None:
        """Run the review in a background thread."""
        import threading
        thread = threading.Thread(target=self.run, daemon=True, name="self-review")
        thread.start()


# Process-wide instance.
_RL_PATH = Path(__file__).resolve().parent.parent / "memory" / "rl_policy.json"
_IMPROVE_PATH = Path(__file__).resolve().parent.parent / "memory" / "improvements.json"

self_review = SelfReview()


def schedule_nightly() -> None:
    """Schedule a nightly self-review at 2 AM."""
    import threading

    def _nightly():
        while True:
            now = time.localtime()
            # Calculate seconds until 2 AM tomorrow
            target = time.time()
            # Simple approach: sleep until next 2 AM
            import datetime
            tomorrow = datetime.date.today() + datetime.timedelta(days=1)
            target_dt = datetime.datetime.combine(tomorrow, datetime.time(2, 0))
            sleep_secs = (target_dt - datetime.datetime.now()).total_seconds()
            if sleep_secs <= 0:
                sleep_secs += 24 * 3600
            time.sleep(sleep_secs)
            try:
                report = self_review.run()
                logger.info("Nightly self-review complete: %d suggestions",
                           len(report["suggestions"]))
            except Exception as exc:
                logger.error("Nightly review failed: %s", exc)

    thread = threading.Thread(target=_nightly, daemon=True, name="self-review-scheduler")
    thread.start()
