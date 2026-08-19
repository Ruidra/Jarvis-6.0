"""
Multi-agent collaboration for Jarvis.

A lightweight, dependency-free orchestration of role-playing agents for complex
tasks — planner proposes a plan, coder implements it, reviewer critiques and
(optionally) loops back.  Works with the local LLM client
(``core/llm_client.py``, Ollama/LM Studio) but accepts an injectable ``llm``
callable so it is fully unit-testable offline.

LangGraph/AutoGen can be dropped in later by implementing the same ``llm``
signature.

Example::

    from core.multi_agent import MultiAgentTeam
    team = MultiAgentTeam(llm=my_llm_callable)
    result = team.solve("Write a Python function to flatten a nested list.")
    print(result["final"], result["rounds"])
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_LLM = Callable[[str, str | None], str]  # (prompt, system) -> text


def _local_llm(prompt: str, system: str | None = None) -> str:
    from core.llm_client import call_llm_text

    return call_llm_text(prompt, system=system)


class MultiAgentTeam:
    def __init__(
        self,
        llm: Callable[[str, str | None], str] | None = None,
        max_rounds: int = 2,
        reviewer_can_loop: bool = True,
    ) -> None:
        self.llm = llm or _local_llm
        self.max_rounds = max_rounds
        self.reviewer_can_loop = reviewer_can_loop

    def solve(self, task: str, context: str = "") -> dict:
        plan = self._plan(task, context)
        current = plan
        rounds: list[dict] = []
        for r in range(self.max_rounds):
            implementation = self._implement(task, current)
            review = self._review(task, implementation)
            rounds.append({"round": r + 1, "plan": current, "code": implementation, "review": review})
            if review.get("approved") or not self.reviewer_can_loop:
                current = implementation
                break
            current = review.get("improved", implementation)
        return {"final": current, "rounds": rounds}

    # ── roles ───────────────────────────────────────────────────────────────
    def _plan(self, task: str, context: str) -> str:
        sys = "You are the PLANNER. Break the task into a concise step-by-step plan."
        prompt = f"TASK: {task}\nCONTEXT: {context}\n\nReturn a short numbered plan."
        return self.llm(prompt, sys).strip()

    def _implement(self, task: str, plan: str) -> str:
        sys = "You are the CODER. Produce working code or a concrete solution based on the plan."
        prompt = f"TASK: {task}\nPLAN:\n{plan}\n\nImplement it now."
        return self.llm(prompt, sys).strip()

    def _review(self, task: str, implementation: str) -> dict:
        sys = "You are the REVIEWER. Critique the solution. Respond with JSON: {\"approved\": bool, \"improved\": \"<better version or same>\"}."
        prompt = f"TASK: {task}\nSOLUTION:\n{implementation}\n\nReview it."
        raw = self.llm(prompt, sys).strip()
        try:
            import json

            data = json.loads(_extract_json(raw))
            data.setdefault("approved", False)
            data.setdefault("improved", implementation)
            return data
        except Exception:  # noqa: BLE001 - non-JSON review responses
            return {"approved": False, "improved": implementation, "raw": raw}


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text
