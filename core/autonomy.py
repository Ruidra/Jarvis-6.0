"""
JARVIS Advanced Autonomy Engine — JARVIS 6.4.

Enables JARVIS to independently plan, execute, and recover from complex
multi-step goals **without** requiring the user to micromanage each step.

Capabilities:
  * Break a high-level objective into a dependency-ordered task plan.
  * Execute sub-tasks (tool calls, agent invocations, web research).
  * Detect failure / deviation and re-plan dynamically (recovery loop).
  * Track goal progress with a lightweight state machine.
  * Surface status to the user ("I'm working on goal X: 3 of 5 steps done").

Example::

    from core.autonomy import AutonomyEngine

    engine = AutonomyEngine()
    plan = engine.plan(
        "Organise my Q4 project files and email Sarah the summary"
    )
    # -> [{"step": 1, "task": "List files in project directory", "deps": []}, ...]

    for step in plan:
        result = engine.execute_step(step)
        if not result.success:
            engine.replan(result, step)

Storage: memory/autonomy_goals.json (atomic, via core.json_store).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from core.json_store import JsonStore, read_json, atomic_write_json

logger = logging.getLogger("jarvis.autonomy")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an autonomy plan."""
    idx: int
    task: str
    deps: list[int] = field(default_factory=list)        # indices of prerequisites
    tool: str = ""                                        # optional tool to call
    params: dict[str, Any] = field(default_factory=dict)  # tool params / instructions
    max_replans: int = 2                                   # how many times to retry/replan
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    attempts: int = 0
    created_ts: float = field(default_factory=time.time)


@dataclass
class AutonomyPlan:
    """A full plan tracked by the autonomy engine."""
    goal: str
    steps: list[PlanStep]
    created: float
    updated: float
    status: str = "in_progress"   # in_progress | completed | failed | abandoned


PlanExecutor = Callable[[PlanStep], dict[str, Any]]
"""Signature for a custom step executor: takes PlanStep, returns {success, output, error}."""


class AutonomyEngine:
    """Independent planning + execution engine for complex multi-step goals.

    The planner uses the local LLM (or Gemini fallback) to decompose a natural-
    language goal into ordered steps. Each step is executed by an *executor*
    callable provided by the host application (main.py). If a step fails, the
    engine can re-plan around the failure.
    """

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store = JsonStore(
            Path(store_path) if store_path
            else (Path(__file__).resolve().parent.parent / "memory" / "autonomy_goals.json")
        )
        self._active: AutonomyPlan | None = None
        self._executors: dict[str, PlanExecutor] = {}

    # ── Planner ────────────────────────────────────────────────────────────────
    def plan(self, goal: str) -> list[PlanStep]:
        """Break *goal* into ordered steps. Returns PlanStep objects."""
        system = (
            "You are a meticulous autonomous task planner. Given a goal, decompose it "
            "into 3-8 concrete, ordered sub-tasks. Each sub-task should be doable in "
            "under 30 seconds and should reference a specific action. Return ONLY a "
            "JSON array of objects, each with: step (int), task (str), deps (list[int]), "
            "tool (str), params (dict). Do not include any prose."
        )
        prompt = f"GOAL: {goal}\n\nDecompose this into ordered sub-tasks."
        raw = self._llm(prompt, system)
        steps = self._parse_steps(raw)
        if not steps:
            # Fallback: single-step plan
            steps = [PlanStep(idx=1, task=goal, tool="", params={}, deps=[])]
        self._active = AutonomyPlan(goal=goal, steps=steps, created=time.time(), updated=time.time())
        self._save_active()
        return steps

    def _llm(self, prompt: str, system: str | None) -> str:
        try:
            from core.llm_client import call_llm_text
            return call_llm_text(prompt, system=system, timeout=60)
        except Exception:
            try:
                from core.gemini_text import generate
                return generate(prompt, system=system, timeout=60)
            except Exception:
                return ""

    @staticmethod
    def _parse_steps(raw: str) -> list[PlanStep]:
        """Parse LLM JSON output into PlanStep objects."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            arr = json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return []
        steps: list[PlanStep] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            steps.append(PlanStep(
                idx=item.get("step", len(steps) + 1),
                task=item.get("task", ""),
                deps=item.get("deps", []),
                tool=item.get("tool", ""),
                params=item.get("params", {}),
            ))
        return steps

    # ── Executor registry ──────────────────────────────────────────────────────
    def register_executor(self, tool_name: str, executor: PlanExecutor) -> None:
        """Register a callable that knows how to execute a step's *tool*."""
        self._executors[tool_name] = executor

    # ── Execution & recovery ──────────────────────────────────────────────────
    def execute_step(self, step: PlanStep) -> dict[str, Any]:
        """Execute a single step. Returns {success, output, error}."""
        executor = self._executors.get(step.tool)
        if not executor:
            return {"success": False, "error": f"No executor for tool '{step.tool}'"}
        step.status = StepStatus.RUNNING
        step.attempts += 1
        try:
            result = executor(step)
            step.result = result.get("output", "")
            if result.get("success"):
                step.status = StepStatus.DONE
            else:
                if step.attempts < step.max_replans:
                    step.status = StepStatus.PENDING  # will be replanned
                    self.replan(result, step)
                else:
                    step.status = StepStatus.FAILED
            return result
        except Exception as exc:
            logger.error("autonomy step failed: %s", exc)
            step.status = StepStatus.FAILED if step.attempts >= step.max_replans else StepStatus.PENDING
            step.result = str(exc)
            return {"success": False, "error": str(exc)}

    def replan(self, failure: dict, failed_step: PlanStep) -> list[PlanStep]:
        """Attempt a re-plan around a failed step."""
        prompt = (
            f"The goal was: '{self._active.goal}'\n"
            f"A sub-task failed:\n"
            f"  Step {failed_step.idx}: {failed_step.task}\n"
            f"  Error: {failure.get('error', 'unknown')}\n"
            f"  Tool: {failed_step.tool}\n\n"
            f"Suggest an alternative approach or a simpler way to achieve the same goal."
        )
        system = (
            "You are a recovery planner. Suggest 1-3 alternative steps to replace the "
            "failed step. Return ONLY a JSON array of objects with: task (str), "
            "tool (str), params (dict)."
        )
        raw = self._llm(prompt, system)
        alternatives = self._parse_steps(raw)
        if alternatives:
            for alt in alternatives:
                alt.idx = max(s.idx for s in self._active.steps) + 1
                alt.deps = [failed_step.idx]  # depend on the original step being skipped
                self._active.steps.append(alt)
            self._active.updated = time.time()
            self._save_active()
        return alternatives

    # ── State management ───────────────────────────────────────────────────────
    def next_due_step(self) -> PlanStep | None:
        """Return the next PENDING step whose deps are all DONE, or None."""
        for step in self._active.steps if self._active else []:
            if step.status != StepStatus.PENDING:
                continue
            deps_done = all(
                self._active.steps[d - 1].status == StepStatus.DONE
                for d in step.deps if 0 < d <= len(self._active.steps)
            )
            if deps_done:
                return step
        return None

    def progress(self) -> dict[str, Any]:
        """Return a human-readable progress summary."""
        if not self._active:
            return {"status": "idle"}
        done_count = sum(1 for s in self._active.steps if s.status == StepStatus.DONE)
        total = len(self._active.steps)
        failed = sum(1 for s in self._active.steps if s.status == StepStatus.FAILED)
        return {
            "goal": self._active.goal,
            "status": self._active.status,
            "progress": f"{done_count}/{total} steps done",
            "done": done_count,
            "total": total,
            "failed": failed,
            "steps": [
                {"idx": s.idx, "task": s.task, "status": s.status.value, "result": s.result[:100]}
                for s in self._active.steps
            ],
        }

    def is_complete(self) -> bool:
        if not self._active:
            return True
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED)
                   for s in self._active.steps)

    def complete(self) -> None:
        if self._active:
            self._active.status = "completed" if self.is_complete() else "failed"
            self._active.updated = time.time()
            self._save_active()

    def abort(self) -> None:
        if self._active:
            self._active.status = "abandoned"
            self._active.updated = time.time()
            self._save_active()
            self._active = None

    def _save_active(self) -> None:
        if not self._active:
            return
        state = read_json(self.store.path, {}) or {}
        state.setdefault("plans", [])
        # Update or append the active plan
        existing = None
        for i, p in enumerate(state["plans"]):
            if p.get("goal") == self._active.goal and p.get("status") != "completed":
                existing = i
                break
        plan_dict = {
            "goal": self._active.goal,
            "created": self._active.created,
            "updated": self._active.updated,
            "status": self._active.status,
            "steps": [
                {
                    "idx": s.idx, "task": s.task, "deps": s.deps,
                    "tool": s.tool, "params": s.params,
                    "max_replans": s.max_replans, "status": s.status.value,
                    "result": s.result, "attempts": s.attempts,
                }
                for s in self._active.steps
            ],
        }
        if existing is not None:
            state["plans"][existing] = plan_dict
        else:
            state["plans"].append(plan_dict)
        atomic_write_json(self.store.path, state)

    @property
    def active_goal(self) -> str | None:
        return self._active.goal if self._active else None


# Process-wide instance.
autonomy = AutonomyEngine()
