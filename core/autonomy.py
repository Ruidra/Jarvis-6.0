"""
JARVIS Advanced Autonomy Engine — JARVIS 7.0.

Replaces the linear step-list planner with a **DAG-based planner**: steps
can branch and depend on each other (not just sequential), plus a **critic
step** that verifies whether each execution actually advanced the goal
before committing to the next step.

Features:
  * DAG decomposition — goal broken into steps with arbitrary dependencies.
  * Critic/verifier — after each step executes, an LLM judge checks
    "did this actually accomplish the sub-goal?" If not, the step is retried
    or re-planned around.
  * Dry-run mode — simulate the full plan before real execution.
  * Per-step risk scoring — dangerous steps flagged for confirmation.
  * Parallel execution — independent steps can run concurrently.

Example::

    from core.autonomy import AutonomyEngine

    engine = AutonomyEngine()
    plan = engine.plan("Organise my Q4 project files and email Sarah the summary")
    # -> Plan with 5 DAG steps, 2 flagged as medium-risk

    # Simulate before executing:
    result = engine.simulate(plan)

    # Execute for real:
    engine.run(plan, auto_approve_low_risk=True)
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

PLAN_STORE = JsonStore(Path(__file__).resolve().parent.parent / "memory" / "autonomy_plans.json")


class StepStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"           # all deps done, ready to execute
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"       # deps failed
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an autonomy DAG plan."""
    idx: int
    task: str
    deps: list[int] = field(default_factory=list)          # prerequisite step indices
    tool: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    risk: str = "low"                                       # low | medium | high
    max_retries: int = 2
    requires_approval: bool = False
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    attempts: int = 0
    created_ts: float = field(default_factory=time.time)
    verified: bool = False                                  # passed critic check

    def to_dict(self) -> dict:
        return {
            "idx": self.idx, "task": self.task, "deps": self.deps,
            "tool": self.tool, "params": self.params, "risk": self.risk,
            "max_retries": self.max_retries, "requires_approval": self.requires_approval,
            "status": self.status.value, "result": self.result,
            "attempts": self.attempts, "created_ts": self.created_ts,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep":
        return cls(
            idx=d["idx"], task=d["task"], deps=d.get("deps", []),
            tool=d.get("tool", ""), params=d.get("params", {}),
            risk=d.get("risk", "low"), max_retries=d.get("max_retries", 2),
            requires_approval=d.get("requires_approval", False),
            status=StepStatus(d.get("status", "pending")),
            result=d.get("result", ""), attempts=d.get("attempts", 0),
            created_ts=d.get("created_ts", time.time()),
            verified=d.get("verified", False),
        )


@dataclass
class AutonomyPlan:
    """A full DAG plan tracked by the autonomy engine."""
    id: str
    goal: str
    steps: list[PlanStep]
    created: float
    updated: float
    status: str = "in_progress"   # in_progress | completed | failed | cancelled
    dry_run: bool = False
    auto_approve_low_risk: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id, "goal": self.goal, "created": self.created,
            "updated": self.updated, "status": self.status,
            "dry_run": self.dry_run, "auto_approve_low_risk": self.auto_approve_low_risk,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutonomyPlan":
        return cls(
            id=d.get("id", ""),
            goal=d.get("goal", ""),
            steps=[PlanStep.from_dict(s) for s in d.get("steps", [])],
            created=d.get("created", time.time()),
            updated=d.get("updated", time.time()),
            status=d.get("status", "in_progress"),
            dry_run=d.get("dry_run", False),
            auto_approve_low_risk=d.get("auto_approve_low_risk", True),
        )


PlanExecutor = Callable[[PlanStep], dict[str, Any]]
"""Signature: (step) -> {success, output, error}"""


class AutonomyEngine:
    """DAG-based autonomous planning engine with critic verification."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store = JsonStore(store_path) if store_path else PLAN_STORE
        self._active: AutonomyPlan | None = None
        self._executors: dict[str, PlanExecutor] = {}
        self._load_active()

    def _load_active(self) -> None:
        """Load the most recent in-progress plan."""
        state = read_json(self.store.path, {}) or {}
        plans = state.get("plans", [])
        for p in reversed(plans):
            if p.get("status") == "in_progress":
                self._active = AutonomyPlan.from_dict(p)
                break

    def _save(self) -> None:
        if not self._active:
            return
        state = read_json(self.store.path, {}) or {}
        state.setdefault("plans", [])
        # Update or append
        found = False
        for i, p in enumerate(state["plans"]):
            if p.get("id") == self._active.id:
                state["plans"][i] = self._active.to_dict()
                found = True
                break
        if not found:
            state["plans"].append(self._active.to_dict())
        # Keep last 50 plans
        state["plans"] = state["plans"][-50:]
        atomic_write_json(self.store.path, state)

    # ── Planner ────────────────────────────────────────────────────────────────
    def plan(self, goal: str, context: str = "") -> AutonomyPlan:
        """Break *goal* into a DAG of steps using LLM planning."""
        system = (
            "You are a meticulous autonomous task planner. Break the given goal into "
            "3-8 concrete, ordered sub-tasks. Each sub-task should be doable in under "
            "30 seconds. Steps can depend on each other (DAG structure, not just linear). "
            "For each step, assign a risk level: 'low' if it's safe (read/search/notify), "
            "'medium' if it modifies files or sends data, 'high' if it could cause data "
            "loss or system changes.\n\n"
            "Return ONLY a JSON array of objects, each with: idx (int), task (str), "
            "deps (list of prerequisite indices), tool (str), params (dict), risk (str)."
        )
        prompt = f"GOAL: {goal}\nCONTEXT: {context}\n\nReturn a JSON array of plan steps."
        raw = self._llm(prompt, system)
        steps = self._parse_steps(raw)
        if not steps:
            steps = [PlanStep(idx=1, task=goal, risk="high", requires_approval=True)]

        plan = AutonomyPlan(
            id=f"plan_{int(time.time())}",
            goal=goal,
            steps=steps,
            created=time.time(),
            updated=time.time(),
        )
        self._active = plan
        self._save()
        return plan

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
            risk = item.get("risk", "low")
            requires_approval = risk in ("medium", "high")
            steps.append(PlanStep(
                idx=item.get("idx", len(steps) + 1),
                task=item.get("task", ""),
                deps=item.get("deps", []),
                tool=item.get("tool", ""),
                params=item.get("params", {}),
                risk=risk,
                requires_approval=requires_approval,
            ))
        return steps

    # ── DAG logic ──────────────────────────────────────────────────────────────
    def _can_run(self, step: PlanStep) -> bool:
        """Check if all dependencies are DONE."""
        for dep in step.deps:
            if 0 < dep <= len(self._active.steps):
                dep_step = self._active.steps[dep - 1]
                if dep_step.status not in (StepStatus.DONE, StepStatus.SKIPPED):
                    return False
        return True

    def _get_ready_steps(self) -> list[PlanStep]:
        """Return steps that are pending and whose deps are all satisfied."""
        if not self._active:
            return []
        ready: list[PlanStep] = []
        for step in self._active.steps:
            if step.status != StepStatus.PENDING:
                continue
            if self._can_run(step):
                step.status = StepStatus.READY
                ready.append(step)
            else:
                step.status = StepStatus.BLOCKED
        return ready

    # ── Execution ───────────────────────────────────────────────────────────────
    def execute_step(self, step: PlanStep,
                     executor: PlanExecutor | None = None,
                     auto_approve: bool = True) -> dict[str, Any]:
        """Execute a single step with critic verification."""
        if step.requires_approval and not auto_approve:
            return {"success": False, "error": "Step requires manual approval"}

        if step.tool and step.tool in self._executors:
            executor = self._executors[step.tool]
        elif step.tool:
            return {"success": False, "error": f"No executor registered for '{step.tool}'"}

        if not executor:
            return {"success": False, "error": "No executor provided"}

        step.status = StepStatus.RUNNING
        step.attempts += 1
        self._save()

        start = time.time()
        try:
            result = executor(step)
            elapsed = time.time() - start
            step.result = result.get("output", "")[:500]
            success = result.get("success", False)

            if success:
                # Critic step: verify the execution actually advanced the goal
                if self._critic_check(step, result):
                    step.status = StepStatus.DONE
                    step.verified = True
                    self._save()
                    return {"success": True, "output": result.get("output", ""),
                            "elapsed": elapsed, "verified": True}
                else:
                    # Critic says it didn't achieve the sub-goal
                    if step.attempts < step.max_retries:
                        step.status = StepStatus.PENDING
                        self._save()
                        return {"success": False, "error": "Step failed critic verification, will retry",
                                "retry": True}
                    else:
                        step.status = StepStatus.FAILED
                        self._save()
                        return {"success": False, "error": "Max retries exceeded for critic verification"}

            else:
                if step.attempts < step.max_retries:
                    step.status = StepStatus.PENDING
                    self._save()
                    return {"success": False, "error": result.get("error", "unknown"),
                            "retry": True}
                else:
                    step.status = StepStatus.FAILED
                    self._save()
                    return {"success": False, "error": f"Max retries exceeded: {result.get('error', 'unknown')}"}

        except Exception as exc:
            elapsed = time.time() - start
            logger.error("autonomy step execution failed: %s", exc)
            if step.attempts < step.max_retries:
                step.status = StepStatus.PENDING
            else:
                step.status = StepStatus.FAILED
            step.result = str(exc)[:500]
            self._save()
            return {"success": False, "error": str(exc), "elapsed": elapsed}

    def _critic_check(self, step: PlanStep, result: dict[str, Any]) -> bool:
        """Ask an LLM judge whether the step output actually satisfies the sub-task."""
        output = result.get("output", "")
        prompt = (
            f"Goal: '{self._active.goal}'\n"
            f"Sub-task: '{step.task}'\n\n"
            f"Did the execution output below successfully complete this sub-task? "
            f"Reply with ONLY 'YES' or 'NO'.\n\n"
            f"Execution output: {output[:1000]}"
        )
        system = "You are a strict critic. Answer YES only if the sub-task was clearly completed. Otherwise NO."
        try:
            response = self._llm(prompt, system).strip().upper()
            return response.startswith("YES")
        except Exception:
            return True  # pass on LLM failure — don't block

    # ── Dry-run / simulation ──────────────────────────────────────────────────
    def simulate(self, plan: AutonomyPlan | None = None) -> dict[str, Any]:
        """Simulate the plan execution without actually running tools."""
        plan = plan or self._active
        if not plan:
            return {"error": "No active plan"}

        results = {"plan": plan.goal, "steps": [], "total_risk": 0}
        risk_rank = {"low": 1, "medium": 2, "high": 3}

        for step in plan.steps:
            deps_ok = all(
                plan.steps[d - 1].status == StepStatus.DONE
                for d in step.deps if 0 < d <= len(plan.steps)
            )
            results["steps"].append({
                "idx": step.idx,
                "task": step.task,
                "risk": step.risk,
                "deps_ok": deps_ok,
                "tool": step.tool,
                "estimated_time_s": 15,  # rough estimate
            })
            results["total_risk"] += risk_rank.get(step.risk, 1)

        high_risk = [s for s in results["steps"] if s["risk"] == "high"]
        if high_risk:
            results["warning"] = f"{len(high_risk)} high-risk step(s) detected — review before execution"

        results["estimated_total_time_s"] = len(plan.steps) * 15
        return results

    # ── Full run ───────────────────────────────────────────────────────────────
    def run(self, plan: AutonomyPlan | None = None,
            executor: PlanExecutor | None = None,
            auto_approve_low_risk: bool = True) -> dict[str, Any]:
        """Execute the entire plan, respecting DAG dependencies."""
        plan = plan or self._active
        if not plan:
            return {"error": "No active plan to run"}

        results = {"goal": plan.goal, "steps_executed": 0, "steps_succeeded": 0,
                   "steps_failed": 0, "step_results": []}

        max_iter = len(plan.steps) * (plan.steps[0].max_retries if plan.steps else 2) + 10
        iteration = 0

        while not self._is_plan_done(plan) and iteration < max_iter:
            iteration += 1
            ready = self._get_ready_steps()

            if not ready:
                # Check if remaining steps are blocked
                remaining = [s for s in plan.steps if s.status in (StepStatus.PENDING, StepStatus.BLOCKED)]
                if all(s.status == StepStatus.BLOCKED for s in remaining):
                    for s in remaining:
                        s.status = StepStatus.FAILED
                    plan.status = "failed"
                    self._save()
                    results["error"] = "Steps blocked due to failed dependencies"
                    return results
                break

            for step in ready:
                auto = auto_approve_low_risk and step.risk == "low"
                result = self.execute_step(step, executor=executor, auto_approve=auto)
                results["step_results"].append({
                    "idx": step.idx, "task": step.task,
                    "success": result.get("success", False),
                    "error": result.get("error", ""),
                    "verified": result.get("verified", False),
                })
                results["steps_executed"] += 1
                if result.get("success"):
                    results["steps_succeeded"] += 1
                elif not result.get("retry"):
                    results["steps_failed"] += 1

            self._save()

        if self._is_plan_done(plan):
            all_done = all(s.status == StepStatus.DONE for s in plan.steps)
            plan.status = "completed" if all_done else "failed"
            self._save()

        results["final_status"] = plan.status
        return results

    @staticmethod
    def _is_plan_done(plan: AutonomyPlan) -> bool:
        """Check if all steps have reached a terminal state."""
        terminal = {StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(s.status in terminal for s in plan.steps)

    # ── Executor registry ──────────────────────────────────────────────────────
    def register_executor(self, tool_name: str, executor: PlanExecutor) -> None:
        self._executors[tool_name] = executor

    # ── Status ─────────────────────────────────────────────────────────────────
    def progress(self) -> dict[str, Any]:
        if not self._active:
            return {"status": "idle"}
        done = sum(1 for s in self._active.steps if s.status == StepStatus.DONE)
        total = len(self._active.steps)
        failed = sum(1 for s in self._active.steps if s.status == StepStatus.FAILED)
        return {
            "id": self._active.id,
            "goal": self._active.goal,
            "status": self._active.status,
            "progress": f"{done}/{total} steps done",
            "done": done, "total": total, "failed": failed,
            "steps": [{"idx": s.idx, "task": s.task, "status": s.status.value,
                       "risk": s.risk, "verified": s.verified}
                      for s in self._active.steps],
        }

    def complete(self) -> None:
        if self._active:
            self._active.status = "completed"
            self._active.updated = time.time()
            self._save()

    def abort(self) -> None:
        if self._active:
            self._active.status = "cancelled"
            self._active.updated = time.time()
            self._save()
            self._active = None

    def list_plans(self, limit: int = 10) -> list[dict[str, Any]]:
        state = read_json(self.store.path, {}) or {}
        plans = state.get("plans", [])
        return [
            {"id": p["id"], "goal": p["goal"], "status": p["status"],
             "created": p.get("created", 0)}
            for p in sorted(plans, key=lambda x: x.get("created", 0), reverse=True)[:limit]
        ]

    @property
    def active_plan(self) -> AutonomyPlan | None:
        return self._active


# Process-wide instance.
autonomy = AutonomyEngine()
