"""
Self-reflection / error recovery for Jarvis tools.

When an agentic step fails, Jarvis should not just give up — it should
reflect on the error and try a corrected plan.  :class:`SelfHealingExecutor`
runs a sequence of steps; when a step raises, it asks a *replanner* (an LLM by
default, rule-based fallback otherwise) to produce corrected arguments, and
retries up to ``max_iterations`` times.

Example::

    from core.self_healing import SelfHealingExecutor

    def open_app(args): ...
    exec = SelfHealingExecutor(replanner=my_llm_replanner)
    exec.run([{"fn": open_app, "args": {"name": "blender"}}])
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

Step = dict[str, Any]  # {"fn": Callable, "args": dict}


def _default_replanner(error: Exception, step: Step, history: list[dict]) -> Step:
    """Rule-based fallback: retry the same step, surfacing the error in args."""
    args = dict(step.get("args", {}))
    args["_last_error"] = str(error)
    args["_attempt"] = len(history) + 1
    return {"fn": step["fn"], "args": args}


class SelfHealingExecutor:
    def __init__(
        self,
        replanner: Callable[[Exception, Step, list[dict]], Step] | None = None,
        max_iterations: int = 3,
    ) -> None:
        self.replanner = replanner or _default_replanner
        self.max_iterations = max_iterations

    def run(self, steps: list[Step]) -> list[dict]:
        """Execute ``steps``; recover from per-step failures via the replanner.

        Returns a list of per-step results: {"ok": bool, "result" | "error", ...}.
        A step only fails permanently after ``max_iterations`` unsuccessful tries.
        """
        results: list[dict] = []
        for step in steps:
            ok, final, errs = self._run_step(step)
            results.append(
                {"ok": ok, "result": final if ok else None, "error": errs[-1] if errs else None}
            )
            if not ok:
                logger.error("Step failed permanently after %d tries: %s", self.max_iterations, step)
        return results

    def _run_step(self, step: Step) -> tuple[bool, Any, list[Exception]]:
        errors: list[Exception] = []
        current = dict(step)
        for _ in range(self.max_iterations):
            try:
                result = current["fn"](**current.get("args", {}))
                return True, result, errors
            except Exception as exc:  # noqa: BLE001 - we want to recover from anything
                errors.append(exc)
                logger.warning("Step attempt failed: %s", exc)
                current = self.replanner(exc, current, errors) or current
        return False, None, errors


def llm_replanner_factory(system_prompt: str | None = None) -> Callable:
    """Build a replanner that asks the local LLM to fix a failed tool call.

    Falls back to the rule-based default if the LLM backend is unavailable.
    """
    try:
        from core.llm_client import call_llm_text
    except Exception:  # noqa: BLE001
        return _default_replanner

    def replanner(error: Exception, step: Step, history: list[dict]) -> Step:
        fn_name = getattr(step.get("fn"), "__name__", "tool")
        prompt = (
            f"A tool call to '{fn_name}' failed with: {error}\n"
            f"Original args: {step.get('args')}\n"
            "Return ONLY a corrected JSON object of new arguments to retry with."
        )
        try:
            fix = call_llm_text(prompt, system=system_prompt or "You are a tool-call fixer.")
            import json

            args = json.loads(_extract_json(fix))
            return {"fn": step["fn"], "args": args}
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_replanner failed (%s); using rule-based fallback", exc)
            return _default_replanner(error, step, history)

    return replanner


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text
