"""
agents/research_agent.py — JARVIS Research Agent.

Specialist for: deep research, fact-checking, reports, comparisons,
academic research, market research, competitive analysis.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("jarvis.agent.research")


def _llm(prompt: str, system: str | None = None, timeout: int = 120) -> str:
    try:
        from core.llm_client import call_llm_text
        return call_llm_text(prompt, system=system, timeout=timeout)
    except Exception:
        from core.gemini_text import generate
        return generate(prompt, system=system, timeout=timeout)


def execute(task: str, context: str = "", workspace: str | None = None) -> dict[str, Any]:
    """Execute a research task."""
    start = time.time()

    system = (
        "You are JARVIS's Research Agent — an expert researcher and analyst. "
        "You conduct thorough, well-sourced research on any topic. "
        "You verify facts, compare options, and produce structured reports. "
        "Always cite sources when possible and distinguish between facts and opinions. "
        "Return your response as:\n"
        "1. Brief executive summary (2-3 sentences)\n"
        "2. Detailed findings with sections\n"
        "3. Key takeaways and recommendations"
    )

    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nConduct thorough research on this topic."

    try:
        result = _llm(prompt, system=system, timeout=180)
        elapsed = time.time() - start
        return {
            "agent": "research",
            "status": "success",
            "result": result,
            "elapsed_seconds": round(elapsed, 1),
            "summary": _extract_summary(result),
        }
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Research agent failed: %s", e)
        return {
            "agent": "research",
            "status": "error",
            "result": f"Research agent failed: {e}",
            "elapsed_seconds": round(elapsed, 1),
            "summary": "Failed to complete research task.",
        }


def _extract_summary(result: str) -> str:
    if not result:
        return "No output generated."
    lines = result.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and not line.startswith("#") and not line.startswith("```"):
            return line[:120]
    return lines[0][:120] if lines else "Task completed."
