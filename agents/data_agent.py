"""
agents/data_agent.py — JARVIS Data Agent.

Specialist for: data analysis, CSV/Excel processing, databases,
data visualization, statistical analysis, data cleaning, reports.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("jarvis.agent.data")


def _llm(prompt: str, system: str | None = None, timeout: int = 120) -> str:
    try:
        from core.llm_client import call_llm_text
        return call_llm_text(prompt, system=system, timeout=timeout)
    except Exception:
        from core.gemini_text import generate
        return generate(prompt, system=system, timeout=timeout)


def execute(task: str, context: str = "", workspace: str | None = None) -> dict[str, Any]:
    """Execute a data analysis task."""
    start = time.time()

    system = (
        "You are JARVIS's Data Agent — an expert data analyst and scientist. "
        "You analyze data, create visualizations, clean datasets, and produce insights. "
        "You write efficient Python code using pandas, numpy, matplotlib, and other data tools. "
        "For analysis: provide key statistics, trends, and insights. "
        "For visualization: describe the chart type, axes, and key data points. "
        "For data cleaning: identify issues and provide corrected data or code. "
        "Return your response as:\n"
        "1. Brief summary of findings\n"
        "2. Detailed analysis with numbers and statistics\n"
        "3. Code or data transformations if applicable\n"
        "4. Visualizations description or recommendations"
    )

    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nAnalyze this data task."

    try:
        result = _llm(prompt, system=system, timeout=180)
        elapsed = time.time() - start
        return {
            "agent": "data",
            "status": "success",
            "result": result,
            "elapsed_seconds": round(elapsed, 1),
            "summary": _extract_summary(result),
        }
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Data agent failed: %s", e)
        return {
            "agent": "data",
            "status": "error",
            "result": f"Data agent failed: {e}",
            "elapsed_seconds": round(elapsed, 1),
            "summary": "Failed to complete data task.",
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
