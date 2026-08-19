"""
agents/video_agent.py — JARVIS Video Agent.

Specialist for: video scripts, storyboards, editing plans, thumbnail concepts,
video summaries, YouTube content strategy.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("jarvis.agent.video")


def _llm(prompt: str, system: str | None = None, timeout: int = 120) -> str:
    try:
        from core.llm_client import call_llm_text
        return call_llm_text(prompt, system=system, timeout=timeout)
    except Exception:
        from core.gemini_text import generate
        return generate(prompt, system=system, timeout=timeout)


def execute(task: str, context: str = "", workspace: str | None = None) -> dict[str, Any]:
    """Execute a video-related task."""
    start = time.time()

    system = (
        "You are JARVIS's Video Agent — an expert video producer, scriptwriter, and content strategist. "
        "You create compelling video scripts, storyboards, editing plans, and content strategies. "
        "For scripts: include hook, intro, main content, call-to-action, with timing estimates. "
        "For storyboards: describe each scene with visual, audio, and text details. "
        "For editing plans: specify cuts, transitions, effects, pacing. "
        "For thumbnails: describe composition, text, colors, and visual hierarchy. "
        "Return your response as:\n"
        "1. Brief summary\n"
        "2. Main deliverable (script/storyboard/plan)\n"
        "3. Production notes and tips"
    )

    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nCreate the video deliverable."

    try:
        result = _llm(prompt, system=system, timeout=180)
        elapsed = time.time() - start
        return {
            "agent": "video",
            "status": "success",
            "result": result,
            "elapsed_seconds": round(elapsed, 1),
            "summary": _extract_summary(result),
        }
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Video agent failed: %s", e)
        return {
            "agent": "video",
            "status": "error",
            "result": f"Video agent failed: {e}",
            "elapsed_seconds": round(elapsed, 1),
            "summary": "Failed to complete video task.",
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
