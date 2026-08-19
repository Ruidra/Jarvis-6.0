"""
agents/photo_agent.py — JARVIS Photo/Image Generation Agent (Enhanced).

Specialist for: image generation, image analysis, design concepts,
illustrations, thumbnails, photo editing. Now with real image generation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.agent.photo")

_BASE = Path(__file__).resolve().parent.parent


def _llm(prompt: str, system: str | None = None, timeout: int = 120) -> str:
    try:
        from core.llm_client import call_llm_text
        return call_llm_text(prompt, system=system, timeout=timeout)
    except Exception:
        from core.gemini_text import generate
        return generate(prompt, system=system, timeout=timeout)


def execute(task: str, context: str = "", workspace: str | None = None) -> dict[str, Any]:
    """Execute an image/photo generation or analysis task."""
    start = time.time()
    ws = Path(workspace) if workspace else _BASE

    # Determine task type
    task_lower = task.lower()
    is_generation = any(kw in task_lower for kw in [
        "generate", "create", "make", "draw", "produce", "design", "illustrate"
    ])

    result = None
    generated_path = None

    if is_generation:
        # Phase 1: Create optimized prompt
        prompt_result = _create_image_prompt(task, context)
        if prompt_result["success"]:
            # Phase 2: Generate the image
            gen_result = _generate_image(prompt_result["prompt"], ws)
            generated_path = gen_result.get("path")
            result = gen_result.get("message") or prompt_result.get("prompt", "")
        else:
            result = prompt_result.get("error", "Failed to create image prompt.")
    else:
        # Analysis mode
        result = _analyze_image_request(task, context)

    elapsed = time.time() - start
    return {
        "agent": "photo",
        "status": "success" if result else "error",
        "result": result or "No output generated.",
        "generated_path": generated_path,
        "elapsed_seconds": round(elapsed, 1),
        "summary": _extract_summary(result),
    }


def _create_image_prompt(task: str, context: str) -> dict[str, Any]:
    """Create an optimized image generation prompt."""
    system = (
        "You are an expert image generation prompt engineer. "
        "Create detailed, production-ready prompts for AI image generators. "
        "Include: subject, style, lighting, composition, colors, mood, quality modifiers. "
        "Output ONLY the prompt text, nothing else."
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nCreate an optimized image generation prompt."

    try:
        result = _llm(prompt, system=system, timeout=60)
        return {"success": True, "prompt": result.strip()}
    except Exception as e:
        return {"success": False, "error": f"Prompt creation failed: {e}"}


def _generate_image(prompt: str, ws: Path) -> dict[str, Any]:
    """Generate an image using the image_gen action."""
    try:
        from actions.image_gen import image_generate
        result = image_generate(
            parameters={"prompt": prompt, "width": 1024, "height": 1024},
            response=None,
            player=None,
        )
        if result and "saved" in result.lower():
            # Extract path from result
            lines = result.splitlines()
            path = None
            for line in lines:
                if "saved" in line.lower() or "path" in line.lower():
                    parts = line.split(":")
                    if len(parts) > 1:
                        path = parts[1].strip()
                    break
            return {"success": True, "path": path, "message": result}
        return {"success": True, "path": None, "message": result or "Image generated."}
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        # Fallback: return the prompt so user can use it manually
        return {
            "success": False,
            "path": None,
            "message": f"Image generation failed: {e}\n\nOptimized prompt for manual use:\n{prompt}",
        }


def _analyze_image_request(task: str, context: str) -> str:
    """Analyze or describe an image-related request."""
    system = (
        "You are an expert image analyst and visual designer. "
        "Provide detailed analysis, descriptions, or design specifications. "
        "Be specific about composition, lighting, colors, mood, and technical details."
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nProvide your analysis or design."

    try:
        return _llm(prompt, system=system, timeout=120)
    except Exception as e:
        return f"Image analysis failed: {e}"


def _extract_summary(result: str) -> str:
    if not result:
        return "No output generated."
    lines = result.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and not line.startswith("#") and not line.startswith("```"):
            return line[:120]
    return lines[0][:120] if lines else "Task completed."
