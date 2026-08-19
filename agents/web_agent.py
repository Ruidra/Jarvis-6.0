"""
agents/web_agent.py — JARVIS Web Development Agent (Enhanced).

Specialist for: websites, landing pages, web apps, HTML/CSS/JS, React,
frontend/backend, deployment. Now with real file creation and tool usage.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.agent.web")

_BASE = Path(__file__).resolve().parent.parent


def _llm(prompt: str, system: str | None = None, timeout: int = 120) -> str:
    try:
        from core.llm_client import call_llm_text
        return call_llm_text(prompt, system=system, timeout=timeout)
    except Exception:
        from core.gemini_text import generate
        return generate(prompt, system=system, timeout=timeout)


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.debug("web_agent sub-call failed: %s", e)
        return None


def execute(task: str, context: str = "", workspace: str | None = None) -> dict[str, Any]:
    """Execute a web development task and return structured result."""
    start = time.time()
    ws = Path(workspace) if workspace else _BASE

    # Phase 1: Plan the website structure
    plan = _plan_website(task, context)
    if not plan["success"]:
        return {
            "agent": "web",
            "status": "error",
            "result": plan["error"],
            "saved_path": None,
            "elapsed_seconds": round(time.time() - start, 1),
            "summary": "Failed to plan website.",
        }

    # Phase 2: Build the files
    build_result = _build_website(plan["plan"], task, ws)
    elapsed = time.time() - start

    return {
        "agent": "web",
        "status": "success" if build_result["success"] else "error",
        "result": build_result["message"],
        "saved_path": build_result.get("path"),
        "files_created": build_result.get("files", []),
        "elapsed_seconds": round(elapsed, 1),
        "summary": build_result.get("summary", "Website build complete."),
    }


def _plan_website(task: str, context: str) -> dict[str, Any]:
    """Use LLM to plan the website structure."""
    system = (
        "You are a web development planner. Given a task, output a JSON plan for the website. "
        "The JSON must have this exact format:\n"
        '{"type": "single" | "multi", "pages": [{"name": "index", "title": "Page Title"}], '
        '"features": ["responsive", "dark_mode"], "framework": "html" | "react" | "vue"}\n'
        "Be concise. Return ONLY valid JSON, no markdown, no explanation."
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nOutput the website plan as JSON."

    try:
        raw = _llm(prompt, system=system, timeout=60)
        # Extract JSON from response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {"success": False, "error": "LLM did not return valid JSON plan."}
        plan = json.loads(raw[start:end + 1])
        return {"success": True, "plan": plan}
    except Exception as e:
        return {"success": False, "error": f"Planning failed: {e}"}


def _build_website(plan: dict, task: str, ws: Path) -> dict[str, Any]:
    """Build the actual website files."""
    try:
        system = (
            "You are an expert web developer. Build a complete, production-ready website. "
            "Output ONLY valid HTML with embedded CSS and JavaScript. "
            "Make it modern, responsive, and visually appealing. "
            "Include: semantic HTML5, modern CSS (flexbox/grid, custom properties), "
            "vanilla JavaScript for interactivity. "
            "No markdown, no explanation, just the raw HTML file content."
        )
        prompt = f"TASK: {task}\nPLAN: {json.dumps(plan)}\n\nBuild the complete HTML file now."

        html_content = _llm(prompt, system=system, timeout=180)
        if not html_content or "<" not in html_content:
            return {"success": False, "message": "LLM did not generate valid HTML."}

        # Clean up the response
        html_content = html_content.strip()
        if html_content.startswith("```"):
            lines = html_content.split("\n")
            html_content = "\n".join(lines[1:])
        if html_content.endswith("```"):
            html_content = html_content[:-3]
        html_content = html_content.strip()

        # Ensure it has proper HTML structure
        if "<!DOCTYPE html>" not in html_content and "<html" not in html_content:
            html_content = f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n<title>Website</title>\n</head>\n<body>\n{html_content}\n</body>\n</html>"

        # Save the file
        out_dir = ws / "web_output"
        out_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        out_path = out_dir / f"web_{ts}.html"
        out_path.write_text(html_content, encoding="utf-8")

        # Try to create additional pages if multi-page
        files_created = [str(out_path)]
        if plan.get("type") == "multi" and len(plan.get("pages", [])) > 1:
            for page in plan["pages"][1:]:
                page_name = page.get("name", "page")
                page_path = out_dir / f"web_{ts}_{page_name}.html"
                page_content = _generate_page(task, page, plan, html_content)
                page_path.write_text(page_content, encoding="utf-8")
                files_created.append(str(page_path))

        return {
            "success": True,
            "message": f"Website built successfully with {len(files_created)} file(s).",
            "path": str(out_path),
            "files": files_created,
            "summary": f"Built {'multi-page' if plan.get('type') == 'multi' else 'single-page'} website with {len(files_created)} file(s).",
        }
    except Exception as e:
        logger.error("Website build failed: %s", e)
        return {"success": False, "message": f"Build failed: {e}", "files": []}


def _generate_page(task: str, page: dict, plan: dict, base_html: str) -> str:
    """Generate a variation of the base HTML for additional pages."""
    system = (
        "You are an expert web developer. Given a base HTML page and a new page spec, "
        "create a new HTML page that matches the style but with different content. "
        "Output ONLY the raw HTML, no markdown, no explanation."
    )
    prompt = f"TASK: {task}\nPAGE: {json.dumps(page)}\nBASE HTML (for style reference):\n{base_html[:2000]}\n\nGenerate the new page HTML."

    try:
        result = _llm(prompt, system=system, timeout=60)
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0]
        return result.strip()
    except Exception:
        return base_html


def _extract_summary(result: str) -> str:
    if not result:
        return "No output generated."
    lines = result.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and not line.startswith("#") and not line.startswith("```"):
            return line[:120]
    return lines[0][:120] if lines else "Task completed."
