"""
core/agent_manager.py — JARVIS Agent Manager (Manager of Specialists).

JARVIS acts as the manager. When a complex task arrives, the manager:
  1. Classifies the task → picks the best specialist agent
  2. Dispatches the task with full context
  3. Reviews the agent's output
  4. Polishes the final result and returns it to JARVIS for delivery

Agents are workers. JARVIS is the manager. The user is the boss.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("jarvis.agent_manager")

# Agent registry: name -> (module_path, description, keywords)
AGENT_REGISTRY: dict[str, dict[str, Any]] = {
    "web": {
        "module": "agents.web_agent",
        "description": "Web development: websites, landing pages, web apps, HTML/CSS/JS, React, frontend/backend, deployment",
        "keywords": ["website", "web", "landing", "page", "html", "css", "javascript", "js",
                     "react", "vue", "angular", "frontend", "backend", "web app", "site",
                     "wordpress", "shopify", "ecommerce", "online store", "webpage"],
        "priority": 10,
    },
    "photo": {
        "module": "agents.photo_agent",
        "description": "Image generation, analysis, design concepts, illustrations, thumbnails, photo editing",
        "keywords": ["image", "photo", "picture", "generate", "draw", "illustration",
                     "design", "thumbnail", "visual", "art", "painting", "photo edit",
                     "background", "banner", "logo", "graphic", "midjourney", "dall-e"],
        "priority": 10,
    },
    "video": {
        "module": "agents.video_agent",
        "description": "Video scripts, storyboards, editing plans, thumbnail concepts, YouTube content",
        "keywords": ["video", "youtube", "script", "storyboard", "editing", "reel",
                     "tiktok", "short", "thumbnail", "content plan", "video idea",
                     "podcast", "stream", "animation", "motion"],
        "priority": 9,
    },
    "app": {
        "module": "agents.app_agent",
        "description": "Desktop/mobile apps, installers, packaging, GUI apps, CLI tools, cross-platform",
        "keywords": ["app", "application", "desktop app", "mobile app", "software",
                     "program", "tool", "installer", "exe", "apk", "ipa",
                     "gui", "tkinter", "pyqt", "electron", "flutter", "kivy"],
        "priority": 10,
    },
    "code": {
        "module": "agents.code_agent",
        "description": "Code review, debugging, refactoring, optimization, implementation, testing",
        "keywords": ["code", "review", "debug", "bug", "fix", "refactor", "optimize",
                     "implement", "function", "class", "api", "library", "module",
                     "test", "unit test", "documentation", "docstring"],
        "priority": 10,
    },
    "research": {
        "module": "agents.research_agent",
        "description": "Deep research, reports, fact-checking, competitive analysis, academic research",
        "keywords": ["research", "report", "analyze", "analysis", "investigate",
                     "fact check", "compare", "comparison", "study", "survey",
                     "market research", "competitor", "trends", "data report"],
        "priority": 8,
    },
    "data": {
        "module": "agents.data_agent",
        "description": "Data analysis, CSV/Excel, databases, visualization, statistics, data cleaning",
        "keywords": ["data", "csv", "excel", "database", "sql", "visualization",
                     "chart", "graph", "statistics", "analytics", "pandas",
                     "dashboard", "bi", "business intelligence", "data science"],
        "priority": 9,
    },
}


def classify_task(task: str) -> str | None:
    """Classify a task and return the best agent name, or None if no match."""
    task_lower = task.lower()
    best_agent = None
    best_score = 0

    for agent_name, info in AGENT_REGISTRY.items():
        score = 0
        for kw in info["keywords"]:
            if kw in task_lower:
                score += len(kw)  # longer matches are more specific
        if score > best_score:
            best_score = score
            best_agent = agent_name

    return best_agent if best_score > 0 else None


def get_agent_info(agent_name: str) -> dict[str, Any] | None:
    """Get info about a specific agent."""
    return AGENT_REGISTRY.get(agent_name)


def list_agents() -> list[dict[str, Any]]:
    """List all available agents with their info."""
    return [
        {"name": name, "description": info["description"], "priority": info["priority"]}
        for name, info in AGENT_REGISTRY.items()
    ]


def dispatch(task: str, context: str = "", preferred_agent: str | None = None, workspace: str | None = None) -> dict[str, Any]:
    """
    Dispatch a task to the appropriate agent.

    Returns a dict with:
      - agent: name of the agent used
      - status: success | error | fallback
      - result: the agent's output
      - summary: brief summary
      - elapsed_seconds: time taken
    """
    start = time.time()

    # Determine which agent to use
    agent_name = preferred_agent
    if not agent_name:
        agent_name = classify_task(task)

    if not agent_name:
        return {
            "agent": None,
            "status": "fallback",
            "result": _handle_fallback(task, context),
            "summary": "No specific agent matched; using general handler.",
            "elapsed_seconds": round(time.time() - start, 1),
        }

    agent_info = AGENT_REGISTRY.get(agent_name)
    if not agent_info:
        return {
            "agent": None,
            "status": "error",
            "result": f"Agent '{agent_name}' not found in registry.",
            "summary": "Agent dispatch failed.",
            "elapsed_seconds": round(time.time() - start, 1),
        }

    # Import and execute the agent
    try:
        module_path = agent_info["module"]
        import importlib
        mod = importlib.import_module(module_path)
        result = mod.execute(task=task, context=context, workspace=workspace)
        result["agent"] = agent_name
        result["elapsed_seconds"] = round(time.time() - start, 1)
        return result
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Agent dispatch failed for %s: %s", agent_name, e)
        return {
            "agent": agent_name,
            "status": "error",
            "result": f"Agent '{agent_name}' failed: {e}",
            "summary": f"Agent execution failed: {e}",
            "elapsed_seconds": round(elapsed, 1),
        }


def _handle_fallback(task: str, context: str) -> str:
    """Fallback when no specific agent matches — use general LLM."""
    try:
        from core.llm_client import call_llm_text
        system = (
            "You are JARVIS's general-purpose assistant. Complete the user's task "
            "to the best of your ability using available tools and knowledge. "
            "Be thorough, accurate, and actionable."
        )
        return call_llm_text(f"TASK: {task}\nCONTEXT: {context}", system=system, timeout=120)
    except Exception as e:
        return f"Could not process task: {e}"


def review_agent_output(task: str, agent_result: dict[str, Any]) -> dict[str, Any]:
    """
    Manager review: check if the agent's output is complete and high quality.
    Returns feedback dict with 'approved', 'feedback', 'needs_revision'.
    """
    result_text = agent_result.get("result", "")
    status = agent_result.get("status", "error")

    if status != "success":
        return {
            "approved": False,
            "needs_revision": True,
            "feedback": f"Agent returned error status. Result: {result_text[:200]}",
        }

    if not result_text or len(result_text.strip()) < 20:
        return {
            "approved": False,
            "needs_revision": True,
            "feedback": "Agent output is too short or empty. Task may not have been completed.",
        }

    # Quality checks
    issues = []
    if "error" in result_text.lower()[:200]:
        issues.append("Output contains error messages")
    if "failed" in result_text.lower()[:200]:
        issues.append("Output indicates failure")
    if "TODO" in result_text or "FIXME" in result_text:
        issues.append("Output contains TODO/FIXME markers")

    if issues:
        return {
            "approved": False,
            "needs_revision": True,
            "feedback": "Issues found: " + "; ".join(issues),
        }

    return {
        "approved": True,
        "needs_revision": False,
        "feedback": "Output looks complete and high quality.",
    }


def orchestrate(task: str, context: str = "", workspace: str | None = None, max_revisions: int = 1, preferred_agent: str | None = None) -> dict[str, Any]:
    """
    Full manager workflow: dispatch → review → revise if needed → return final result.

    This is the main entry point for JARVIS to delegate tasks to specialist agents.
    """
    start = time.time()

    # Step 1: Dispatch
    logger.info("Manager dispatching task: %s", task[:80])
    result = dispatch(task, context=context, workspace=workspace, preferred_agent=preferred_agent)

    # Step 2: Review
    review = review_agent_output(task, result)

    # Step 3: Revise if needed
    revision_count = 0
    while not review["approved"] and revision_count < max_revisions:
        revision_count += 1
        logger.info("Manager requesting revision %d: %s", revision_count, review["feedback"])

        # Re-dispatch with feedback
        revised_task = f"{task}\n\nFEEDBACK FROM MANAGER: {review['feedback']}\nPlease improve and complete this task."
        result = dispatch(revised_task, context=context, workspace=workspace)
        review = review_agent_output(task, result)

    elapsed = time.time() - start
    result["revisions"] = revision_count
    result["review"] = review
    result["total_elapsed_seconds"] = round(elapsed, 1)

    logger.info(
        "Manager completed task via %s agent in %.1fs (%d revisions)",
        result.get("agent"), elapsed, revision_count
    )
    return result
