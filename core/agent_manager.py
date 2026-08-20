"""
core/agent_manager.py — JARVIS Agent Manager (Manager of Specialists).

JARVIS acts as the manager. When a complex task arrives, the manager:
  1. Routes the task → picks the best specialist agent (keywords + LLM router)
  2. Dispatches it with full context, including what worked on similar past work
  3. Reviews the agent's output (heuristics + an LLM critic)
  4. Sends it back for revision when it isn't good enough
  5. Records the mission so routing and context keep improving

It can also run a **squad**: several specialists working the same objective in
parallel (``orchestrate_parallel``) or a decomposed multi-step plan
(``run_mission``), then merge everything into one deliverable.

Agents are workers. JARVIS is the manager. The user is the boss.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Iterable

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
    """Classify a task and return the best agent name, or None if no match.

    Keyword scoring first (instant, free). Ties and misses are resolved by the
    LLM router, and past success rates break remaining ties.
    """
    scores = keyword_scores(task)
    if scores:
        best = max(scores.values())
        winners = [name for name, s in scores.items() if s == best]
        if len(winners) == 1:
            return winners[0]
        # Tie → prefer the agent with the better track record.
        return _best_by_history(winners)
    return route_with_llm(task)


def keyword_scores(task: str) -> dict[str, int]:
    """Raw keyword match score per agent (longer keyword = more specific)."""
    task_lower = (task or "").lower()
    out: dict[str, int] = {}
    for agent_name, info in AGENT_REGISTRY.items():
        score = sum(len(kw) for kw in info["keywords"] if kw in task_lower)
        if score:
            out[agent_name] = score
    return out


def _best_by_history(candidates: Iterable[str]) -> str:
    names = list(candidates)
    try:
        from core.agent_memory import agent_scores

        stats = agent_scores()
        names.sort(
            key=lambda n: (
                stats.get(n, {}).get("success_rate", 0.0),
                stats.get(n, {}).get("runs", 0),
            ),
            reverse=True,
        )
    except Exception:
        pass
    return names[0]


def route_with_llm(task: str) -> str | None:
    """Ask the reasoning model which specialist fits. Returns None if unsure."""
    try:
        from core.brain import think_json

        roster = "\n".join(
            f"- {name}: {info['description']}" for name, info in AGENT_REGISTRY.items()
        )
        data = think_json(
            f"TASK:\n{task}\n\nSPECIALISTS:\n{roster}\n\n"
            'Reply as {"agent": "<name or none>", "confidence": 0.0-1.0, "why": "<8 words>"}',
            system=(
                "You route tasks to the single best specialist agent for JARVIS. "
                "Use 'none' only when no specialist fits at all."
            ),
            timeout=45,
            default=None,
        )
        if not isinstance(data, dict):
            return None
        name = str(data.get("agent", "")).strip().lower()
        if name in AGENT_REGISTRY and float(data.get("confidence", 1) or 0) >= 0.35:
            logger.info("LLM router chose '%s' (%s)", name, data.get("why", ""))
            return name
    except Exception as exc:  # noqa: BLE001 — routing must never block a mission
        logger.info("LLM routing unavailable: %s", exc)
    return None


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

    Past successful missions on similar tasks are injected into the agent's
    context automatically, so the workforce compounds experience.

    Returns a dict with:
      - agent: name of the agent used
      - status: success | error | fallback
      - result: the agent's output
      - summary: brief summary
      - elapsed_seconds: time taken
    """
    start = time.time()

    # Determine which agent to use
    agent_name = preferred_agent if preferred_agent in AGENT_REGISTRY else None
    if not agent_name:
        agent_name = classify_task(task)

    # Enrich the context with lessons from similar past missions.
    context = _with_experience(task, context)

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
        if not isinstance(result, dict):
            result = {"status": "success", "result": str(result), "summary": ""}
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


def _with_experience(task: str, context: str) -> str:
    """Prepend lessons from similar successful missions (if any)."""
    try:
        from core.agent_memory import context_for

        past = context_for(task)
        if past:
            return f"{context}\n\n{past}".strip()
    except Exception:
        pass
    return context


def _handle_fallback(task: str, context: str) -> str:
    """Fallback when no specific agent matches — use the unified brain."""
    system = (
        "You are JARVIS's general-purpose specialist. Complete the user's task "
        "to the best of your ability. Be thorough, accurate, and actionable: "
        "give concrete steps, code or content, not vague advice."
    )
    try:
        from core.brain import think

        return think(f"TASK: {task}\nCONTEXT: {context}", system=system, timeout=120)
    except Exception as e:
        return f"Could not process task: {e}"


def review_agent_output(task: str, agent_result: dict[str, Any], deep: bool = True) -> dict[str, Any]:
    """
    Manager review: is the agent's output complete and high quality?

    Two stages: cheap structural checks first (empty / error / TODO markers),
    then an LLM critic that judges the work against the original task. Returns
    ``{'approved', 'needs_revision', 'feedback', 'score'}``.
    """
    result_text = str(agent_result.get("result", "") or "")
    status = agent_result.get("status", "error")

    if status != "success":
        return {
            "approved": False,
            "needs_revision": True,
            "score": 0.0,
            "feedback": f"Agent returned error status. Result: {result_text[:200]}",
        }

    if not result_text or len(result_text.strip()) < 20:
        return {
            "approved": False,
            "needs_revision": True,
            "score": 0.0,
            "feedback": "Agent output is too short or empty. Task may not have been completed.",
        }

    # Quality checks (structural, instant)
    issues = []
    head = result_text.lower()[:300]
    for marker, msg in (
        ("failed:", "Output starts by reporting a failure"),
        ("could not", "Output says it could not do the work"),
        ("traceback", "Output contains a Python traceback"),
    ):
        if marker in head:
            issues.append(msg)
    if "TODO" in result_text or "FIXME" in result_text:
        issues.append("Output contains TODO/FIXME markers")
    if "<placeholder" in head or "lorem ipsum" in head:
        issues.append("Output contains placeholder content")

    if issues:
        return {
            "approved": False,
            "needs_revision": True,
            "score": 0.2,
            "feedback": "Issues found: " + "; ".join(issues),
        }

    if deep:
        verdict = _llm_critique(task, result_text)
        if verdict is not None:
            return verdict

    return {
        "approved": True,
        "needs_revision": False,
        "score": 0.8,
        "feedback": "Output looks complete and high quality.",
    }


def _llm_critique(task: str, result_text: str) -> dict[str, Any] | None:
    """LLM critic. Returns a verdict dict, or None when no model is reachable."""
    try:
        from core.brain import think_json

        data = think_json(
            f"ORIGINAL TASK:\n{task}\n\nAGENT OUTPUT:\n{result_text[:6000]}\n\n"
            'Judge it. Reply as {"score": 0.0-1.0, "complete": true/false, '
            '"missing": ["..."], "feedback": "<one actionable sentence>"}',
            system=(
                "You are a demanding but fair quality reviewer for JARVIS. "
                "Score 0.8+ only when the output fully accomplishes the task and "
                "is directly usable. Penalise vagueness, missing deliverables and "
                "unfinished work."
            ),
            timeout=60,
            default=None,
        )
        if not isinstance(data, dict) or "score" not in data:
            return None
        score = float(data.get("score", 0) or 0)
        complete = bool(data.get("complete", score >= 0.7))
        missing = data.get("missing") or []
        feedback = str(data.get("feedback", "")).strip() or "No feedback provided."
        if isinstance(missing, list) and missing:
            feedback += " Missing: " + "; ".join(str(m) for m in missing[:4])
        approved = complete and score >= 0.7
        return {
            "approved": approved,
            "needs_revision": not approved,
            "score": round(score, 2),
            "feedback": feedback,
        }
    except Exception as exc:  # noqa: BLE001 — critique is a bonus, not a gate
        logger.info("LLM critique unavailable: %s", exc)
        return None


def orchestrate(task: str, context: str = "", workspace: str | None = None, max_revisions: int = 1, preferred_agent: str | None = None) -> dict[str, Any]:
    """
    Full manager workflow: dispatch → review → revise if needed → return final result.

    This is the main entry point for JARVIS to delegate tasks to specialist agents.
    Every completed mission is recorded in agent memory so routing, context and
    quality keep improving over time.
    """
    start = time.time()

    # Step 1: Dispatch
    logger.info("Manager dispatching task: %s", task[:80])
    result = dispatch(task, context=context, workspace=workspace, preferred_agent=preferred_agent)

    # Step 2: Review
    review = review_agent_output(task, result)
    best = (result, review)

    # Step 3: Revise if needed (keeping the best attempt seen so far)
    revision_count = 0
    while not review["approved"] and revision_count < max_revisions:
        revision_count += 1
        logger.info("Manager requesting revision %d: %s", revision_count, review["feedback"])

        revised_task = (
            f"{task}\n\nFEEDBACK FROM MANAGER: {review['feedback']}\n"
            "Fix exactly these problems and deliver the complete result."
        )
        result = dispatch(
            revised_task, context=context, workspace=workspace,
            preferred_agent=result.get("agent") or preferred_agent,
        )
        review = review_agent_output(task, result)
        if review.get("score", 0) >= best[1].get("score", 0):
            best = (result, review)

    result, review = best
    elapsed = time.time() - start
    result["revisions"] = revision_count
    result["review"] = review
    result["total_elapsed_seconds"] = round(elapsed, 1)

    _remember(task, result, review, elapsed, revision_count)

    logger.info(
        "Manager completed task via %s agent in %.1fs (%d revisions, score %.2f)",
        result.get("agent"), elapsed, revision_count, review.get("score", 0),
    )
    return result


def _remember(task: str, result: dict, review: dict, elapsed: float, revisions: int) -> None:
    try:
        from core.agent_memory import record

        record(
            task=task,
            agent=result.get("agent"),
            status=str(result.get("status", "")),
            approved=bool(review.get("approved")),
            summary=str(result.get("summary", "")),
            result=str(result.get("result", "")),
            elapsed=elapsed,
            revisions=revisions,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("agent memory record failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# squad mode — several specialists at once
# ──────────────────────────────────────────────────────────────────────────────
def orchestrate_parallel(
    tasks: list[str] | list[dict[str, Any]],
    context: str = "",
    workspace: str | None = None,
    max_workers: int = 4,
    merge: bool = True,
) -> dict[str, Any]:
    """Run several missions **simultaneously** and merge the results.

    ``tasks`` may be plain strings, or dicts like
    ``{"task": "...", "agent": "web", "context": "..."}``.
    """
    start = time.time()
    normalised: list[dict[str, Any]] = []
    for item in tasks:
        if isinstance(item, dict):
            normalised.append({
                "task": str(item.get("task", "")).strip(),
                "agent": item.get("agent"),
                "context": item.get("context", ""),
            })
        else:
            normalised.append({"task": str(item).strip(), "agent": None, "context": ""})
    normalised = [t for t in normalised if t["task"]]
    if not normalised:
        return {"status": "error", "result": "No tasks given.", "missions": []}

    workers = max(1, min(int(max_workers), len(normalised)))
    missions: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                orchestrate,
                t["task"],
                context=(f"{context}\n{t['context']}").strip(),
                workspace=workspace,
                max_revisions=1,
                preferred_agent=t["agent"],
            ): t
            for t in normalised
        }
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {
                    "agent": t["agent"], "status": "error",
                    "result": f"Mission failed: {exc}", "summary": str(exc),
                    "review": {"approved": False, "score": 0.0, "feedback": str(exc)},
                }
            res["task"] = t["task"]
            missions.append(res)

    approved = [m for m in missions if m.get("review", {}).get("approved")]
    elapsed = time.time() - start

    combined = "\n\n".join(
        f"### {m.get('agent') or 'general'} — {m['task'][:90]}\n{str(m.get('result', ''))[:4000]}"
        for m in missions
    )
    final = combined
    if merge and len(missions) > 1:
        final = _merge_results(missions) or combined

    return {
        "status": "success" if approved else "partial",
        "agent": "squad",
        "result": final,
        "summary": (
            f"{len(approved)}/{len(missions)} missions approved in {elapsed:.1f}s "
            f"(parallel, {workers} workers)"
        ),
        "missions": [
            {
                "task": m["task"],
                "agent": m.get("agent"),
                "status": m.get("status"),
                "score": m.get("review", {}).get("score", 0),
                "summary": m.get("summary", ""),
            }
            for m in missions
        ],
        "total_elapsed_seconds": round(elapsed, 1),
    }


def _merge_results(missions: list[dict[str, Any]]) -> str:
    """Ask the brain to weave parallel mission outputs into one deliverable."""
    try:
        from core.brain import think

        blocks = "\n\n".join(
            f"--- {m.get('agent') or 'general'} on '{m['task'][:80]}' ---\n"
            f"{str(m.get('result', ''))[:3000]}"
            for m in missions
        )
        return think(
            f"Several specialists worked in parallel. Merge their work into ONE "
            f"coherent deliverable with no repetition, keeping every concrete "
            f"detail, path, code block and figure:\n\n{blocks}",
            system=(
                "You are JARVIS's editor. Produce the final consolidated answer. "
                "Keep it organised with short headings. Never invent facts."
            ),
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("merge unavailable: %s", exc)
        return ""


def plan_mission(goal: str, max_steps: int = 4) -> list[dict[str, Any]]:
    """Break a big objective into specialist sub-tasks (LLM planner)."""
    roster = ", ".join(AGENT_REGISTRY)
    try:
        from core.brain import think_json

        data = think_json(
            f"GOAL:\n{goal}\n\nAVAILABLE SPECIALISTS: {roster}\n\n"
            f"Split the goal into at most {max_steps} independent sub-tasks that "
            f"can run in parallel. Reply as "
            '{"steps": [{"task": "...", "agent": "<specialist or none>"}]}',
            system=(
                "You are JARVIS's mission planner. Prefer few, meaty, independent "
                "sub-tasks. Never create dependent steps that need each other's output."
            ),
            timeout=60,
            default=None,
        )
        steps = (data or {}).get("steps") if isinstance(data, dict) else None
        out: list[dict[str, Any]] = []
        for s in (steps or [])[:max_steps]:
            if not isinstance(s, dict):
                continue
            t = str(s.get("task", "")).strip()
            if not t:
                continue
            agent = str(s.get("agent", "")).strip().lower()
            out.append({"task": t, "agent": agent if agent in AGENT_REGISTRY else None})
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        logger.info("planner unavailable: %s", exc)
    return [{"task": goal, "agent": None}]


def run_mission(goal: str, context: str = "", workspace: str | None = None,
                max_steps: int = 4) -> dict[str, Any]:
    """Plan a goal into sub-tasks, run them as a parallel squad, merge the result."""
    steps = plan_mission(goal, max_steps=max_steps)
    if len(steps) == 1:
        res = orchestrate(steps[0]["task"], context=context, workspace=workspace,
                          preferred_agent=steps[0]["agent"])
        res["plan"] = steps
        return res
    res = orchestrate_parallel(steps, context=context, workspace=workspace)
    res["plan"] = steps
    return res


def workforce_report() -> str:
    """What the squad looks like and how well it has performed."""
    lines = ["JARVIS specialist workforce:"]
    for name, info in AGENT_REGISTRY.items():
        lines.append(f"  • {name}: {info['description']}")
    try:
        from core.agent_memory import summary

        lines.append("")
        lines.append(summary())
    except Exception:
        pass
    return "\n".join(lines)
