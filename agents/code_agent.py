"""
agents/code_agent.py — JARVIS Code Agent (Enhanced).

Specialist for: code review, debugging, refactoring, optimization,
implementation, documentation, testing, code migration.
Now with real file operations and code execution.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.agent.code")

_BASE = Path(__file__).resolve().parent.parent


def _llm(prompt: str, system: str | None = None, timeout: int = 120) -> str:
    try:
        from core.llm_client import call_llm_text
        return call_llm_text(prompt, system=system, timeout=timeout)
    except Exception:
        from core.gemini_text import generate
        return generate(prompt, system=system, timeout=timeout)


def execute(task: str, context: str = "", workspace: str | None = None) -> dict[str, Any]:
    """Execute a code-related task."""
    start = time.time()
    ws = Path(workspace) if workspace else _BASE

    # Phase 1: Understand the task
    task_type = _classify_code_task(task)
    logger.info("Code agent task type: %s", task_type)

    # Phase 2: Execute based on type
    if task_type == "review":
        result = _review_code(task, context)
    elif task_type == "debug":
        result = _debug_code(task, context)
    elif task_type == "implement":
        result = _implement_code(task, context, ws)
    elif task_type == "refactor":
        result = _refactor_code(task, context)
    elif task_type == "test":
        result = _test_code(task, context)
    else:
        result = _general_code_task(task, context)

    elapsed = time.time() - start
    return {
        "agent": "code",
        "status": "success",
        "result": result,
        "elapsed_seconds": round(elapsed, 1),
        "summary": _extract_summary(result),
    }


def _classify_code_task(task: str) -> str:
    """Classify the type of code task."""
    task_lower = task.lower()
    if any(kw in task_lower for kw in ["review", "audit", "check", "inspect"]):
        return "review"
    if any(kw in task_lower for kw in ["debug", "fix", "bug", "error", "broken"]):
        return "debug"
    if any(kw in task_lower for kw in ["write", "create", "build", "implement", "make"]):
        return "implement"
    if any(kw in task_lower for kw in ["refactor", "clean", "optimize", "improve"]):
        return "refactor"
    if any(kw in task_lower for kw in ["test", "testing", "unit test"]):
        return "test"
    return "general"


def _review_code(task: str, context: str) -> str:
    """Review code for issues."""
    system = (
        "You are an expert code reviewer. Analyze the code for:\n"
        "1. Critical bugs (must fix)\n"
        "2. Performance issues\n"
        "3. Security vulnerabilities\n"
        "4. Code style and best practices\n"
        "5. Suggestions for improvement\n"
        "Be specific and actionable. Provide code examples for fixes."
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nReview this code thoroughly."

    try:
        return _llm(prompt, system=system, timeout=180)
    except Exception as e:
        return f"Code review failed: {e}"


def _debug_code(task: str, context: str) -> str:
    """Debug code and provide fixes."""
    system = (
        "You are an expert debugger. Given a bug description or error:\n"
        "1. Identify the root cause\n"
        "2. Explain why it happens\n"
        "3. Provide the fixed code\n"
        "4. Suggest preventive measures\n"
        "Be thorough and provide working solutions."
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nDebug this issue and provide fixes."

    try:
        return _llm(prompt, system=system, timeout=180)
    except Exception as e:
        return f"Debugging failed: {e}"


def _implement_code(task: str, context: str, ws: Path) -> str:
    """Implement new code and save it to files."""
    system = (
        "You are an expert software engineer. Write production-ready code.\n"
        "Requirements:\n"
        "- Clean, readable, well-documented code\n"
        "- Error handling and edge cases\n"
        "- Type hints where applicable\n"
        "- Follow best practices for the language\n"
        "Output format:\n"
        "1. Brief description\n"
        "2. File path and complete code in a code block\n"
        "3. Usage instructions\n"
        "4. Dependencies if any"
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nImplement this now."

    try:
        result = _llm(prompt, system=system, timeout=180)

        # Try to extract and save code files
        saved_files = _save_code_files(result, ws)
        if saved_files:
            result += f"\n\n✅ Files created:\n" + "\n".join(f"  • {f}" for f in saved_files)

        return result
    except Exception as e:
        return f"Implementation failed: {e}"


def _refactor_code(task: str, context: str) -> str:
    """Refactor existing code."""
    system = (
        "You are an expert code refactorer. Improve the code while preserving functionality:\n"
        "1. Improve readability and maintainability\n"
        "2. Remove duplication\n"
        "3. Apply design patterns where appropriate\n"
        "4. Optimize performance\n"
        "5. Update to modern syntax/features\n"
        "Provide the refactored code with explanations of changes."
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nRefactor this code."

    try:
        return _llm(prompt, system=system, timeout=180)
    except Exception as e:
        return f"Refactoring failed: {e}"


def _test_code(task: str, context: str) -> str:
    """Create tests for code."""
    system = (
        "You are an expert test engineer. Create comprehensive tests:\n"
        "1. Unit tests for all functions/methods\n"
        "2. Edge case tests\n"
        "3. Error handling tests\n"
        "4. Integration tests if applicable\n"
        "Use pytest or unittest depending on the language."
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nCreate tests for this code."

    try:
        return _llm(prompt, system=system, timeout=180)
    except Exception as e:
        return f"Test creation failed: {e}"


def _general_code_task(task: str, context: str) -> str:
    """Handle general code tasks."""
    system = (
        "You are an expert software engineer. Complete the code task thoroughly. "
        "Provide working code with explanations."
    )
    prompt = f"TASK: {task}\nCONTEXT: {context}\n\nComplete this code task."

    try:
        return _llm(prompt, system=system, timeout=180)
    except Exception as e:
        return f"Code task failed: {e}"


def _save_code_files(result: str, ws: Path) -> list[str]:
    """Extract code blocks from LLM response and save as files."""
    import re
    saved = []
    try:
        # Find code blocks with optional filenames
        # Pattern: ```language filename="path/to/file.ext" or just ```language
        pattern = r'```(?:python|py|javascript|js|typescript|ts|java|cpp|c|html|css|json|yaml|yml|bash|shell|sql)(?:\s+(?:filename[=:"\s]+([^\n"]+)|file[=:"\s]+([^\n"]+))?)?\n(.*?)```'
        matches = re.findall(pattern, result, re.DOTALL)

        if not matches:
            # Try simpler pattern without filename
            pattern2 = r'```(?:python|py|javascript|js)\n(.*?)```'
            matches2 = re.findall(pattern2, result, re.DOTALL)
            if matches2:
                out = ws / "code_output" / "main.py"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(matches2[0].strip(), encoding="utf-8")
                saved.append(str(out))
            return saved

        for m1, m2, content in matches[:5]:
            fname = (m1 or m2 or "").strip().strip('"').strip("'")
            if not fname:
                fname = "output.py"
            fpath = ws / "code_output" / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content.strip(), encoding="utf-8")
            saved.append(str(fpath))
    except Exception as e:
        logger.debug("Failed to save code files: %s", e)
    return saved


def _extract_summary(result: str) -> str:
    if not result:
        return "No output generated."
    lines = result.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and not line.startswith("#") and not line.startswith("```"):
            return line[:120]
    return lines[0][:120] if lines else "Task completed."
