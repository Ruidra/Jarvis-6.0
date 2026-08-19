"""
Jarvis Plugin — Quiz / Study Mode.

An advanced learning skill. Ask JARVIS to quiz you on any topic; it generates a
short multiple-choice / short-answer quiz via the local LLM, stores it, and can
grade your answers and track a running score.

Triggers (spoken): "quiz me on ...", "test me", "study mode", "pop quiz".

Requires a local LLM (Ollama or OpenAI-compatible) configured in
config/api_keys.json (``llm_provider`` / ``llm_url`` / ``llm_model``). If the LLM
is unavailable the plugin reports it clearly instead of crashing.

State lives in ``memory/quiz.json`` (atomic, via core.json_store).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path

from core.json_store import JsonStore, read_json, atomic_write_json

logger = logging.getLogger("jarvis.plugin.quiz")

PLUGIN = {
    "name": "quiz",
    "description": (
        "Interactive quiz / study mode. Generates a short quiz on any topic using the "
        "local LLM, then grades the user's answers and tracks a running score. "
        "Use when the user says 'quiz me', 'test me on X', 'study mode', 'pop quiz', "
        "or asks to be tested / learn something through questions."
    ),
    "triggers": ["quiz", "test me", "study mode", "pop quiz", "exam"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "generate | answer | score | list (default: generate)",
            },
            "topic": {
                "type": "STRING",
                "description": "Subject to quiz on (for generate). e.g. 'quantum physics'.",
            },
            "count": {
                "type": "INTEGER",
                "description": "Number of questions to generate (default 5, max 10).",
            },
            "answer": {
                "type": "STRING",
                "description": "The user's free-text answer when action='answer'.",
            },
            "quiz_id": {
                "type": "STRING",
                "description": "Optional specific quiz id to answer/score. Defaults to the latest.",
            },
        },
        "required": [],
    },
}


def _store() -> JsonStore:
    base = Path(__file__).resolve().parent.parent
    return JsonStore(base / "memory" / "quiz.json")


def _llm(prompt: str, system: str | None = None) -> str:
    # Prefer the local LLM (Ollama / OpenAI-compatible); fall back to Gemini
    # (the same key the assistant already uses) so quizzes work zero-config.
    try:
        from core.llm_client import call_llm_text

        return call_llm_text(prompt, system=system, timeout=120)
    except Exception:  # noqa: BLE001
        from core.gemini_text import generate

        return generate(prompt, system=system, timeout=120)


def _extract_json(text: str) -> dict | None:
    try:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        logger.warning("quiz json parse failed: %s", e)
    return None


def _generate(topic: str, count: int) -> str:
    count = max(1, min(int(count), 10))
    system = (
        "You are a rigorous tutor. Produce a quiz as a JSON object ONLY, no prose. "
        "Schema: {\"title\": str, \"questions\": [{\"q\": str, \"a\": str, "
        "\"explanation\": str}]}. Each 'a' is the concise correct answer. "
        "Questions should test understanding, not trivia."
    )
    prompt = f"Create a {count}-question quiz on: {topic}."
    try:
        raw = _llm(prompt, system=system)
    except Exception as e:  # noqa: BLE001
        return (f"I couldn't generate a quiz — the local LLM isn't reachable "
                f"({e}). Start Ollama or configure an OpenAI-compatible server "
                f"in config/api_keys.json.")

    data = _extract_json(raw)
    if not data or not isinstance(data.get("questions"), list):
        return ("I generated something but couldn't structure it into a quiz. "
                "Check that your local LLM is running and try again.")

    qid = uuid.uuid4().hex[:8]
    now = time.time()
    store = _store()
    state = read_json(store.path, {}) or {}
    state.setdefault("quizzes", {})
    state["quizzes"][qid] = {
        "id": qid,
        "title": data.get("title", topic),
        "topic": topic,
        "created": now,
        "questions": data["questions"],
        "answered": False,
        "score": None,
    }
    state["current"] = qid
    atomic_write_json(store.path, state)

    lines = [f"📝 Quiz #{qid} — {data.get('title', topic)} ({len(data['questions'])} questions):"]
    for i, item in enumerate(data["questions"], 1):
        lines.append(f"{i}. {item.get('q', '')}")
    lines.append("\nReply with your answers and I'll grade them (or say 'grade my quiz').")
    return "\n".join(lines)


def _answer(answer_text: str, quiz_id: str | None) -> str:
    store = _store()
    state = read_json(store.path, {}) or {}
    quizzes = state.get("quizzes", {})
    if not quizzes:
        return "There's no active quiz to answer. Ask me to 'quiz you on' a topic first."
    qid = quiz_id or state.get("current")
    quiz = quizzes.get(qid)
    if not quiz:
        return f"I couldn't find quiz '{qid}'. Available: {', '.join(quizzes)}."

    questions = quiz.get("questions", [])
    system = (
        "You are a fair grader. Given a question, the model answer, and the student's "
        "answer, return ONLY JSON: {\"correct\": bool, \"points\": int (0-1), "
        "\"feedback\": str}. Be lenient with phrasing but strict on meaning."
    )
    total = 0.0
    parts = [f"📊 Grading quiz #{qid} — {quiz.get('title', '')}:"]
    for i, item in enumerate(questions, 1):
        prompt = (
            f"Q{i}: {item.get('q','')}\n"
            f"Model answer: {item.get('a','')}\n"
            f"Student answer: {answer_text}\n"
            "(Grade this single question against the student's overall answer.)"
        )
        try:
            res = _extract_json(_llm(prompt, system=system)) or {}
        except Exception:  # noqa: BLE001
            res = {}
        pts = float(res.get("points", 0) or 0)
        total += min(1.0, max(0.0, pts))
        mark = "✅" if pts >= 0.5 else "❌"
        parts.append(f"{mark} Q{i} ({pts:.0%}): {res.get('feedback', '')}")
    score = round(100 * total / max(1, len(questions)))
    quiz["answered"] = True
    quiz["score"] = score
    quizzes[qid] = quiz
    state["quizzes"] = quizzes
    atomic_write_json(store.path, state)
    parts.append(f"\nFinal score: {score}% on {len(questions)} questions.")
    return "\n".join(parts)


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    action = (args.get("action") or "generate").lower().strip()
    user = (ctx.get("user_name") or "sir").title()

    if action == "list":
        store = _store()
        state = read_json(store.path, {}) or {}
        quizzes = state.get("quizzes", {})
        if not quizzes:
            return "No quizzes yet. Ask me to quiz you on a topic."
        out = [f"Saved quizzes ({len(quizzes)}):"]
        for q in sorted(quizzes.values(), key=lambda x: -x.get("created", 0)):
            sc = q.get("score")
            out.append(f"• #{q['id']} {q.get('title','')} — "
                       f"{'answered, score '+str(sc)+'%' if sc is not None else 'not yet answered'}")
        return "\n".join(out)

    if action in ("answer", "grade", "score"):
        ans = args.get("answer") or ""
        if not ans:
            return "Tell me your answers and I'll grade them, e.g. 'my answers are ...'."
        return _answer(ans, args.get("quiz_id"))

    # default: generate
    topic = (args.get("topic") or "").strip()
    if not topic:
        # try to pull a topic from the raw intent text
        topic = re.sub(r"(?i)(quiz|test|study|me|on|about|pop|exam)", "", intent or "").strip(" .,'")
    if not topic:
        return (f"Sure {user} — what topic should I quiz you on? "
                f"e.g. 'quiz me on World War 2'.")
    return _generate(topic, int(args.get("count") or 5))
