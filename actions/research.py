"""
Deep research for JARVIS.

Performs a multi-angle, source-grounded investigation of a topic using
DuckDuckGo web search (requires no Gemini key), then compiles a structured,
cited report. Optionally synthesises an executive summary with Gemini when a
key is available.
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

try:
    from .web_search import _ddg_search
except ImportError:  # allow running this module directly
    from actions.web_search import _ddg_search


def _get_api_key() -> str | None:
    from core.security import safe_read_config
    return safe_read_config().get("gemini_api_key")


_RESEARCH_ANGLES = [
    "overview, definition and basics",
    "latest news and recent developments 2025 2026",
    "key benefits, advantages and real-world use cases",
    "main risks, challenges, criticisms and limitations",
    "important statistics, data and facts",
    "expert opinions and future outlook",
]


def _dedupe(urls: list[str]) -> list[str]:
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect(topic: str, depth: int) -> tuple[str, list[str]]:
    angles = _RESEARCH_ANGLES[: max(3, min(depth, len(_RESEARCH_ANGLES)))]
    sections: list[str] = []
    all_sources: list[str] = []
    used_snippets: set[str] = set()

    for angle in angles:
        query = f"{topic} {angle}"
        try:
            results = _ddg_search(query, max_results=5)
        except Exception as e:
            print(f"[Research] search failed for {query!r}: {e}")
            results = []

        lines = [f"### {angle.capitalize()}"]
        for r in results:
            snip = (r.get("snippet") or "").strip()
            url = r.get("url") or ""
            if url:
                all_sources.append(url)
            if snip and snip.lower() not in used_snippets:
                used_snippets.add(snip.lower())
                lines.append(f"  - {snip}")
        if len(lines) == 1:
            lines.append("  - No specific results found.")
        sections.append("\n".join(lines))

    return "\n\n".join(sections), _dedupe(all_sources)


def _synthesize(topic: str, report: str) -> str | None:
    key = _get_api_key()
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-flash-latest",
            contents=(
                f"Write a concise 3-4 sentence executive summary of the following "
                f"research notes about '{topic}'. Be neutral and factual.\n\n{report}"
            ),
        )
        return (resp.text or "").strip() or None
    except Exception as e:
        print(f"[Research] synthesis skipped ({e})")
        return None


def deep_research(parameters: dict, player=None) -> str:
    params = parameters or {}
    topic = (params.get("topic") or "").strip()
    if not topic:
        return "Please specify a topic to research."

    try:
        depth = int(params.get("depth") or 6)
    except (TypeError, ValueError):
        depth = 6

    if player:
        player.write_log(f"[Research] Investigating: {topic}")

    report, sources = _collect(topic, depth)

    header = f"DEEP RESEARCH REPORT - {topic.upper()}\n" + "=" * 60 + "\n"
    summary = _synthesize(topic, report)

    body = ""
    if summary:
        body += "EXECUTIVE SUMMARY\n" + "-" * 60 + "\n" + summary + "\n\n"
    body += "FINDINGS BY ANGLE\n" + "-" * 60 + "\n" + report

    sources_block = (
        "\n\nSOURCES (" + str(len(sources)) + ")\n" + "-" * 60 + "\n"
        + ("\n".join(f"{i+1}. {u}" for i, u in enumerate(sources[:15])) or "  (none)")
    )

    full = header + body + sources_block
    if player:
        player.write_log(f"[Research] Done - {len(sources)} sources.")
    return full
