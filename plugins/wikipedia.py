"""
Jarvis Plugin — Wikipedia (instant knowledge).

Fetches a concise encyclopedia summary for any topic via Wikipedia's free REST
API (no key required). Great for "tell me about X", "who is Y", "what is Z".

Args:
  topic : the subject to look up
  sentences : max sentences to return (default 3, max 8)
"""

from __future__ import annotations

import logging
import urllib.parse

import requests

logger = logging.getLogger("jarvis.plugin.wikipedia")

PLUGIN = {
    "name": "wikipedia",
    "description": (
        "Fetch a concise encyclopedia summary from Wikipedia for any topic, person, "
        "or concept. Use for 'tell me about X', 'who is Y', 'what is Z', or general "
        "knowledge questions that aren't current events."
    ),
    "triggers": ["tell me about", "who is", "what is", "wikipedia", "explain"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "topic":     {"type": "STRING", "description": "Subject to look up."},
            "sentences": {"type": "INTEGER", "description": "Max sentences (default 3, max 8)."},
        },
        "required": ["topic"],
    },
}

_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_HEADERS = {"User-Agent": "Jarvis/1.0 (knowledge plugin)"}


def _summary(topic: str, sentences: int) -> str | None:
    title = urllib.parse.quote(topic.strip().replace(" ", "_"))
    try:
        r = requests.get(f"{_API}{title}", headers=_HEADERS, timeout=10)
        if r.status_code == 404:
            # try a search to resolve the title
            s = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": topic,
                        "format": "json", "srlimit": 1},
                headers=_HEADERS, timeout=10,
            ).json()
            hits = s.get("query", {}).get("search", [])
            if not hits:
                return None
            title = urllib.parse.quote(hits[0]["title"].replace(" ", "_"))
            r = requests.get(f"{_API}{title}", headers=_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        extract = data.get("extract") or ""
        if sentences and len(extract) > sentences * 120:
            # crude sentence cap
            parts = extract.split(". ")
            extract = ". ".join(parts[:sentences]).strip()
            if not extract.endswith("."):
                extract += "."
        return extract or None
    except Exception as e:  # noqa: BLE001
        logger.warning("wikipedia lookup failed: %s", e)
        return None


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    topic = (args.get("topic") or "").strip()
    if not topic:
        return "What topic should I look up on Wikipedia?"
    sentences = max(1, min(int(args.get("sentences") or 3), 8))
    try:
        out = _summary(topic, sentences)
    except Exception as e:  # noqa: BLE001
        return f"I couldn't reach Wikipedia right now ({e})."
    if not out:
        return f"I couldn't find a Wikipedia article for '{topic}'."
    return f"📚 {out}"
