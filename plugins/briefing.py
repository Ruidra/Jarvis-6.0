"""
Jarvis Plugin — Daily / Evening Briefing (autonomous orchestrator).

The missing god-level capability: instead of the user asking for weather,
then calendar, then markets one by one, this plugin pulls everything into one
concise spoken briefing and synthesises it with the LLM.

It chains existing plugins + actions:
  - calendar (today / upcoming)
  - weather_report (live weather)
  - markets (BTC / ETH prices)
  - habits (today's summary)
  - web_search (top headlines)
Then composes a natural briefing via the local LLM (Gemini fallback).

Usage:
  "give me my briefing"
  "morning brief"
  "evening brief"
  "status report"
"""

from __future__ import annotations

import logging

logger = logging.getLogger("jarvis.plugin.briefing")

PLUGIN = {
    "name": "briefing",
    "description": (
        "Autonomous daily/evening briefing. Pulls calendar, weather, markets, news, "
        "and habits into one concise spoken report. Use when the user says 'give me "
        "my briefing', 'morning brief', 'evening brief', or 'status report'."
    ),
    "triggers": ["briefing", "daily brief", "morning brief", "evening brief",
                 "my briefing", "status report", "my day"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING",
                     "description": "morning | evening | full (default: full)"},
            "city": {"type": "STRING",
                     "description": "City for weather (optional; pulled from memory if omitted)."},
        },
        "required": [],
    },
}


def _llm(prompt: str, system: str | None = None) -> str:
    try:
        from core.llm_client import call_llm_text
        return call_llm_text(prompt, system=system, timeout=120)
    except Exception:  # noqa: BLE001
        from core.gemini_text import generate
        return generate(prompt, system=system, timeout=120)


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        logger.debug("briefing sub-call failed: %s", e)
        return None


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    reg = ctx.get("plugins")
    mode = (args.get("mode") or "full").lower().strip()
    city = (args.get("city") or "").strip()

    # 1. Calendar
    cal = _safe(reg.dispatch, "calendar", {"action": "today"}, ctx) if reg else None
    cal = (cal or "").strip()
    if cal.startswith("Your calendar is empty"):
        cal = "No calendar events today."

    # 2. Weather
    weather = ""
    try:
        from actions.weather_report import weather_action
        params = {"city": city} if city else {}
        w = _safe(weather_action, parameters=params, player=None)
        weather = (w or "").strip()
    except Exception:
        weather = ""

    # 3. Markets
    markets = []
    for coin in ("bitcoin", "ethereum"):
        txt = _safe(reg.dispatch, "markets",
                    {"action": "price", "coin": coin, "vs": "usd"}, ctx) if reg else None
        txt = (txt or "").strip()
        if txt and not txt.startswith("I couldn't"):
            markets.append(txt)
    markets_text = " | ".join(markets) if markets else ""

    # 4. Habits
    habits = _safe(reg.dispatch, "habit", {"action": "summary"}, ctx) if reg else None
    habits = (habits or "").strip()
    if "Nothing logged" in habits:
        habits = "No habits logged yet today."

    # 5. News headlines
    news = ""
    try:
        from actions.web_search import web_search as web_search_action
        n = _safe(web_search_action,
                  parameters={"query": "top news today", "mode": "news"}, player=None)
        news = (n or "").strip()
    except Exception:
        news = ""
    if not news:
        news = "No news available right now."
    # Trim to ~2 lines for the briefing prompt
    news_lines = [ln.strip() for ln in news.splitlines() if ln.strip()]
    news_short = "\n".join(news_lines[:6])

    # 6. Synthesize
    sys = (
        "You are JARVIS composing a concise, premium personal briefing. "
        "Be brief, structured, and natural. Use short paragraphs. "
        "Prefer bullets. Address the user as 'sir' (English) or 'efendim' (Turkish) — "
        "but keep the entire briefing in the SAME language as the user's last message "
        "(you'll see it in the transcript). If unknown, default to English."
    )
    prompt = (
        f"Compose a {'morning' if mode == 'morning' else 'evening' if mode == 'evening' else 'full'} briefing.\n\n"
        f"[Calendar]\n{cal or 'No events.'}\n\n"
        f"[Weather]\n{weather or 'Not available.'}\n\n"
        f"[Markets]\n{markets_text or 'Not available.'}\n\n"
        f"[Habits today]\n{habits or 'None logged.'}\n\n"
        f"[Top headlines]\n{news_short or 'None.'}\n\n"
        "Write a tight, useful briefing. Keep it under 120 words."
    )
    try:
        text = _llm(prompt, system=sys)
    except Exception as e:  # noqa: BLE001
        return f"I couldn't compose the briefing right now ({e})."

    # Fallback if model produced nothing
    if not text:
        parts = []
        if cal and "No calendar" not in cal:
            parts.append(cal)
        if weather:
            parts.append(weather)
        if markets_text:
            parts.append(markets_text)
        if habits and "No habits" not in habits:
            parts.append(habits)
        if news_lines:
            parts.append("Headlines: " + "; ".join(news_lines[:3]))
        text = "\n\n".join(parts) or "All systems quiet. Nothing to report."

    header = {
        "morning": "🌅 Good morning, sir. Here's your briefing:",
        "evening": "🌙 Evening briefing, sir:",
    }.get(mode, "📋 Your briefing:")
    return f"{header}\n\n{text}"
