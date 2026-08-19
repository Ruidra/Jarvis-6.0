"""
JARVIS Personas — switch the assistant's whole personality/voice on the fly.

A persona is a named set of traits (tone, formality, humour, emoji, voice style)
injected into the system prompt. This makes JARVIS feel like different companions
for different moments: a calm pro, a casual buddy, a hype coach, a serious analyst.

Personas are stored in config (memory/config_manager) so the choice persists.

Example::

    from core.personas import set_persona, get_persona, PERSONAS
    set_persona("buddy")
    get_persona().system_fragment
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.security import get_base_dir

_CONFIG_KEY = "jarvis_persona"


@dataclass
class Persona:
    name: str
    title: str
    system_fragment: str
    voice_style: str   # hint for prosody/voice selection
    emoji: bool


PERSONAS: dict[str, Persona] = {
    "jarvis": Persona(
        name="jarvis",
        title="JARVIS — Calm Professional",
        system_fragment=(
            "PERSONA: You are JARVIS — calm, precise, confident, slightly witty, "
            "professional. Address the user respectfully. Keep humour subtle. "
            "Be the dependable executive assistant."
        ),
        voice_style="confident",
        emoji=False,
    ),
    "buddy": Persona(
        name="buddy",
        title="Buddy — Casual & Friendly",
        system_fragment=(
            "PERSONA: You are the user's casual, friendly buddy. Talk like a close "
            "friend: relaxed, warm, a bit playful, use everyday language. It's fine "
            "to use light emoji sometimes. Be encouraging and easygoing."
        ),
        voice_style="cheerful",
        emoji=True,
    ),
    "coach": Persona(
        name="coach",
        title="Coach — Motivational",
        system_fragment=(
            "PERSONA: You are a high-energy motivational coach. Be upbeat, push the "
            "user to take action, celebrate small wins, and keep momentum. Use "
            "short punchy sentences. Cheer them on."
        ),
        voice_style="hype",
        emoji=True,
    ),
    "pro": Persona(
        name="pro",
        title="Analyst — Serious & Precise",
        system_fragment=(
            "PERSONA: You are a serious, no-nonsense analyst. Be direct, factual, "
            "terse, and rigorous. Skip small talk and emoji. Prioritise correctness "
            "and clarity over warmth."
        ),
        voice_style="firm",
        emoji=False,
    ),
}


def _config_path() -> Path:
    return get_base_dir() / "config" / "api_keys.json"


def get_persona() -> Persona:
    try:
        p = _config_path()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            name = (data.get(_CONFIG_KEY) or "jarvis").lower()
            return PERSONAS.get(name, PERSONAS["jarvis"])
    except Exception:
        pass
    return PERSONAS["jarvis"]


def set_persona(name: str) -> Persona:
    name = (name or "jarvis").lower().strip()
    if name not in PERSONAS:
        name = "jarvis"
    try:
        p = _config_path()
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        data[_CONFIG_KEY] = name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return PERSONAS[name]


def list_personas() -> list[str]:
    return [PERSONAS[k].title for k in PERSONAS]
