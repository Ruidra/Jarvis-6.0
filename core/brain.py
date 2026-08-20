"""
core/brain.py — JARVIS unified reasoning layer ("the brain").

Every internal component that needs to *think* (agents, the orchestrator, tool
planners, self-critique) goes through here instead of talking to one LLM
directly. That gives the whole system three things it did not have before:

1. **It never dies.** Backends are tried in order and the first working one
   wins: local LLM (Ollama / LM Studio) → Gemini → a graceful error string.
   Previously an agent would hard-fail if Ollama was not installed.
2. **It remembers.** Identical prompts inside the TTL window are served from a
   local cache, so repeated planning/classification is instant and free.
3. **It is honest about its budget.** Calls are timed, counted and exposed via
   :func:`stats`, and long prompts are trimmed instead of blowing up.

Usage::

    from core.brain import think, think_json

    text = think("Summarise this error", system="You are a debugger")
    data = think_json("Return {\\"agent\\": \\"web\\"} for this task: ...")
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("jarvis.brain")

_MAX_PROMPT_CHARS = 24000
_CACHE_TTL = 900.0  # 15 min

_lock = threading.Lock()
_cache: dict[str, tuple[float, str]] = {}
_stats = {
    "calls": 0,
    "cache_hits": 0,
    "local_ok": 0,
    "gemini_ok": 0,
    "failures": 0,
    "total_seconds": 0.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# configuration
# ──────────────────────────────────────────────────────────────────────────────
def preferred_backend() -> str:
    """'local' (default) or 'gemini' — set ``"brain_backend"`` in api_keys.json."""
    try:
        from config import get_config

        raw = str(get_config().get("brain_backend", "local")).strip().lower()
        return "gemini" if raw in ("gemini", "google", "cloud") else "local"
    except Exception:
        return "local"


def _cache_key(prompt: str, system: str | None, temperature: float) -> str:
    import hashlib

    blob = f"{system or ''}\x00{prompt}\x00{temperature:.2f}"
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


def _trim(text: str, limit: int = _MAX_PROMPT_CHARS) -> str:
    if not text or len(text) <= limit:
        return text or ""
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.25) :]
    return f"{head}\n\n…[{len(text) - len(head) - len(tail)} chars trimmed]…\n\n{tail}"


# ──────────────────────────────────────────────────────────────────────────────
# backends
# ──────────────────────────────────────────────────────────────────────────────
def _call_local(prompt: str, system: str | None, timeout: int) -> str:
    from core.llm_client import call_llm_text

    return call_llm_text(prompt, system=system, timeout=timeout)


def _call_gemini(prompt: str, system: str | None, timeout: int) -> str:
    from core.gemini_text import generate

    return generate(prompt, system=system, timeout=timeout)


# ──────────────────────────────────────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────────────────────────────────────
def think(
    prompt: str,
    system: str | None = None,
    timeout: int = 120,
    temperature: float = 0.2,
    use_cache: bool = True,
) -> str:
    """Generate text with automatic backend fallback. Never raises."""
    prompt = _trim(prompt)
    key = _cache_key(prompt, system, temperature)

    if use_cache:
        with _lock:
            hit = _cache.get(key)
            if hit and (time.time() - hit[0]) < _CACHE_TTL:
                _stats["cache_hits"] += 1
                return hit[1]

    order = ["local", "gemini"] if preferred_backend() == "local" else ["gemini", "local"]
    started = time.time()
    errors: list[str] = []

    for backend in order:
        try:
            text = (_call_local if backend == "local" else _call_gemini)(prompt, system, timeout)
            text = (text or "").strip()
            if not text:
                errors.append(f"{backend}: empty response")
                continue
            with _lock:
                _stats["calls"] += 1
                _stats[f"{backend}_ok"] = _stats.get(f"{backend}_ok", 0) + 1
                _stats["total_seconds"] += time.time() - started
                if use_cache:
                    _cache[key] = (time.time(), text)
                    if len(_cache) > 256:
                        oldest = sorted(_cache.items(), key=lambda kv: kv[1][0])[:64]
                        for k, _ in oldest:
                            _cache.pop(k, None)
            return text
        except Exception as exc:  # noqa: BLE001 — try the next backend
            errors.append(f"{backend}: {exc}")
            logger.info("brain backend %s failed: %s", backend, exc)

    with _lock:
        _stats["calls"] += 1
        _stats["failures"] += 1
        _stats["total_seconds"] += time.time() - started
    detail = " | ".join(errors) or "no backend available"
    logger.error("brain: all backends failed (%s)", detail)
    return (
        "I could not reach a reasoning model. "
        "Check your Gemini API key in config/api_keys.json, or start a local LLM "
        f"(Ollama: 'ollama serve'). Details: {detail}"
    )


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BARE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def think_json(
    prompt: str,
    system: str | None = None,
    timeout: int = 120,
    default: Any = None,
    retries: int = 1,
) -> Any:
    """Ask for JSON and parse it robustly (fenced blocks, prose, trailing text)."""
    sys_msg = (system or "") + (
        "\n\nReply with VALID JSON ONLY. No prose, no markdown fences, no comments."
    )
    for attempt in range(max(1, retries + 1)):
        raw = think(prompt, system=sys_msg, timeout=timeout, use_cache=(attempt == 0))
        for candidate in _json_candidates(raw):
            try:
                return json.loads(candidate)
            except Exception:
                continue
        prompt = (
            f"{prompt}\n\nYour previous reply was not valid JSON:\n{raw[:400]}\n"
            "Return ONLY the JSON object."
        )
    return default


def _json_candidates(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return
    m = _JSON_BLOCK.search(raw)
    if m:
        yield m.group(1)
    if raw[:1] in "{[":
        yield raw
    m = _JSON_BARE.search(raw)
    if m:
        yield m.group(1)


def available_backends() -> dict[str, bool]:
    """Which reasoning backends can actually be reached right now."""
    out = {"local": False, "gemini": False}
    try:
        from core.llm_client import get_llm_settings, get_llm_provider
        import requests

        url, _ = get_llm_settings()
        health = f"{url}/api/tags" if get_llm_provider() == "ollama" else f"{url}/v1/models"
        out["local"] = requests.get(health, timeout=2).status_code == 200
    except Exception:
        out["local"] = False
    try:
        from config import get_secret

        out["gemini"] = bool(get_secret("gemini_api_key"))
    except Exception:
        out["gemini"] = False
    return out


def stats() -> dict:
    with _lock:
        data = dict(_stats)
    calls = max(1, data["calls"])
    data["avg_seconds"] = round(data["total_seconds"] / calls, 2)
    data["cached_entries"] = len(_cache)
    data["preferred"] = preferred_backend()
    return data


def clear_cache() -> None:
    with _lock:
        _cache.clear()
