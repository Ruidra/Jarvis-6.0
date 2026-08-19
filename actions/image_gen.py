"""
Image generation tool for JARVIS.

Uses the free Pollinations text-to-image endpoint (no API key required).  Returns
the saved image path so the assistant can describe/show it, and a confidence tag
since the output is non-deterministic.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def _out_dir() -> Path:
    try:
        import sys as _sys
        if getattr(_sys, "frozen", False):
            base = Path(_sys.executable).parent
        else:
            base = Path(__file__).resolve().parents[1]
        d = base / "outputs" / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:
        return Path("outputs/images")


def image_generate(parameters: dict, response=None, player=None) -> str:
    prompt = (parameters or {}).get("prompt", "")
    if not prompt:
        return "No prompt provided for image generation."
    width = int((parameters or {}).get("width", 1024))
    height = int((parameters or {}).get("height", 1024))
    width = max(256, min(1792, width))
    height = max(256, min(1792, height))

    safe = re.sub(r"[^a-z0-9]+", "_", prompt.lower())[:40].strip("_") or "image"
    out = _out_dir() / f"{safe}.png"

    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt, safe="")
        + f"?width={width}&height={height}&nologo=true&model=flux&seed="
        + str(hash(prompt) % 1000000)
    )
    try:
        logger.info("image_gen: fetching %s", prompt[:60])
        req = urllib.request.Request(url, headers={"User-Agent": "Jarvis/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        out.write_bytes(data)
        return f"Generated image saved to {out} (prompt: {prompt}). This is AI-generated, so verify it looks right."
    except Exception as e:  # noqa: BLE001
        logger.warning("image_gen failed: %s", e)
        return f"Image generation failed: {e}. (Needs internet access to pollinations.ai)"
