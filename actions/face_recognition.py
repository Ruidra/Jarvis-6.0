"""
Face recognition for JARVIS — know who the boss is.

Uses the camera (via ``actions.screen_processor._capture_camera``) and the
existing Gemini multimodal API to verify whether the person in front of the
webcam is the enrolled boss.  No heavy ``face_recognition``/dlib dependency is
required — the reference photo is compared against the live frame by Gemini
Vision, which is already paid for by the assistant's API key.

Tools exposed:
  * ``enroll_boss()``  — capture the current camera frame and store it as the
                         boss reference photo (config/boss/reference.jpg).
  * ``identify()``     — capture a live frame, compare to the reference, and
                         return a结构化 instruction the assistant speaks:
                           - boss    -> greet naturally as "Boss"
                           - unknown -> challenge: "Who are you? Where is my
                                        boss? And what can I do for you?"
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False

from google import genai

# Reuse the existing camera + config plumbing.
from actions.screen_processor import (  # noqa: F401
    _capture_camera,
    _get_api_key,
    _base_dir,
)

_BOSS_DIR = _base_dir() / "config" / "boss"
_BOSS_REF = _BOSS_DIR / "reference.jpg"

# Candidate multimodal models, in priority order. The first that successfully
# answers is used. This guards against a 404 when the active API key lacks
# access to one specific model (e.g. gemini-2.5-flash) — we transparently fall
# back to the next one instead of failing the whole recognition.
_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def _ensure_dir() -> None:
    _BOSS_DIR.mkdir(parents=True, exist_ok=True)


def _boss_name() -> str:
    try:
        from memory.config_manager import get_user_name

        name = (get_user_name() or "").strip()
        if name:
            return name
    except Exception:
        pass
    return "sir"


def _encode(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _parse_verification(text: str) -> tuple[bool, int]:
    """Extract {match, confidence} from the model's JSON-ish reply."""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        match = bool(data.get("match", False))
        conf = int(data.get("confidence", 0) or 0)
        return match, max(0, min(100, conf))
    except Exception:
        # Fallback: keyword scan if JSON parsing failed.
        low = text.lower()
        if '"match": true' in low or "match: true" in low or "same person" in low:
            return True, 70
        if '"match": false' in low or "match: false" in low or "different person" in low:
            return False, 70
        return False, 0


def enroll_boss(player=None) -> str:
    """Capture the current camera frame and store it as the boss reference."""
    if not _CV2:
        return ("[FACE:NoCV2] OpenCV is required for face recognition. "
                "Run: pip install opencv-python")
    try:
        img_bytes, _ = _capture_camera()
    except Exception as e:
        return f"[FACE:Error] Camera capture failed: {e}"

    _ensure_dir()
    try:
        if _PIL:
            from io import BytesIO

            img = PIL.Image.open(BytesIO(img_bytes)).convert("RGB")
            img.save(_BOSS_REF, format="JPEG", quality=90)
        else:
            _BOSS_REF.write_bytes(img_bytes)
    except Exception as e:
        return f"[FACE:Error] Could not save boss reference: {e}"

    if player and hasattr(player, "write_log"):
        player.write_log("[Face] Boss reference enrolled.")
    return ("[FACE:Enrolled] Boss reference saved. I now know who you are. "
            "From now on, when I see you I will say 'Boss'.")


def identify(player=None) -> str:
    """Compare the live camera frame to the boss reference and report who it is."""
    if not _CV2:
        return "[FACE:NoCV2] OpenCV is required. Run: pip install opencv-python"
    if not _BOSS_REF.exists():
        return ("[FACE:NoReference] I don't have a reference photo of my boss yet. "
                "Say 'enroll my face' or 'remember my face' so I can recognize you.")

    try:
        live_bytes, mime = _capture_camera()
    except Exception as e:
        return f"[FACE:Error] Camera failed: {e}"
    try:
        ref_bytes = _BOSS_REF.read_bytes()
    except Exception as e:
        return f"[FACE:Error] Could not read reference: {e}"

    prompt = (
        "You are JARVIS's face-identity verifier. Two images are provided:\n"
        "IMAGE 1 = the enrolled reference photo of JARVIS's BOSS.\n"
        "IMAGE 2 = a live webcam frame captured just now.\n"
        "Decide whether the SAME person appears in both. Weigh face, hair, "
        "build, and distinctive features. Do NOT be fooled by similar clothing.\n"
        "Respond with ONLY a JSON object: "
        '{"match": true|false, "confidence": 0-100, "reason": "short"}.'
    )

    client = genai.Client(
        api_key=_get_api_key(), http_options={"api_version": "v1beta"}
    )
    last_err: Exception | None = None
    text = ""
    for _model in _MODELS:
        try:
            resp = client.models.generate_content(
                model=_model,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": _encode(ref_bytes)}},
                            {"inline_data": {"mime_type": mime or "image/jpeg", "data": _encode(live_bytes)}},
                            {"text": prompt},
                        ],
                    }
                ],
            )
            text = (resp.text or "").strip()
            if text:
                break
            last_err = RuntimeError(f"{_model}: empty response")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    else:
        return f"[FACE:Error] Vision check failed: {last_err}"

    match, conf = _parse_verification(text)
    name = _boss_name()

    if match:
        if player and hasattr(player, "write_log"):
            player.write_log(f"[Face] Boss identified ({conf}%).")
        return (f"[FACE:Boss] This is my boss {name}. Confidence {conf}%. "
                f"Greet him naturally as 'Boss' and ask how you can help.")
    if player and hasattr(player, "write_log"):
        player.write_log(f"[Face] Unknown person ({conf}%).")
    return ("[FACE:Unknown] An unknown person is at the camera. "
            "Say exactly: 'Who are you? Where is my boss? And what can I do for you?'")


def face_recognize(parameters: Optional[dict] = None, player=None, **_kwargs) -> str:
    """Entry point used by the tool dispatcher.

    ``parameters['action']``:
      'identify' (default) — verify who is in front of the camera
      'enroll'             — save the current frame as the boss reference
      'status'             — report whether a reference exists
    """
    params = parameters or {}
    action = (params.get("action") or "identify").lower().strip()

    if action in ("enroll", "register", "remember"):
        return enroll_boss(player=player)
    if action == "status":
        if _BOSS_REF.exists():
            return f"[FACE:Status] Boss reference is enrolled ({_BOSS_REF})."
        return "[FACE:Status] No boss reference enrolled yet."
    return identify(player=player)
