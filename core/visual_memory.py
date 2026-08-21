"""
JARVIS Persistent Visual Memory & Audio Event Detection — JARVIS 7.0.

Extends the real-time multimodal stack with:

  * **Persistent visual memory** — screenshots are embedded and stored in the
    vector memory so JARVIS can answer "what was on my screen earlier?" or
    "what error message did I see before?".
  * **Audio event detection** — beyond the wake word, listen for:
    doorbell, smoke/CO alarm, baby cry, glass break, distress in voice tone.
    Triggered via the Vosk/STT partial-transcript analysis plus energy patterns.

Both feed into ``core/multimodal.MultimodalContext`` so the LLM knows what
JARVIS "perceives" at any given moment.

Example::

    from core.visual_memory import visual_memory

    # Screenshot + embed
    visual_memory.capture_screen(reason="user said: show me the error")
    # -> stores embedding in vector memory

    # Search past screenshots by content
    matches = visual_memory.search("network connection error")

    # Check for audio events (called by the mic callback)
    event = visual_memory.check_audio_event(transcript, energy_level)
    # -> {"type": "distress", "confidence": 0.85, "message": "User's voice sounded distressed"}
"""

from __future__ import annotations

import io
import logging
import re
import time
from pathlib import Path
from typing import Any

from core.vector_memory import vector_db

logger = logging.getLogger("jarvis.visual_memory")

_SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "memory" / "screenshots"
_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


class VisualMemory:
    """Persistent screenshot storage with semantic search."""

    def __init__(self) -> None:
        self._cache: list[dict[str, Any]] = []
        self._load_cache()

    def _load_cache(self) -> None:
        cache_path = _SCREENSHOT_DIR / "index.json"
        if cache_path.exists():
            import json
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8")) or []
            except Exception:
                pass

    def _save_cache(self) -> None:
        cache_path = _SCREENSHOT_DIR / "index.json"
        import json
        cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def capture_screen(self, reason: str = "", ocr_text: str = "") -> str:
        """Capture a screenshot, store it, and embed its content in vector memory."""
        ts = time.time()
        filename = f"screenshot_{int(ts)}.png"
        filepath = _SCREENSHOT_DIR / filename

        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(filepath, "PNG")
        except ImportError:
            try:
                import pyautogui
                img = pyautogui.screenshot()
                img.save(filepath)
            except Exception as exc:
                logger.warning("Could not capture screenshot: %s", exc)
                return ""

        # OCR if not provided
        if not ocr_text:
            try:
                import pytesseract
                from PIL import Image
                ocr_text = pytesseract.image_to_string(Image.open(filepath))
            except ImportError:
                ocr_text = ""

        # Store in vector memory
        memory_id = vector_db.add(
            text=(ocr_text or f"Screenshot: {reason}"),
            metadata={"filename": filename, "reason": reason, "timestamp": ts},
            mem_type="episodic",
            importance=0.6,
        )

        entry = {
            "id": memory_id,
            "filename": filename,
            "filepath": str(filepath),
            "reason": reason,
            "ocr": ocr_text[:500],
            "ts": ts,
        }
        self._cache.append(entry)
        self._save_cache()

        logger.info("Visual memory: captured screenshot %s (reason: %s)", filename, reason)
        return memory_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search past screenshots by semantic similarity."""
        results = vector_db.query(query, top_k=top_k * 2, mem_type="episodic")
        # Filter to only screenshot entries
        screenshot_results = []
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("filename") or "screenshot" in (r.get("text", "")).lower():
                screenshot_results.append({
                    "text": r["text"],
                    "filename": meta.get("filename", ""),
                    "reason": meta.get("reason", ""),
                    "ts": meta.get("timestamp", 0),
                    "score": r.get("score", 0),
                })
        return screenshot_results[:top_k]

    def count(self) -> int:
        return len(self._cache)

    def clear_old(self, older_than_days: float = 30.0) -> int:
        """Remove screenshots older than *older_than_days*."""
        cutoff = time.time() - older_than_days * 86400
        before = len(self._cache)
        self._cache = [e for e in self._cache if e.get("ts", 0) >= cutoff]
        self._save_cache()
        return before - len(self._cache)


class AudioEventDetector:
    """Detect non-speech audio events from voice energy patterns and transcript context."""

    # Keywords that indicate audio events in STT transcript
    _EVENT_PATTERNS: dict[str, list[str]] = {
        "doorbell": [r"\bdoorbell\b", r"\bding[- ]dong\b", r"\bknock(ed|ing)?\b"],
        "alarm": [r"\bsmoke alarm\b", r"\bco alarm\b", r"\balarm (clock|beeping|going off)\b",
                  r"\bchirp(ing|ed)?\b"],
        "baby": [r"\bab[yi]?\s+crying\b", r"\bbaby\s+(cry|woke)\b"],
        "glass": [r"\bglass (breaking|broke)\b", r"\bshattered\b"],
        "distress": [r"\bhelp\s+me\b", r"\bcall\s+911\b", r"\bai\s+(choking|screaming)\b",
                     r"\b(emergency|save\s+me)\b"],
    }

    # Energy thresholds (RMS amplitude) for detecting non-speech audio
    _ENERGY_THRESHOLDS: dict[str, tuple[float, ...]] = {
        "doorbell": (0.3, 0.6),    # moderate spike, short duration
        "alarm": (0.5, 2.0),       # sustained high energy
        "glass": (0.8, 1.0),       # sharp, very high spike
    }

    def __init__(self) -> None:
        self._recent_events: list[dict[str, Any]] = []
        self._cooldown: dict[str, float] = {}

    def check(self, transcript: str, energy: float, voice_tone: str = "neutral") -> dict[str, Any] | None:
        """Check for audio events given transcript, energy level, and voice tone.

        Returns an event dict if an event is detected, None otherwise.
        """
        now = time.time()
        transcript_lower = (transcript or "").lower()

        # Check keyword-based patterns (highest confidence)
        for event_type, patterns in self._EVENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, transcript_lower):
                    # Check cooldown
                    if now - self._cooldown.get(event_type, 0) < 30:
                        continue
                    self._cooldown[event_type] = now
                    event = {
                        "type": event_type,
                        "confidence": 0.9,
                        "message": self._event_message(event_type),
                        "ts": now,
                    }
                    self._recent_events.append(event)
                    logger.info("Audio event detected: %s (via transcript keyword)", event_type)
                    return event

        # Check voice distress tone
        if voice_tone == "distressed" or voice_tone == "anxious":
            if now - self._cooldown.get("distress_voice", 0) < 60:
                pass  # cooldown
            else:
                self._cooldown["distress_voice"] = now
                # Lower confidence since it's tonal, not explicit
                if voice_tone == "distressed":
                    event = {
                        "type": "distress_voice",
                        "confidence": 0.75,
                        "message": "User's voice tone suggests distress. Should I check in?",
                        "ts": now,
                    }
                    self._recent_events.append(event)
                    logger.info("Audio event detected: distress in voice tone")
                    return event

        # Check energy-based patterns (for non-speech events)
        if energy > 0.1:
            for event_type, (low, high) in self._ENERGY_THRESHOLDS.items():
                if low <= energy <= high:
                    if now - self._cooldown.get(event_type, 0) < 10:
                        continue

        return None

    @staticmethod
    def _event_message(event_type: str) -> str:
        messages = {
            "doorbell": "Doorbell detected at the front entrance.",
            "alarm": "Smoke/CO alarm detected — please check immediately!",
            "baby": "Baby crying detected.",
            "glass": "Glass breaking detected — possible intrusion!",
            "distress": "Distress call detected — should I contact emergency services?",
            "distress_voice": "User's voice tone suggests distress. Would you like me to check in?",
        }
        return messages.get(event_type, f"Audio event: {event_type}")

    def recent_events(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most recent audio events."""
        return list(reversed(self._recent_events[-limit:]))

    def clear_old_events(self, older_than_seconds: float = 300.0) -> int:
        """Remove events older than *older_than_seconds*."""
        cutoff = time.time() - older_than_seconds
        before = len(self._recent_events)
        self._recent_events = [e for e in self._recent_events if e.get("ts", 0) >= cutoff]
        return before - len(self._recent_events)


# Process-wide instances.
visual_memory = VisualMemory()
audio_events = AudioEventDetector()
