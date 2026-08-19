"""
Vision 2.0 for Jarvis — continuous screen understanding + OCR.

Wraps the existing ``actions/screen_processor`` capture pipeline and adds:
  * ``ocr_screen()`` — extract text from the current screen (pytesseract/easyocr)
  * ``describe_screen()`` — short natural-language description (via the local LLM
    when available, else a structured fallback)
  * ``watch()`` — periodic capture loop that emits ``vision.screen`` events.

All heavy deps are imported lazily so this module is import-safe headless.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from core.event_bus import bus

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(self) -> None:
        self._ocr = None
        self._stop = threading.Event()

    # ── capture ─────────────────────────────────────────────────────────────────
    def _capture(self):
        try:
            from actions.screen_processor import _capture_screen

            return _capture_screen()
        except Exception as exc:  # noqa: BLE001
            logger.warning("screen capture unavailable: %s", exc)
            return None

    # ── OCR ────────────────────────────────────────────────────────────────────
    def _load_ocr(self):
        if self._ocr is not None:
            return self._ocr
        try:
            import pytesseract  # type: ignore

            self._ocr = ("tesseract", pytesseract)
        except Exception:  # noqa: BLE001
            try:
                import easyocr  # type: ignore

                self._ocr = ("easyocr", easyocr.Reader(["en"]))
            except Exception:  # noqa: BLE001
                self._ocr = None
        return self._ocr

    def ocr_screen(self, image: Any = None) -> str:
        ocr = self._load_ocr()
        img = image or self._capture()
        if img is None or ocr is None:
            return ""
        try:
            if ocr[0] == "tesseract":
                return ocr[1].image_to_string(img).strip()
            return " ".join(ocr[1].readtext(img, detail=0)).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR failed: %s", exc)
            return ""

    # ── describe ─────────────────────────────────────────────────────────────────
    def describe_screen(self, image: Any = None) -> str:
        img = image or self._capture()
        if img is None:
            return "No screen capture available."
        text = self.ocr_screen(img)
        try:
            from core.llm_client import call_llm_text

            prompt = (
                "Describe this screen briefly for a voice assistant user. "
                f"On-screen text: {text[:1500]}"
            )
            return call_llm_text(prompt, system="You are Jarvis' vision. Be concise.").strip()
        except Exception:  # noqa: BLE001 - fall back to raw OCR summary
            return f"Screen shows text: {text[:300] or '(no readable text)'}"

    # ── watch loop ───────────────────────────────────────────────────────────────
    def watch(self, interval_s: float = 5.0) -> None:
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    desc = self.describe_screen()
                    bus.emit("vision.screen", {"description": desc}, source="vision")
                except Exception as exc:  # noqa: BLE001
                    logger.error("vision watch error: %s", exc)
                self._stop.wait(interval_s)

        threading.Thread(target=_loop, daemon=True, name="vision-watch").start()

    def stop_watch(self) -> None:
        self._stop.set()
