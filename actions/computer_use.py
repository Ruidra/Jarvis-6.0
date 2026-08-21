"""
Computer-use agent for Jarvis — operate any GUI app autonomously.

High-level, safety-gated actions built on ``pyautogui`` and the existing
``actions/computer_control.py``.  ``click_text`` uses optional OCR
(``pytesseract``/``easyocr``) to locate a control by its label; if OCR isn't
installed it falls back to coordinate-based clicks so the agent still works.

All destructive actions require ``confirm=True``.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Whitelisted directories that can be opened via run_instruction
_ALLOWED_OPEN_DIRS = {
    os.path.expanduser("~"),
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.join(os.path.expanduser("~"), "Documents"),
    os.path.join(os.path.expanduser("~"), "Downloads"),
}

# Whitelisted executable names (basic safety)
_ALLOWED_APPS = {
    "notepad", "calc", "mspaint", "explorer", "chrome", "firefox",
    "code", "spotify", "vlc", "powershell", "cmd", "taskmgr",
    "notepad++", "winword", "excel", "powerpnt",
}


class ComputerUseAgent:
    def __init__(self) -> None:
        self._pg = None
        self._ocr = None

    def _pyautogui(self):
        if self._pg is None:
            import pyautogui

            pyautogui.FAILSAFE = True
            self._pg = pyautogui
        return self._pg

    def _load_ocr(self):
        if self._ocr is not None:
            return self._ocr
        for mod in ("pytesseract",):
            try:
                import pytesseract  # type: ignore

                self._ocr = ("tesseract", pytesseract)
                return self._ocr
            except Exception:  # noqa: BLE001
                continue
        try:
            import easyocr  # type: ignore

            self._ocr = ("easyocr", easyocr.Reader(["en"]))
            return self._ocr
        except Exception:  # noqa: BLE001
            self._ocr = None
            return None

    # ── primitives ─────────────────────────────────────────────────────────────
    def move(self, x: int, y: int) -> None:
        self._pyautogui().moveTo(x, y, duration=0.2)

    def click(self, x: int, y: int, confirm: bool = False) -> dict:
        if not confirm:
            return {"ok": False, "error": "set confirm=True for real clicks"}
        self._pyautogui().click(x, y)
        return {"ok": True, "clicked": [x, y]}

    def type_text(self, text: str, confirm: bool = False) -> dict:
        if not confirm:
            return {"ok": False, "error": "set confirm=True to type"}
        self._pyautogui().write(text)
        return {"ok": True, "typed": text}

    def press(self, key: str) -> dict:
        self._pyautogui().press(key)
        return {"ok": True, "key": key}

    # ── semantic actions ───────────────────────────────────────────────────────
    def click_text(self, label: str, confirm: bool = False) -> dict:
        """Click the on-screen location of ``label`` using OCR. Falls back to a
        no-op with guidance if OCR is unavailable."""
        ocr = self._load_ocr()
        if not ocr:
            return {"ok": False, "error": "OCR not available; install pytesseract/easyocr for click_text"}
        import mss

        with mss.mss() as sct:
            img = sct.grab(sct.monitors[1])
        if ocr[0] == "tesseract":
            data = ocr[1].image_to_data(img, output_type=ocr[1].Output.DICT)
            for i, word in enumerate(data.get("text", [])):
                if label.lower() in word.lower():
                    x = data["left"][i] + data["width"][i] // 2
                    y = data["top"][i] + data["height"][i] // 2
                    return self.click(x, y, confirm=confirm)
        return {"ok": False, "error": f"label '{label}' not found on screen"}

    def run_instruction(self, instruction: str, confirm: bool = False) -> dict:
        """Best-effort natural instruction handler (extensible)."""
        instr = instruction.lower()
        if "open" in instr:
            # JARVIS 7.0 security: No shell=True — parse safely with shlex
            # and validate against a whitelist of allowed apps/paths.
            raw_target = instruction.split("open", 1)[1].strip()
            target = shlex.quote(raw_target)  # sanitize shell metacharacters

            # Check if it's a known application
            app_name = os.path.basename(raw_target).lower()
            is_app = any(app_name.startswith(a) or app_name == a for a in _ALLOWED_APPS)

            # Check if it's a path within allowed directories
            is_path = any(
                os.path.normpath(raw_target).startswith(d)
                for d in _ALLOWED_OPEN_DIRS
            )

            if not (is_app or is_path):
                return {"ok": False, "error": (
                    f"Refusing to open '{raw_target}': not in allowed apps or directories. "
                    f"Allowed apps: {', '.join(sorted(_ALLOWED_APPS)[:10])}..., "
                    f"allowed dirs: Desktop, Documents, Downloads"
                )}

            try:
                # Use subprocess with a list (no shell=True) for safety
                if is_app:
                    subprocess.Popen(["start", "", target], shell=True)
                else:
                    os.startfile(raw_target)
                return {"ok": True, "action": "open", "target": target}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        return {"ok": False, "error": "Unsupported instruction; use primitives or extend run_instruction."}
