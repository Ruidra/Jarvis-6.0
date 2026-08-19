"""
JARVIS Prosody Speaker — an emotion-tuned *local* voice.

Gemini Live already speaks with its own natural voice, so the live conversation
stays fast and native. But for JARVIS's own short notifications (errors,
unlocks, motivation, check-ins) we can give him a *real emotional voice* on
demand: this wraps an offline/online TTS engine and applies the prosody hints
(rate / pitch / style) produced by the emotion engine.

Fallback chain (graceful, never blocks the assistant):
    Kokoro (offline, if installed)  →  Edge TTS (free, online)  →  no-op

Example::

    from core.prosody_speaker import ProsodySpeaker
    spk = ProsodySpeaker()
    spk.speak("Hang in there, Boss — we've got this.", prosody={"rate": 0.95, "pitch": 0.9})
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProsodySpeaker:
    """Speak short phrases locally with optional emotional prosody."""

    def __init__(self) -> None:
        self._engine = None
        self._kind = "none"
        self._lock = threading.Lock()
        self._try_init()

    def _try_init(self) -> None:
        # Kokoro is fully offline and fast (producer/consumer + warmup in tts.py)
        try:
            from core.tts import create_tts_player  # noqa: F401
            self._engine = "kokoro"
            self._kind = "kokoro"
            logger.info("ProsodySpeaker: Kokoro TTS available")
            return
        except Exception:
            pass
        # Edge TTS is free and online (no API key) and supports rate/pitch styles.
        try:
            import edge_tts  # type: ignore  # noqa: F401
            self._engine = "edge"
            self._kind = "edge"
            logger.info("ProsodySpeaker: Edge TTS available")
            return
        except Exception:
            logger.info("ProsodySpeaker: no local TTS engine available (will be a no-op)")

    @property
    def available(self) -> bool:
        return self._kind != "none"

    def speak(self, text: str, prosody: dict | None = None) -> bool:
        """Speak ``text`` with ``prosody`` (e.g. {"rate":0.9,"pitch":1.1}).

        Runs in a background thread so it never blocks the caller. Returns
        True if a voice engine handled it, False if no engine was available.
        """
        if not text or not self.available:
            return False
        prosody = prosody or {}
        t = threading.Thread(target=self._speak_sync, args=(text, prosody),
                             daemon=True, name="prosody-speak")
        t.start()
        return True

    def _speak_sync(self, text: str, prosody: dict) -> None:
        try:
            if self._kind == "kokoro":
                self._speak_kokoro(text, prosody)
            elif self._kind == "edge":
                self._speak_edge(text, prosody)
        except Exception as exc:  # noqa: BLE001 - voice is non-critical
            logger.warning("ProsodySpeaker failed: %s", exc)

    # -- Kokoro ------------------------------------------------------------ #
    def _speak_kokoro(self, text: str, prosody: dict) -> None:
        # Kokoro's TTSPlayer accepts voice + speed; map pitch→voice choice loosely.
        from core.tts import create_tts_player
        player = create_tts_player({"engine": "kokoro"})
        speed = float(prosody.get("rate", 1.0)) or 1.0
        player.speak(text, speed=speed)
        player.stop()

    # -- Edge TTS ---------------------------------------------------------- #
    def _speak_edge(self, text: str, prosody: dict) -> None:
        import asyncio
        import edge_tts  # type: ignore

        rate = prosody.get("rate", 1.0)
        pitch = prosody.get("pitch", 1.0)
        # Edge expects strings like "+10%"/"-5%"
        rpct = f"{int(round((rate - 1.0) * 100)):+d}%"
        ppct = f"{int(round((pitch - 1.0) * 100)):+d}%"
        voice = "en-US-GuyNeural" if prosody.get("style") in ("firm", "confident") \
            else "en-US-AriaNeural"
        comm = edge_tts.Communicate(text, voice, rate=rpct, pitch=ppct)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(comm.stream_to_file(_temp_wav()))
            self._play_wav(_temp_wav())
        finally:
            loop.close()

    def _play_wav(self, path: str) -> None:
        try:
            import sounddevice as sd  # type: ignore
            import soundfile as sf     # type: ignore
            data, sr = sf.read(path, dtype="float32")
            sd.play(data, sr)
            sd.wait()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ProsodySpeaker wav playback failed: %s", exc)


def _temp_wav() -> str:
    import tempfile
    return str(Path(tempfile.gettempdir()) / "jarvis_prosody.wav")

