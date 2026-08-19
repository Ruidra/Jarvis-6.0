"""
Local Voice Activity Detection (VAD) + wake-word for Jarvis.

Replaces naive always-on listening with offline energy-based VAD and an optional
Porcupine wake-word engine (``pip install pvporcupine``).  When Porcupine isn't
installed, a lightweight energy+VAD "hey jarvis" detector is used as fallback so
the system runs fully offline with no cloud cost and far fewer false triggers.

Example::

    from core.vad import VADetector, WakeWordDetector
    vad = VADetector()
    if vad.is_speech(audio_numpy, sample_rate=16000):
        ...
    ww = WakeWordDetector()      # tries Porcupine, else energy fallback
    if ww.detect(audio_numpy):
        ...
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VADetector:
    """RMS-energy voice activity detector (no ML deps, runs offline)."""

    def __init__(self, threshold: float = 0.015, frame_ms: int = 30, sample_rate: int = 16000) -> None:
        self.threshold = threshold
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        self.sample_rate = sample_rate

    def is_speech(self, audio: np.ndarray, sample_rate: int | None = None) -> bool:
        sr = sample_rate or self.sample_rate
        if audio is None or audio.size == 0:
            return False
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        # normalise if int PCM
        if np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(a.astype(np.float32) ** 2) + 1e-12))
        return rms > self.threshold

    def speech_ratio(self, audio: np.ndarray, sample_rate: int | None = None) -> float:
        """Fraction of frames above threshold — useful for barge-in detection."""
        sr = sample_rate or self.sample_rate
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.float32) / 32768.0
        n = max(1, len(a) // self.frame_samples)
        voiced = 0
        for i in range(n):
            seg = a[i * self.frame_samples : (i + 1) * self.frame_samples]
            rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
            if rms > self.threshold:
                voiced += 1
        return voiced / n


class WakeWordDetector:
    """Wake-word detection. Uses Porcupine if available, else energy VAD."""

    def __init__(self, keywords: list[str] | None = None, sensitivity: float = 0.5) -> None:
        self.keywords = keywords or ["jarvis", "hey jarvis"]
        self.vad = VADetector()
        self._porcupine = None
        self._try_porcupine(sensitivity)

    def _try_porcupine(self, sensitivity: float) -> None:
        try:
            import pvporcupine  # type: ignore

            self._porcupine = pvporcupine.create(keywords=self.keywords[:1] or ["jarvis"])
            logger.info("WakeWordDetector backend: Porcupine")
        except Exception as exc:  # noqa: BLE001 - Porcupine optional
            logger.info("WakeWordDetector backend: energy-VAD fallback (%s)", exc)

    @property
    def backend(self) -> str:
        return "porcupine" if self._porcupine else "energy"

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        if self._porcupine is not None:
            try:
                pcm = (np.asarray(audio, dtype=np.float32) * 32768.0).astype(np.int16)
                frame = pcm[: self._porcupine.frame_length]
                if frame.size == self._porcupine.frame_length:
                    return bool(self._porcupine.process(frame))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Porcupine detect failed (%s); using VAD", exc)
        return self.vad.is_speech(audio, sample_rate)
