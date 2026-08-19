"""
Hand gesture detection from MediaPipe landmarks.
"""

import time
import logging
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GestureType(Enum):
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    POINT = "POINT"
    TWO_FINGERS = "TWO_FINGERS"
    PINCH = "PINCH"
    CLAP = "CLAP"
    WAVE = "WAVE"
    NONE = "NONE"


class GestureDetector:
    TIPS = [4, 8, 12, 16, 20]
    PIPS = [3, 6, 10, 14, 18]

    def __init__(self, debounce: int = 3):
        self.debounce = debounce
        self._hist = []
        self._wave_x = []
        self._last_wave = 0.0
        self._wave_cd = 1.0

    def detect(self, landmarks: np.ndarray, handedness: str,
               now: Optional[float] = None) -> GestureType:
        if landmarks is None or len(landmarks) < 21:
            return GestureType.NONE
        now = now or time.time()
        g = self._classify(landmarks, handedness)
        if g == GestureType.OPEN_PALM and self._is_wave(landmarks, now):
            g = GestureType.WAVE
        self._hist.append(g)
        if len(self._hist) > self.debounce:
            self._hist.pop(0)
        if len(self._hist) == self.debounce:
            maj = max(set(self._hist), key=self._hist.count)
            if maj != GestureType.NONE:
                return maj
        return GestureType.NONE

    def _classify(self, lm: np.ndarray, hand: str) -> GestureType:
        fingers = self._fingers(lm, hand)
        pinch = float(np.linalg.norm(lm[4] - lm[8]))
        if pinch < 0.05:
            return GestureType.PINCH
        extended = sum(fingers)
        if extended >= 4:
            return GestureType.OPEN_PALM
        if extended == 0:
            return GestureType.FIST
        if extended == 1 and fingers[1]:
            return GestureType.POINT
        if extended == 2 and fingers[1] and fingers[2]:
            return GestureType.TWO_FINGERS
        return GestureType.NONE

    def _fingers(self, lm: np.ndarray, hand: str) -> list[bool]:
        out = []
        thumb_tip, thumb_ip = lm[4], lm[3]
        if hand == "Right":
            out.append(thumb_tip[0] < thumb_ip[0] - 0.02)
        else:
            out.append(thumb_tip[0] > thumb_ip[0] + 0.02)
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            out.append(lm[tip][1] < lm[pip][1] - 0.01)
        return out

    def _is_wave(self, lm: np.ndarray, now: float) -> bool:
        if now - self._last_wave < self._wave_cd:
            return False
        self._wave_x.append(lm[0][0])
        if len(self._wave_x) > 20:
            self._wave_x.pop(0)
        if len(self._wave_x) < 10:
            return False
        xs = self._wave_x[-10:]
        changes = sum(1 for i in range(1, len(xs)) if (xs[i] - xs[i-1]) * (xs[i-1] - xs[i-2]) < 0)
        if changes >= 3:
            self._last_wave = now
            self._wave_x.clear()
            logger.info("[GESTURE] WAVE")
            return True
        return False
