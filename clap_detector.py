"""
Temporal clap detection using two-hand palm tracking.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HandPalmState:
    hand: str
    center: np.ndarray
    wrist: np.ndarray
    timestamp: float


@dataclass
class ClapEvent:
    confidence: float
    timestamp: float
    left_palm: np.ndarray
    right_palm: np.ndarray


class ClapDetector:
    def __init__(self, sensitivity: float = 1.0, cooldown: float = 2.0):
        self.sensitivity = sensitivity
        self.cooldown = cooldown
        self._last_clap_time = 0.0
        self._left: list[HandPalmState] = []
        self._right: list[HandPalmState] = []
        self._approach_threshold = 0.15 * sensitivity
        self._separation_threshold = 0.10 * sensitivity
        self._min_dist = 0.08
        self._max_dist = 0.35

    def update(self, left_palm: Optional[np.ndarray], right_palm: Optional[np.ndarray],
               left_wrist: Optional[np.ndarray] = None, right_wrist: Optional[np.ndarray] = None,
               timestamp: Optional[float] = None) -> Optional[ClapEvent]:
        now = timestamp or time.time()
        if now - self._last_clap_time < self.cooldown:
            self._trim(now)
            return None

        if left_palm is not None:
            self._left.append(HandPalmState("Left", left_palm, left_wrist if left_wrist is not None else left_palm, now))
        if right_palm is not None:
            self._right.append(HandPalmState("Right", right_palm, right_wrist if right_wrist is not None else right_palm, now))

        self._trim(now)

        if len(self._left) < 3 or len(self._right) < 3:
            return None

        return self._detect(now)

    def _trim(self, now: float):
        cutoff = now - 1.0
        self._left = [h for h in self._left if h.timestamp > cutoff]
        self._right = [h for h in self._right if h.timestamp > cutoff]

    def _detect(self, now: float) -> Optional[ClapEvent]:
        left = self._left[-5:]
        right = self._right[-5:]
        n = min(len(left), len(right))
        if n < 3:
            return None

        dists = []
        for i in range(n):
            d = float(np.linalg.norm(left[i].center - right[i].center))
            dists.append((left[i].timestamp, d))

        min_dist = min(d for _, d in dists)
        min_idx = next(i for i, (_, d) in enumerate(dists) if d == min_dist)

        if not (self._min_dist <= min_dist <= self._max_dist):
            return None

        approach = False
        if min_idx >= 2:
            pre = [d for _, d in dists[max(0, min_idx - 4):min_idx]]
            if len(pre) >= 2:
                avg_pre = sum(pre[:-1]) / len(pre[:-1])
                approach = (avg_pre - pre[-1]) > self._approach_threshold

        separation = False
        if min_idx < len(dists) - 2:
            post = [d for _, d in dists[min_idx + 1:min(len(dists), min_idx + 4)]]
            if len(post) >= 2:
                avg_post = sum(post[1:]) / len(post[1:])
                separation = (post[0] - avg_post) < -self._separation_threshold

        if approach and separation:
            self._last_clap_time = now
            conf = min(1.0, (min_dist / self._min_dist) * 0.5 + 0.5)
            logger.info("[CLAP] Detected confidence=%.2f dist=%.3f", conf, min_dist)
            return ClapEvent(
                confidence=conf, timestamp=now,
                left_palm=left[min_idx].center.copy(),
                right_palm=right[min_idx].center.copy(),
            )
        return None
