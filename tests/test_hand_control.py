"""
Tests for hand control + clap activation modules.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── ClapDetector ───────────────────────────────────────────────────────────────
def _clap_frames(distances, base=0.0, dt=0.03):
    """Build (left, right, timestamp) tuples from a list of palm distances."""
    for i, d in enumerate(distances):
        half = d / 2.0
        left = np.array([0.5 - half, 0.5, 0.0])
        right = np.array([0.5 + half, 0.5, 0.0])
        yield left, right, base + i * dt


def test_clap_detector_basic():
    from clap_detector import ClapDetector

    det = ClapDetector(sensitivity=1.0, cooldown=0.5)

    base = time.time()
    # Distance pattern: flat, steep approach, min in middle of window, steep sep.
    # Min distance (0.10) must fall in [0.08, 0.35] and at window index 2 so
    # there are 2 frames of approach (>0.15 drop) and 2 of separation (>0.10 rise).
    distances = [0.50, 0.50, 0.50, 0.30, 0.10, 0.20, 0.40, 0.50, 0.50]

    result = None
    for left, right, now in _clap_frames(distances, base=base):
        result = det.update(left, right, timestamp=now)
        if result:
            break

    assert result is not None
    assert 0.0 <= result.confidence <= 1.0


def test_clap_detector_cooldown():
    from clap_detector import ClapDetector

    base = time.time()
    det = ClapDetector(sensitivity=1.0, cooldown=2.0)

    distances = [0.50, 0.50, 0.50, 0.30, 0.10, 0.20, 0.40, 0.50, 0.50]

    result = None
    for left, right, now in _clap_frames(distances, base=base):
        r = det.update(left, right, timestamp=now)
        if r:
            result = r
            break

    assert result is not None

    # Immediate re-clap should be blocked by cooldown
    result2 = det.update(
        np.array([0.3, 0.5, 0.0]), np.array([0.7, 0.5, 0.0]),
        timestamp=base + 0.5,
    )
    assert result2 is None


# ── GestureDetector ─────────────────────────────────────────────────────────────
def _make_landmarks(finger_states: list[bool], thumb_extended: bool = True) -> np.ndarray:
    """Create landmarks array where finger_states = [index, middle, ring, pinky].
    Uses correct MediaPipe hand landmark indices:
      0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky.
    """
    lm = np.zeros((21, 3))
    lm[0] = [0.5, 0.5, 0.0]

    # Thumb: MCP=1, PIP=2, DIP=3, TIP=4
    if thumb_extended:
        lm[1] = [0.52, 0.5, 0.0]
        lm[2] = [0.51, 0.5, 0.0]
        lm[3] = [0.50, 0.5, 0.0]
        lm[4] = [0.40, 0.5, 0.0]   # thumb tip to the left (extended for right hand)
    else:
        lm[1] = [0.52, 0.5, 0.0]
        lm[2] = [0.51, 0.5, 0.0]
        lm[3] = [0.50, 0.5, 0.0]
        lm[4] = [0.55, 0.5, 0.0]   # thumb tip close to wrist (not extended)

    finger_base_x = [0.45, 0.55, 0.65, 0.75]
    base_y = 0.5

    for fi, (fx, extended) in enumerate(zip(finger_base_x, finger_states)):
        idx = fi * 4 + 5
        if extended:
            lm[idx]     = [fx, 0.45, 0.0]  # MCP
            lm[idx + 1] = [fx, 0.40, 0.0]  # PIP
            lm[idx + 2] = [fx, 0.35, 0.0]  # DIP
            lm[idx + 3] = [fx, 0.25, 0.0]  # TIP (well above PIP)
        else:
            lm[idx]     = [fx, 0.55, 0.0]
            lm[idx + 1] = [fx, 0.55, 0.0]
            lm[idx + 2] = [fx, 0.55, 0.0]
            lm[idx + 3] = [fx, 0.55, 0.0]

    return lm


def test_gesture_fist():
    from gesture_detector import GestureDetector, GestureType

    det = GestureDetector()
    lm = _make_landmarks([False, False, False, False], thumb_extended=False)
    # Need debounce frames
    result = None
    for _ in range(5):
        result = det.detect(lm, "Right")
    assert result == GestureType.FIST


def test_gesture_open_palm():
    from gesture_detector import GestureDetector, GestureType

    det = GestureDetector()
    lm = _make_landmarks([True, True, True, True], thumb_extended=True)
    result = None
    for _ in range(5):
        result = det.detect(lm, "Right")
    assert result == GestureType.OPEN_PALM


def test_gesture_pinch():
    from gesture_detector import GestureDetector, GestureType

    det = GestureDetector()
    lm = _make_landmarks([False, False, False, False], thumb_extended=False)
    # Move thumb tip close to index tip
    lm[4] = [0.45, 0.55, 0.0]  # thumb tip near index
    lm[8] = [0.46, 0.55, 0.0]  # index tip near thumb
    result = None
    for _ in range(5):
        result = det.detect(lm, "Right")
    assert result == GestureType.PINCH


def test_gesture_point():
    from gesture_detector import GestureDetector, GestureType

    det = GestureDetector()
    lm = _make_landmarks([True, False, False, False], thumb_extended=False)
    result = None
    for _ in range(5):
        result = det.detect(lm, "Right")
    assert result == GestureType.POINT


def test_gesture_two_fingers():
    from gesture_detector import GestureDetector, GestureType

    det = GestureDetector()
    lm = _make_landmarks([True, True, False, False], thumb_extended=False)
    result = None
    for _ in range(5):
        result = det.detect(lm, "Right")
    assert result == GestureType.TWO_FINGERS


# ── config ─────────────────────────────────────────────────────────────────────
def test_config_has_all_keys():
    import config
    assert hasattr(config, 'CAMERA_INDEX')
    assert hasattr(config, 'CAMERA_WIDTH')
    assert hasattr(config, 'CAMERA_HEIGHT')
    assert hasattr(config, 'HAND_DETECTION_CONFIDENCE')
    assert hasattr(config, 'HAND_TRACKING_CONFIDENCE')
    assert hasattr(config, 'CLAP_SENSITIVITY')
    assert hasattr(config, 'CLAP_COOLDOWN')
    assert hasattr(config, 'WAKE_TIMEOUT')
    assert hasattr(config, 'HAND_DEBUG')
    assert hasattr(config, 'WAKE_WORDS')
    assert 'jarvis' in config.WAKE_WORDS
