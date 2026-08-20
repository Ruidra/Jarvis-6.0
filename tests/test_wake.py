"""
Tests for the audio wake engine (clap + wake phrase) that replaced the old
camera/hand-gesture control in JARVIS 6.x.

These cover the core "do no harm" guarantees:
  * two claps within the window arm JARVIS
  * a single clap, well-spaced claps, speech and typing do NOT trigger
  * with WAKE_REQUIRE_CLAP, the phrase alone never wakes JARVIS
  * after a clap arm, a phrase confirms the wake
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from core.wake import ClapEvent, WakeResult

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 16000
_FRAME = 160  # 10 ms @ 16 kHz


# ── synthetic audio helpers ───────────────────────────────────────────────────
def _clap(duration_ms: float = 18.0, peak: float = 0.7, gap_ms: float = 45.0, sr: int = SR):
    """One sharp, bright, fast-decaying noise burst surrounded by silence."""
    gap = int(gap_ms / 1000 * sr)
    burst = max(1, int(duration_ms / 1000 * sr))
    t = np.arange(burst) / sr
    env = np.exp(-t * (3.0 / (duration_ms / 1000)))
    noise = np.random.default_rng(0).uniform(-1, 1, burst)
    body = (noise * env * peak).astype(np.float32)
    return np.concatenate([np.zeros(gap, np.float32), body, np.zeros(gap, np.float32)])


def _stream(detector, audio: np.ndarray, start: float, frame: int = _FRAME):
    """Feed audio in 10 ms frames with correct, advancing timestamps.

    Returns the last *significant* result: a ClapEvent (ClapDetector) or the last
    WakeResult that carried an armed/awake signal (WakeEngine always returns a
    non-None WakeResult, so we must not let a later empty result clobber it).
    """
    out = None
    t = start
    for i in range(0, audio.size, frame):
        chunk = audio[i : i + frame]
        if chunk.size == 0:
            continue
        res = detector.feed(chunk, timestamp=t)
        if isinstance(res, ClapEvent):
            out = res
        elif isinstance(res, WakeResult) and (res.armed or res.awake or res.disarmed):
            out = res
        t += frame / SR
    return out


def _two_claps(clap_gap_s: float = 0.4, sr: int = SR) -> np.ndarray:
    c = _clap(sr=sr)
    gap = np.zeros(int(clap_gap_s * sr), np.float32)
    return np.concatenate([c, gap, c])


def _speech(sr: int = SR, seconds: float = 1.5) -> np.ndarray:
    """Amplitude-modulated tone — sustained, slow attack: NOT a clap."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    carrier = np.sin(2 * np.pi * 220 * t)
    am = 0.25 + 0.25 * np.sin(2 * np.pi * 4 * t)  # syllable-like envelope
    ramp = np.clip(t / 0.15, 0.0, 1.0)            # gradual 150 ms attack
    return (carrier * am * ramp * 0.5).astype(np.float32)


def _typing(sr: int = SR, seconds: float = 1.2) -> np.ndarray:
    """Short, quiet, low-energy clicks — too soft to be a clap."""
    n = int(seconds * sr)
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng(7)
    for _ in range(40):
        pos = rng.integers(0, max(1, n - 40))
        out[pos : pos + 24] = rng.uniform(-0.02, 0.02, 24)
    return out


# ── ClapDetector ──────────────────────────────────────────────────────────────
def test_two_claps_fire():
    from core.wake import ClapDetector

    det = ClapDetector(sensitivity=1.0, cooldown=2.0)
    res = _stream(det, _two_claps(0.4), start=10.0)
    assert res is not None
    assert res.count >= 2
    assert 0.0 <= res.confidence <= 1.0


def test_single_clap_no_fire():
    from core.wake import ClapDetector

    det = ClapDetector(sensitivity=1.0, cooldown=2.0)
    res = _stream(det, _clap(), start=10.0)
    assert res is None


def test_spaced_claps_no_fire():
    from core.wake import ClapDetector

    det = ClapDetector(sensitivity=1.0, cooldown=2.0, window=1.2)
    c = _clap()
    gap = np.zeros(int(2.0 * SR), np.float32)  # > window apart
    res = _stream(det, np.concatenate([c, gap, c]), start=10.0)
    assert res is None


def test_speech_rejected():
    from core.wake import ClapDetector

    det = ClapDetector(sensitivity=1.0, cooldown=2.0)
    res = _stream(det, _speech(), start=10.0)
    assert res is None


def test_typing_rejected():
    from core.wake import ClapDetector

    det = ClapDetector(sensitivity=1.0, cooldown=2.0)
    res = _stream(det, _typing(), start=10.0)
    assert res is None


# ── WakeEngine ────────────────────────────────────────────────────────────────
def test_engine_arms_on_claps():
    from core.wake import WakeEngine

    eng = WakeEngine(require_clap=True, arm_window=10.0)
    armed = False
    for _ in range(3):  # feed the pair a few times to be safe
        r = _stream(eng, _two_claps(0.4), start=time.time())
        if r is not None and r.armed:
            armed = True
            break
        time.sleep(0.1)
    assert armed
    assert eng.armed


def test_require_clap_blocks_phrase_only():
    from core.wake import WakeEngine

    eng = WakeEngine(require_clap=True, arm_window=10.0)
    # Force the phrase detector to "hear" wake up without any clap.
    eng.phrase.detect = lambda *a, **k: "wake up"
    r = _stream(eng, _speech(), start=time.time())
    assert r is None or not r.awake
    assert not eng.armed


def test_wake_after_arm_then_phrase():
    from core.wake import WakeEngine

    eng = WakeEngine(require_clap=True, arm_window=10.0)
    phase = {"value": "arm"}

    def _fake_detect(*_a, **_k):
        # Only "hear" the phrase during the dedicated phrase phase, so the
        # clap audio itself doesn't instantly satisfy the wake.
        return "wake up" if phase["value"] == "phrase" else None

    eng.phrase.detect = _fake_detect

    armed = False
    for _ in range(3):
        r = _stream(eng, _two_claps(0.4), start=time.time())
        if r is not None and r.armed:
            armed = True
            break
        time.sleep(0.1)
    assert armed
    assert eng.armed

    phase["value"] = "phrase"
    r = _stream(eng, _speech(), start=time.time() + 5.0)
    assert r is not None and r.awake
    assert r.phrase == "wake up"


# ── config sanity ─────────────────────────────────────────────────────────────
def test_wake_config_constants_present():
    import config

    for name in (
        "CLAP_ENABLED", "CLAP_SENSITIVITY", "CLAP_COUNT", "CLAP_WINDOW",
        "CLAP_COOLDOWN", "WAKE_TIMEOUT", "WAKE_WORDS", "WAKE_REQUIRE_CLAP",
        "WAKE_BEEP",
    ):
        assert hasattr(config, name), f"config missing {name}"
    assert "wake up" in [w.lower() for w in config.WAKE_WORDS]
    assert config.WAKE_REQUIRE_CLAP is True


def test_hand_camera_config_removed():
    import config

    # Hand-gesture / camera-wake constants must be gone; the plain camera
    # constants are intentionally kept for vision / face recognition.
    for gone in (
        "HAND_DETECTION_CONFIDENCE", "HAND_TRACKING_CONFIDENCE",
        "GESTURE_SWIPE_THRESHOLD", "HAND_DEBUG", "CLAP_COOLDOWN_NOT_USED",
    ):
        assert not hasattr(config, gone), f"legacy config still present: {gone}"
