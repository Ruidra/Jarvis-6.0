"""
Voice-Based Emotion Detection for Jarvis.

Analyzes the *audio signal itself* (not just transcribed text) for paralinguistic
emotional cues: fundamental frequency (pitch), pitch variation (jitter),
energy/volume, zero-crossing rate (tension), and speech rate. Maps these
prosodic features to a coarse emotion label and intensity.

This complements :mod:`core.emotion_engine` which only looks at words.

All computation is dependency-free (pure numpy + soundfile-style PCM math).
Input audio is 16-bit mono PCM at 16 kHz (SEND_SAMPLE_RATE).

Example::

    from core.voice_emotion import VoiceEmotionAnalyzer
    vea = VoiceEmotionAnalyzer()
    result = vea.analyze(pcm_bytes_16bit)
    result.emotion     # 'happy' | 'sad' | 'angry' | 'anxious' | 'neutral'
    result.intensity   # 0.0 .. 1.0
    result.energy      # mean RMS amplitude
    result.pitch_hz    # mean fundamental frequency
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class VoiceEmotionResult:
    emotion: str       # 'happy', 'sad', 'angry', 'anxious', 'neutral', 'surprised'
    intensity: float   # 0.0 .. 1.0
    label: str        # 'positive' | 'negative' | 'neutral'
    pitch_hz: float   # mean estimated F0
    pitch_var: float   # std of F0 (jitter proxy)
    energy: float     # mean RMS amplitude
    zcr: float        # zero-crossing rate (tension proxy)
    speech_rate: float  # zero-crossing-based voicing rate
    prosody: dict     # TTS prosody hints

    def to_emotion_result_compat(self):
        """Return a dict shaped like EmotionEngine.analyze output for compatibility."""
        from core.emotion_engine import EmotionResult, _PROSODY
        return EmotionResult(
            label=self.label,
            score=0.5 if self.emotion == "neutral" else (0.7 if self.label == "positive" else -0.7),
            emotions=[self.emotion],
            dominant=self.emotion,
            intensity=self.intensity,
            prosody=self.prosody,
            empathy_directive="",
            words=[],
        )


# --------------------------------------------------------------------------- #
# Pitch (F0) estimation via autocorrelation — no external deps               #
# --------------------------------------------------------------------------- #

def _estimate_f0(frame: np.ndarray, sample_rate: int) -> float:
    """Estimate fundamental frequency (Hz) via normalized autocorrelation.

    Returns 0.0 if no periodicity is found (unvoiced/silence).
    """
    frame = frame - np.mean(frame)                       # remove DC
    frame = frame * np.hanning(len(frame))               # window
    ac = np.correlate(frame, frame, mode='full')
    ac = ac[len(ac) // 2:]                                # keep positive lags only
    ac = ac / (ac[0] + 1e-10)                             # normalize

    # Search for the first peak above threshold in a pitch-sane lag range
    min_lag = int(sample_rate / 800)   # ~800 Hz upper bound
    max_lag = int(sample_rate / 50)    # ~50 Hz lower bound
    ac[:min_lag] = 0
    ac[max_lag + 1:] = 0

    peak_idx = np.argmax(ac)
    peak_val = ac[peak_idx]

    if peak_val < 0.3 or peak_idx == 0:
        return 0.0  # unvoiced or silence

    # Parabolic interpolation for sub-sample precision
    if 0 < peak_idx < len(ac) - 1:
        α, β, γ = ac[peak_idx - 1], ac[peak_idx], ac[peak_idx + 1]
        denom = (2 * (2 * β - α - γ))
        if abs(denom) > 1e-10:
            offset = 0.5 * (α - γ) / denom
            offset = max(-1, min(1, offset))
        else:
            offset = 0.0
    else:
        offset = 0.0

    f0 = sample_rate / (peak_idx + offset)
    return float(f0)


def _extract_frames(pcm: np.ndarray, sample_rate: int, frame_ms: float = 20.0) -> list[np.ndarray]:
    """Split audio into overlapping frames."""
    frame_size = int(sample_rate * frame_ms / 1000)
    hop_size = frame_size // 2
    frames = []
    for i in range(0, len(pcm) - frame_size + 1, hop_size):
        frames.append(pcm[i:i + frame_size])
    if not frames:
        return [pcm]
    return frames


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame ** 2))) if len(frame) > 0 else 0.0


def _zero_crossing_rate(frame: np.ndarray, sample_rate: int) -> float:
    if len(frame) < 2:
        return 0.0
    signs = np.sign(frame)
    crossings = np.sum(signs[1:] != signs[:-1])
    return float(crossings / (len(frame) / sample_rate))


# --------------------------------------------------------------------------- #
# Main analyzer                                                                #
# --------------------------------------------------------------------------- #

class VoiceEmotionAnalyzer:
    """
    Analyzes a PCM audio segment (int16, mono, 16 kHz) for vocal emotional cues.

    Usage: accumulate user speech audio while JARVIS is listening, then call
    ``analyze()`` on the full segment when the user finishes a turn.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._buffer: list[bytes] = []
        self._buffer_len: int = 0
        self._min_segment_samples: int = int(sample_rate * 0.5)  # need at least 0.5s

    # ── accumulate PCM from the receive loop ──────────────────────────────────
    def add_chunk(self, data: bytes | bytearray | memoryview) -> None:
        if isinstance(data, memoryview):
            data = bytes(data)
        self._buffer.append(data)
        self._buffer_len += len(data)

    def reset(self) -> None:
        self._buffer = []
        self._buffer_len = 0

    def get_buffer(self) -> np.ndarray:
        """Return accumulated PCM as float32 numpy array."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        raw = b"".join(self._buffer)
        arr = np.frombuffer(raw, dtype=np.int16)
        return arr.astype(np.float32) / 32768.0

    # ── core analysis ─────────────────────────────────────────────────────────
    def analyze(
        self,
        pcm: bytes | bytearray | memoryview | np.ndarray | None = None,
    ) -> Optional[VoiceEmotionResult]:
        """Analyze audio and return emotion result, or None if too short/silent."""
        if pcm is not None:
            if isinstance(pcm, (bytes, bytearray, memoryview)):
                arr = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(pcm, np.ndarray):
                arr = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm.astype(np.float32)
            else:
                return None
        else:
            arr = self.get_buffer()
            self.reset()

        if len(arr) < self._min_segment_samples:
            return None

        frames = _extract_frames(arr, self.sample_rate)
        if not frames:
            return None

        # Compute per-frame features, then aggregate
        pitches = []
        energies = []
        zcrs = []

        for frame in frames:
            rms_val = _rms(frame)
            if rms_val < 0.005:  # skip silence (< ~1% RMS)
                continue
            f0 = _estimate_f0(frame, self.sample_rate)
            if f0 > 0:
                pitches.append(f0)
            energies.append(rms_val)
            zcrs.append(_zero_crossing_rate(frame, self.sample_rate))

        if not energies:
            return None

        mean_energy = float(np.mean(energies))
        mean_zcr = float(np.mean(zcrs)) if zcrs else 0.0

        if pitches:
            mean_pitch = float(np.mean(pitches))
            std_pitch = float(np.std(pitches))
            pitch_var = float(np.std(pitches) / (mean_pitch + 1e-10))  # coefficient of variation
        else:
            mean_pitch = 0.0
            std_pitch = 0.0
            pitch_var = 0.0

        # Normalize energy to a 0-1 scale relative to typical speech
        # (0.01 RMS is quiet, 0.1+ is loud/shouting)
        norm_energy = min(1.0, mean_energy / 0.08)

        # ----------------------------------------------------------------------- #
        # Emotion mapping from prosodic features                                  #
        # ----------------------------------------------------------------------- #
        results = _map_prosody_to_emotion(
            mean_pitch=mean_pitch,
            pitch_var=pitch_var,
            energy=norm_energy,
            zcr=mean_zcr,
        )

        # Blend a small amount of voice emotion into any text-based emotion
        # (the caller will merge via EmotionEngine if both are available).
        results["pitch_hz"] = mean_pitch
        results["pitch_std"] = std_pitch
        results["energy"] = mean_energy
        results["zcr"] = mean_zcr

        return results

    def analyze_and_merge(
        self,
        text_emotion_result,
        pcm: bytes | bytearray | memoryview | None = None,
    ):
        """Merge voice emotion into an existing EmotionResult from text analysis.

        Voice prosody provides the *tone*; text provides the *content*. We
        prefer the voice's emotional label when its signal is strong enough,
        otherwise fall back to the text-based result.
        """
        voice = self.analyze(pcm) if pcm is not None else None
        if voice is None:
            return text_emotion_result

        # If voice emotion is clearly negative/positive, let it influence the result
        if voice["intensity"] > 0.4:
            # Voice detected stronger emotion — blend but keep text's content cues
            merged = dict(text_emotion_result.__dict__) if text_emotion_result else {}
            if voice["label"] != "neutral":
                merged["dominant"] = voice["dominant_emotion"]
                merged["label"] = voice["label"]
                merged["emotion_tag"] = f"voice:{voice['dominant_emotion']}"
            return VoiceEmotionResult(
                emotion=voice["dominant_emotion"],
                intensity=voice["intensity"],
                label=voice["label"],
                pitch_hz=voice["pitch_hz"],
                pitch_var=voice.get("pitch_std", 0.0),
                energy=voice["energy"],
                zcr=voice["zcr"],
                speech_rate=voice.get("speech_rate", 0.0),
                prosody=voice.get("prosody", {}),
            )
        return text_emotion_result


# --------------------------------------------------------------------------- #
# Feature → emotion mapping                                                    #
# --------------------------------------------------------------------------- #

def _map_prosody_to_emotion(
    mean_pitch: float,
    pitch_var: float,    # coefficient of variation of F0
    energy: float,       # normalized 0..1
    zcr: float,
) -> dict:
    """
    Map prosodic features to emotion.

    Key heuristics (from speech-acoustic emotion research):
      - High pitch + high pitch variation + high energy → excited/happy (cheer)
      - Low pitch + low pitch variation + low energy → sad
      - High energy + high ZCR + moderate pitch → angry
      - Low pitch variation + moderate energy → anxious/calm
      - Fast speech rate (high ZCR) + high pitch variation → surprised
    """
    # Speech rate proxy: ZCR is correlated with speech rate in voiced regions
    speech_rate = min(1.0, zcr / 6.0)  # normalised 0..1

    emotion = "neutral"
    intensity = 0.0
    label = "neutral"

    # --- Happy / Cheerful ---
    # High, variable pitch + high energy
    if mean_pitch > 160 and pitch_var > 0.08 and energy > 0.5:
        emotion = "happy"
        intensity = min(1.0, (pitch_var * 3 + energy * 1.5) / 2)
        label = "positive"

    # --- Angry ---
    # High energy, high ZCR (tension), mid-to-loud pitch
    elif energy > 0.6 and zcr > 4.0 and mean_pitch > 120:
        emotion = "angry"
        intensity = min(1.0, energy * 1.2)
        label = "negative"

    # --- Sad ---
    # Low pitch, low variation, low energy
    elif mean_pitch < 140 and pitch_var < 0.06 and energy < 0.45:
        emotion = "sad"
        intensity = min(1.0, (0.06 - pitch_var) * 8 + (0.45 - energy) * 1.5)
        label = "negative"

    # --- Anxious / Nervous ---
    # Low pitch variation, moderate energy, mid-high ZCR (speech tension)
    elif pitch_var < 0.05 and zcr > 2.5 and 0.3 < energy < 0.7:
        emotion = "anxious"
        intensity = min(1.0, (0.05 - pitch_var) * 12 + zcr * 0.15)
        label = "negative"

    # --- Surprised ---
    # Sudden high pitch + high variation
    elif mean_pitch > 200 and pitch_var > 0.12:
        emotion = "surprised"
        intensity = min(1.0, pitch_var * 4)
        label = "positive"

    # --- Calm / Neutral ---
    else:
        emotion = "neutral"
        intensity = 0.0
        label = "neutral"

    # Build prosody hints for TTS
    _EMOTION_PROSODY = {
        "happy":     {"rate": 1.06, "pitch": 1.12, "style": "cheerful"},
        "surprised": {"rate": 1.08, "pitch": 1.15, "style": "cheerful"},
        "angry":     {"rate": 1.02, "pitch": 0.95, "style": "firm"},
        "sad":       {"rate": 0.94, "pitch": 0.88, "style": "soft"},
        "anxious":   {"rate": 0.96, "pitch": 0.92, "style": "calm"},
        "neutral":   {"rate": 1.0,  "pitch": 1.0,  "style": "default"},
    }
    prosody = _EMOTION_PROSODY.get(emotion, _EMOTION_PROSODY["neutral"])

    return {
        "emotion": emotion,
        "dominant_emotion": emotion,
        "intensity": round(intensity, 3),
        "label": label,
        "pitch_hz": round(mean_pitch, 1),
        "pitch_std": round(float(np.std([mean_pitch])) if mean_pitch else 0.0, 2),
        "energy": round(energy, 3),
        "zcr": round(zcr, 3),
        "speech_rate": round(speech_rate, 3),
        "prosody": prosody,
    }
