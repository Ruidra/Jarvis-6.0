"""
core/wake.py — JARVIS audio wake engine (clap + wake phrase).

Replaces the old camera/MediaPipe hand-gesture system. Everything here runs on
the microphone only, fully offline, with no camera and no extra model downloads.

Two building blocks:

``ClapDetector``
    Detects a hand clap from raw mic audio. A clap is a very short, very loud
    broadband transient: energy jumps by orders of magnitude within ~10 ms and
    decays almost immediately. Speech, music and typing do not do that, so the
    detector keys on *attack sharpness* + *high-frequency content* + *fast decay*
    rather than plain loudness. Supports requiring N claps inside a time window
    (default: 2 claps within 1.2 s) which makes accidental triggers very rare.

``WakePhraseDetector``
    Confirms the spoken wake phrase ("wake up", "jarvis", ...). Backend chain,
    best first:
        1. Vosk        — real offline speech recognition, exact phrase match
        2. Porcupine   — dedicated wake-word engine (if the user installed it)
        3. energy VAD  — "someone spoke" fallback so the system still works

``WakeEngine``
    Glues them together into the flow the user asked for::

        (asleep) --clap--> ARMED --"wake up"--> AWAKE

    Clap alone arms JARVIS (beep + orange orb); the wake phrase inside
    ``arm_window`` seconds fully wakes it. Optional ``require_clap=False`` lets
    the phrase alone wake JARVIS too.

Everything is dependency-light: numpy only for the mandatory parts.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, Optional

import numpy as np

logger = logging.getLogger("jarvis.wake")

_INT16_MAX = 32768.0


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_float_mono(audio) -> np.ndarray:
    """Normalise any audio chunk (bytes / int16 / float) to float32 mono in [-1, 1]."""
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(audio, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(audio), dtype=np.int16).astype(np.float32) / _INT16_MAX
    else:
        arr = np.asarray(audio)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / _INT16_MAX
        elif arr.dtype == np.int32:
            arr = arr.astype(np.float32) / 2147483648.0
        else:
            arr = arr.astype(np.float32, copy=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-12))


def _zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of sign changes — cheap proxy for 'how much high frequency'."""
    if x.size < 2:
        return 0.0
    signs = np.signbit(x)
    return float(np.count_nonzero(signs[1:] != signs[:-1])) / (x.size - 1)


# ──────────────────────────────────────────────────────────────────────────────
# clap detection
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ClapEvent:
    """One accepted clap pattern (i.e. ``claps_required`` claps in a row)."""

    timestamp: float
    count: int
    confidence: float
    peak: float


class ClapDetector:
    """Microphone clap detector — sharp attack, bright spectrum, fast decay.

    Feed it audio with :meth:`feed`; it returns a :class:`ClapEvent` when the
    required number of claps happened inside ``window`` seconds, else ``None``.

    Detection runs on *overlapping* 12 ms windows with a 5 ms hop, so a clap is
    never missed because it straddled an analysis-block boundary. A window is a
    clap when all of these hold:

    ==================  ====================================================
    loud                absolute peak and RMS above the floor gates
    sharp attack        RMS jumped vs the 20-40 ms *before* the window
    above room noise    RMS far above the adaptive noise floor
    bright              high zero-crossing rate (claps are broadband)
    fast decay          RMS 10 ms later collapsed (speech/music sustain)
    ==================  ====================================================

    Parameters
    ----------
    sample_rate:
        Sample rate of the audio being fed (default 16 kHz).
    sensitivity:
        1.0 = normal. Higher (e.g. 1.5) = easier to trigger, lower (0.6) =
        stricter. Scales the loudness/attack thresholds.
    claps_required:
        How many claps must be heard (default 2 — "clap twice").
    window:
        Seconds allowed between the first and last clap of a pattern.
    cooldown:
        Seconds to ignore audio after firing, so one clap burst = one event.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        sensitivity: float = 1.0,
        claps_required: int = 2,
        window: float = 1.2,
        cooldown: float = 1.5,
        min_gap: float = 0.09,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.sensitivity = max(0.2, float(sensitivity))
        self.claps_required = max(1, int(claps_required))
        self.window = float(window)
        self.cooldown = float(cooldown)
        self.min_gap = float(min_gap)

        # Overlapping analysis: 12 ms window, 5 ms hop.
        self._win = max(64, int(self.sample_rate * 0.012))
        self._hop = max(32, int(self.sample_rate * 0.005))
        self._buf = np.zeros(0, dtype=np.float32)
        self._clock = 0.0            # wall time of the buffer's leading edge
        self._have_clock = False

        # Per-hop feature history: (time, rms, peak, zcr)
        self._feat: Deque[tuple[float, float, float, float]] = deque(maxlen=16)

        # Rolling noise floor (slow) so the detector adapts to the room.
        self._floor = 0.004
        self._claps: Deque[float] = deque(maxlen=8)
        # -inf so the very first clap is never swallowed by the cooldown,
        # whatever clock the caller feeds us (monotonic, 0-based, ...).
        self._last_clap = float("-inf")
        self._last_event = float("-inf")

        # Thresholds (scaled by sensitivity)
        self._min_peak = 0.18 / self.sensitivity        # absolute sample peak
        self._min_rms = 0.07 / self.sensitivity         # window loudness
        self._min_attack = 4.0 / self.sensitivity       # rms jump vs 20-40 ms before
        self._min_floor_ratio = 8.0 / self.sensitivity  # rms vs adaptive noise floor
        self._min_zcr = 0.05                            # broadband / bright content
        self._max_decay = 0.45                          # rms must collapse 10 ms later

    # -- public API ---------------------------------------------------------
    @property
    def noise_floor(self) -> float:
        return self._floor

    def reset(self) -> None:
        self._claps.clear()
        self._buf = np.zeros(0, dtype=np.float32)
        self._feat.clear()
        self._have_clock = False

    def feed(self, audio, timestamp: Optional[float] = None) -> Optional[ClapEvent]:
        """Process an audio chunk. Returns a ClapEvent when the pattern completes."""
        samples = to_float_mono(audio)
        if samples.size == 0:
            return None

        now = time.monotonic() if timestamp is None else float(timestamp)
        if now - self._last_event < self.cooldown:
            # Still in cooldown: keep the noise floor fresh but ignore triggers.
            self._track_floor(samples)
            self._buf = np.zeros(0, dtype=np.float32)
            self._feat.clear()
            self._have_clock = False
            return None

        # Anchor the analysis clock to the leading edge of the buffer.
        if not self._have_clock:
            self._clock = now
            self._have_clock = True
        self._buf = np.concatenate((self._buf, samples)) if self._buf.size else samples

        event: Optional[ClapEvent] = None
        while self._buf.size >= self._win:
            win = self._buf[: self._win]
            self._feat.append(
                (self._clock, _rms(win), float(np.max(np.abs(win))), _zero_crossing_rate(win))
            )
            self._track_floor(win[: self._hop])
            self._buf = self._buf[self._hop :]
            self._clock += self._hop / self.sample_rate

            ev = self._evaluate()
            if ev is not None:
                event = ev
                break

        # Keep the buffer bounded (0.5 s max) — this is a live detector.
        if self._buf.size > self.sample_rate // 2:
            self._buf = self._buf[-(self.sample_rate // 2) :]
            self._have_clock = False
        return event

    # Backwards-compatible alias (old API used .update()).
    update = feed

    # -- internals ----------------------------------------------------------
    def _track_floor(self, samples: np.ndarray) -> None:
        r = _rms(samples)
        # Adapt fast when quieter, slowly when louder → avoids the floor being
        # dragged up by the clap itself.
        alpha = 0.05 if r < self._floor else 0.001
        self._floor = max(1e-4, (1 - alpha) * self._floor + alpha * r)

    def _evaluate(self) -> Optional[ClapEvent]:
        """Test the hop that now has 2 hops of lookahead and enough history."""
        # layout: [... history ...][candidate][+1][+2]
        if len(self._feat) < 9:
            return None
        cand_i = len(self._feat) - 3
        t, rms, peak, zcr = self._feat[cand_i]

        if peak < self._min_peak or rms < self._min_rms:
            return None

        # Must be the local energy maximum (the onset hop, not its tail).
        if rms < self._feat[cand_i - 1][1] or rms < self._feat[cand_i + 1][1]:
            return None

        # Attack: compare against 20-40 ms before the onset (skip the hop right
        # before it, which may already overlap the clap).
        pre = [f[1] for f in list(self._feat)[cand_i - 8 : cand_i - 2]]
        if not pre:
            return None
        baseline = max(min(pre), 1e-5)
        attack = rms / baseline
        if attack < self._min_attack:
            return None

        if rms / max(self._floor, 1e-5) < self._min_floor_ratio:
            return None
        if zcr < self._min_zcr:
            return None  # low/rumbly (door thud, bass) — not a clap

        decay = self._feat[cand_i + 2][1] / max(rms, 1e-5)
        if decay > self._max_decay:
            return None  # sustained (speech / music) — not a clap

        conf = min(
            1.0,
            0.35 * min(peak / max(self._min_peak, 1e-6), 3.0) / 3.0
            + 0.35 * min(attack / max(self._min_attack, 1e-6), 4.0) / 4.0
            + 0.30 * (1.0 - min(decay / max(self._max_decay, 1e-6), 1.0)),
        )
        return self._register(t, conf, peak)

    def _register(self, now: float, conf: float, peak: float) -> Optional[ClapEvent]:
        if now - self._last_clap < self.min_gap:
            return None  # same clap's tail / echo
        self._last_clap = now
        self._claps.append(now)
        # Drop claps that fell out of the pattern window.
        while self._claps and (now - self._claps[0]) > self.window:
            self._claps.popleft()

        logger.debug("[CLAP] hit conf=%.2f peak=%.2f count=%d", conf, peak, len(self._claps))
        if len(self._claps) >= self.claps_required:
            self._last_event = now
            count = len(self._claps)
            self._claps.clear()
            logger.info("[CLAP] pattern accepted (%d claps, conf=%.2f)", count, conf)
            return ClapEvent(timestamp=now, count=count, confidence=conf, peak=peak)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# wake phrase detection
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_WAKE_PHRASES = [
    "wake up",
    "jarvis",
    "hey jarvis",
    "wake up jarvis",
    "jarvis wake up",
]

# Small English model used for exact wake-phrase matching (~40 MB, one-time).
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"


def find_vosk_model() -> Optional[str]:
    """Locate an already-downloaded Vosk model, or return None.

    We deliberately never trigger Vosk's automatic download here: that would
    block JARVIS (or the Sentinel) at startup for tens of megabytes. Use
    :func:`download_vosk_model` (``python sentinel.py --install-vosk``) once and
    every later start finds it instantly.
    """
    from pathlib import Path

    candidates: list[Path] = []
    try:
        from config import get_config

        configured = (get_config().get("vosk_model_path") or "").strip()
        if configured:
            candidates.append(Path(configured))
    except Exception:
        pass

    root = Path(__file__).resolve().parent.parent
    candidates += [
        root / "models" / VOSK_MODEL_NAME,
        root / VOSK_MODEL_NAME,
        Path.home() / ".cache" / "vosk" / VOSK_MODEL_NAME,
    ]
    # Any vosk-model-* directory inside models/ or the vosk cache.
    for parent in (root / "models", Path.home() / ".cache" / "vosk"):
        try:
            if parent.is_dir():
                candidates += sorted(parent.glob("vosk-model-*"))
        except Exception:
            pass

    for path in candidates:
        try:
            if path.is_dir() and (path / "am").is_dir():
                return str(path)
        except Exception:
            continue
    return None


def download_vosk_model(dest: Optional[str] = None) -> str:
    """One-time download of the small English Vosk model. Returns its path."""
    import shutil
    import urllib.request
    import zipfile
    from pathlib import Path

    existing = find_vosk_model()
    if existing:
        return existing

    root = Path(__file__).resolve().parent.parent
    target_dir = Path(dest) if dest else (root / "models")
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
    archive = target_dir / f"{VOSK_MODEL_NAME}.zip"

    logger.info("[WAKE] downloading %s (~40 MB, one time)…", VOSK_MODEL_NAME)
    with urllib.request.urlopen(url, timeout=120) as resp, archive.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)
    archive.unlink(missing_ok=True)

    model_dir = target_dir / VOSK_MODEL_NAME
    if not (model_dir / "am").is_dir():
        raise RuntimeError(f"Vosk model extracted to an unexpected layout: {model_dir}")
    logger.info("[WAKE] Vosk model ready at %s", model_dir)
    return str(model_dir)


class WakePhraseDetector:
    """Detect a spoken wake phrase. Vosk → Porcupine → energy VAD.

    ``detect()`` accepts any audio chunk and returns the matched phrase (str)
    or ``None``. With the Vosk backend the match is a real transcript match, so
    "wake up" only triggers on those words. Without Vosk it degrades to "speech
    was heard", which is still safe because the clap has to arm JARVIS first.
    """

    def __init__(
        self,
        phrases: Optional[Iterable[str]] = None,
        sample_rate: int = 16000,
        sensitivity: float = 0.5,
        energy_threshold: float = 0.012,
    ) -> None:
        self.phrases = [p.strip().lower() for p in (phrases or DEFAULT_WAKE_PHRASES) if p and p.strip()]
        self.sample_rate = int(sample_rate)
        self.energy_threshold = float(energy_threshold)
        self._vosk = None
        self._vosk_model = None
        self._porcupine = None
        self._speech_blocks = 0
        self._last_hit = float("-inf")
        self._cooldown = 1.0

        self._try_vosk()
        if self._vosk is None:
            self._try_porcupine(sensitivity)

    # -- backends -----------------------------------------------------------
    def _try_vosk(self) -> None:
        model_path = find_vosk_model()
        if not model_path:
            logger.info(
                "[WAKE] no local Vosk model — exact phrase matching disabled "
                "(run: python sentinel.py --install-vosk)"
            )
            self._vosk = None
            return
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore

            SetLogLevel(-1)
            model = Model(model_path=model_path)
            # Constrain the grammar to the wake phrases → tiny, fast, accurate.
            grammar = '["' + '", "'.join(sorted(set(self.phrases))) + '", "[unk]"]'
            try:
                self._vosk = KaldiRecognizer(model, self.sample_rate, grammar)
            except Exception:
                self._vosk = KaldiRecognizer(model, self.sample_rate)
            self._vosk_model = model  # keep a reference alive
            logger.info("[WAKE] phrase backend: Vosk (%s)", model_path)
        except Exception as exc:  # noqa: BLE001 — Vosk is optional
            logger.info("[WAKE] Vosk unavailable (%s)", exc)
            self._vosk = None

    def _try_porcupine(self, sensitivity: float) -> None:
        try:
            import pvporcupine  # type: ignore

            self._porcupine = pvporcupine.create(
                keywords=["jarvis"], sensitivities=[max(0.1, min(1.0, sensitivity))]
            )
            logger.info("[WAKE] phrase backend: Porcupine")
        except Exception as exc:  # noqa: BLE001 — Porcupine is optional
            logger.info("[WAKE] phrase backend: energy VAD fallback (%s)", exc)
            self._porcupine = None

    @property
    def backend(self) -> str:
        if self._vosk is not None:
            return "vosk"
        if self._porcupine is not None:
            return "porcupine"
        return "energy"

    @property
    def is_exact(self) -> bool:
        """True when the backend really recognises words (not just loudness)."""
        return self._vosk is not None or self._porcupine is not None

    # -- detection ----------------------------------------------------------
    def reset(self) -> None:
        self._speech_blocks = 0

    def detect(self, audio, timestamp: Optional[float] = None) -> Optional[str]:
        samples = to_float_mono(audio)
        if samples.size == 0:
            return None
        now = time.monotonic() if timestamp is None else float(timestamp)
        if now - self._last_hit < self._cooldown:
            return None

        hit: Optional[str] = None
        if self._vosk is not None:
            hit = self._detect_vosk(samples)
        elif self._porcupine is not None:
            hit = self._detect_porcupine(samples)
        else:
            hit = self._detect_energy(samples)

        if hit:
            self._last_hit = now
            self._speech_blocks = 0
            logger.info("[WAKE] phrase matched via %s: %r", self.backend, hit)
        return hit

    def _detect_vosk(self, samples: np.ndarray) -> Optional[str]:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        try:
            import json

            if self._vosk.AcceptWaveform(pcm):
                text = (json.loads(self._vosk.Result()).get("text") or "").lower()
            else:
                text = (json.loads(self._vosk.PartialResult()).get("partial") or "").lower()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[WAKE] Vosk failure (%s) — falling back to energy", exc)
            self._vosk = None
            return self._detect_energy(samples)
        if not text:
            return None
        for phrase in self.phrases:
            if phrase in text:
                return phrase
        return None

    def _detect_porcupine(self, samples: np.ndarray) -> Optional[str]:
        try:
            pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
            n = self._porcupine.frame_length
            for i in range(0, pcm.size - n + 1, n):
                if self._porcupine.process(pcm[i : i + n]) >= 0:
                    return "jarvis"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[WAKE] Porcupine failure (%s) — falling back to energy", exc)
            self._porcupine = None
            return self._detect_energy(samples)
        return None

    def _detect_energy(self, samples: np.ndarray) -> Optional[str]:
        """Fallback: require ~0.25 s of continuous speech-level audio."""
        if _rms(samples) > self.energy_threshold:
            self._speech_blocks += 1
        else:
            self._speech_blocks = max(0, self._speech_blocks - 1)
        needed = max(2, int(0.25 * self.sample_rate / max(samples.size, 1)))
        if self._speech_blocks >= needed:
            self._speech_blocks = 0
            return "speech"
        return None


# ──────────────────────────────────────────────────────────────────────────────
# combined engine
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class WakeResult:
    """What happened for the audio chunk just processed."""

    armed: bool = False        # a clap pattern just armed JARVIS
    awake: bool = False        # the wake phrase confirmed → wake up now
    disarmed: bool = False     # the arm window expired without a phrase
    phrase: str = ""
    clap: Optional[ClapEvent] = None

    def __bool__(self) -> bool:  # truthy when anything happened
        return bool(self.armed or self.awake or self.disarmed)


class WakeEngine:
    """Clap ➜ wake-phrase state machine driven purely by microphone audio.

    Usage::

        engine = WakeEngine()
        result = engine.feed(pcm_chunk)      # call from the mic callback
        if result.armed: play_beep()
        if result.awake: start_conversation()

    Parameters
    ----------
    require_clap:
        ``True`` (default) → the clap must arm JARVIS before the phrase counts.
        ``False`` → saying the phrase alone is enough (hands-free).
    arm_window:
        Seconds the armed state stays open waiting for the phrase.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        phrases: Optional[Iterable[str]] = None,
        sensitivity: float = 1.0,
        claps_required: int = 2,
        clap_window: float = 1.2,
        clap_cooldown: float = 1.5,
        arm_window: float = 10.0,
        require_clap: bool = True,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.arm_window = float(arm_window)
        self.require_clap = bool(require_clap)
        self.on_event = on_event

        self.clap = ClapDetector(
            sample_rate=sample_rate,
            sensitivity=sensitivity,
            claps_required=claps_required,
            window=clap_window,
            cooldown=clap_cooldown,
        )
        self.phrase = WakePhraseDetector(phrases=phrases, sample_rate=sample_rate)
        self._armed_at: Optional[float] = None

    # -- state --------------------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed_at is not None

    @property
    def backend(self) -> str:
        return self.phrase.backend

    def arm(self, timestamp: Optional[float] = None) -> None:
        """Arm manually (e.g. user pressed a button or typed 'wake')."""
        self._armed_at = time.monotonic() if timestamp is None else float(timestamp)
        self.phrase.reset()

    def disarm(self) -> None:
        self._armed_at = None
        self.phrase.reset()
        self.clap.reset()

    def reset(self) -> None:
        self.disarm()

    # -- main loop ----------------------------------------------------------
    def feed(self, audio, timestamp: Optional[float] = None) -> WakeResult:
        now = time.monotonic() if timestamp is None else float(timestamp)
        result = WakeResult()

        clap_event = self.clap.feed(audio, timestamp=now)
        if clap_event is not None:
            result.clap = clap_event
            if not self.armed:
                self._armed_at = now
                result.armed = True
                self._emit("wake.armed", {"claps": clap_event.count,
                                          "confidence": clap_event.confidence})
            else:
                self._armed_at = now  # re-clap extends the window

        # Expire the armed window.
        if self.armed and (now - float(self._armed_at)) > self.arm_window:
            self.disarm()
            result.disarmed = True
            self._emit("wake.disarmed", {})

        listening_for_phrase = self.armed or not self.require_clap
        if listening_for_phrase:
            phrase = self.phrase.detect(audio, timestamp=now)
            if phrase:
                # With a loudness-only backend, a clap itself can look like
                # "speech"; ignore a phrase hit in the same chunk as the clap.
                if clap_event is not None and not self.phrase.is_exact:
                    return result
                self.disarm()
                result.awake = True
                result.phrase = phrase
                self._emit("wake.awake", {"phrase": phrase, "backend": self.backend})
        return result

    # -- misc ---------------------------------------------------------------
    def _emit(self, name: str, data: dict) -> None:
        if self.on_event is not None:
            try:
                self.on_event(name, data)
            except Exception:  # noqa: BLE001 — never break the audio thread
                logger.debug("wake event callback failed", exc_info=True)
        try:
            from core.event_bus import bus

            bus.emit(name, data, source="wake")
        except Exception:  # noqa: BLE001 — event bus is optional
            pass

    def status(self) -> dict:
        return {
            "armed": self.armed,
            "backend": self.backend,
            "exact_phrase_match": self.phrase.is_exact,
            "phrases": list(self.phrase.phrases),
            "claps_required": self.clap.claps_required,
            "noise_floor": round(self.clap.noise_floor, 5),
        }
