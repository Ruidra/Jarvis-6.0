"""
JARVIS Real-time Multimodal Perception Engine — JARVIS 7.0.

Fuses information from **multiple real-time sensors** into a single
per-frame "scene understanding" snapshot that the model (or the autonomy
engine) can reason over:

  * **Audio**  — wake-engine clap detector, microphone energy levels,
                and the Vosk/STT partial transcript currently in flight.
  * **Vision** — screen OCR (what's on your monitor), webcam snapshot
                (face detection, gesture), or a screenshot from a browser tab.
  * **Text**   — whatever the user just typed or the most recent Gemini
                transcription.
  * **System** — CPU / RAM / battery / network status from the
                background system monitor.

The key design choice: instead of each consumer polling its own sensor,
``MultimodalContext.get()`` returns a **single lightweight dict** snapshot
that merges everything. The snapshot is updated continuously by a background
thread, so the model only needs one call to know what JARVIS "perceives".

Example::

    from core.multimodal import MultimodalContext

    ctx = MultimodalContext()
    snapshot = ctx.perceive()
    # -> {"audio": {...}, "vision": {...}, "system": {...}, "timestamp": 1234.5}
    #
    # snapshot["audio"]["addressed"]   -> True/False
    # snapshot["audio"]["transcript"] -> "jarvis what time is it"
    # snapshot["system"]["cpu_pct"]   -> 12.5
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("jarvis.multimodal")


@dataclass
class AudioPerception:
    """What the mic is telling us right now."""
    addressed: bool = False          # is speech currently directed at JARVIS?
    transcript: str = ""             # latest STT partial or final transcript
    energy: float = 0.0              # current RMS of the mic buffer
    clap_detected: bool = False      # was a clap heard in the last few seconds?


@dataclass
class VisionPerception:
    """What the camera / screen is telling us."""
    screen_text: str = ""            # OCR result from the current screen
    faces_detected: int = 0          # how many faces the webcam sees
    gestures: list[str] = field(default_factory=list)  # hand gestures
    active_window: str = ""          # title of the currently focused window


@dataclass
class SystemPerception:
    """System health and resource state."""
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    battery_pct: float = 0.0
    network_up: bool = True
    uptime_s: float = 0.0


class MultimodalContext:
    """Thread-safe, real-time fused perception of the environment.

    Sensors update their respective fields independently. ``perceive()``
    returns a **shallow snapshot** — cheap enough to call every turn without
    blocking the audio callback thread.
    """

    def __init__(self, poll_interval: float = 1.0) -> None:
        self._lock = threading.RLock()
        self._audio = AudioPerception()
        self._vision = VisionPerception()
        self._system = SystemPerception()
        self._text: str = ""                       # latest user text input
        self._timestamp: float = 0.0

        self._poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None

    # ── Live updates (called by the relevant subsystems) ──────────────────────
    def set_audio(self, transcript: str = "", energy: float = 0.0,
                  clap: bool = False, addressed: bool = False) -> None:
        with self._lock:
            if transcript:
                self._audio.transcript = transcript
            self._audio.energy = energy
            self._audio.clap_detected = clap
            self._audio.addressed = addressed
            self._timestamp = time.time()

    def set_text(self, text: str) -> None:
        with self._lock:
            self._text = text
            self._timestamp = time.time()

    def set_vision(self, screen_text: str = "", faces: int = 0,
                   gestures: list[str] | None = None,
                   active_window: str = "") -> None:
        with self._lock:
            if screen_text:
                self._vision.screen_text = screen_text
            self._vision.faces_detected = faces
            if gestures is not None:
                self._vision.gestures = gestures
            if active_window:
                self._vision.active_window = active_window
            self._timestamp = time.time()

    def set_system(self, cpu: float = 0.0, ram: float = 0.0,
                   battery: float = 0.0, network_up: bool = True,
                   uptime: float = 0.0) -> None:
        with self._lock:
            self._system.cpu_pct = cpu
            self._system.ram_pct = ram
            self._system.battery_pct = battery
            self._system.network_up = network_up
            self._system.uptime_s = uptime
            self._timestamp = time.time()

    # ── Snapshot ───────────────────────────────────────────────────────────────
    def perceive(self) -> dict[str, Any]:
        """Return a merged snapshot of all sensors (cheap, thread-safe)."""
        with self._lock:
            return {
                "timestamp": self._timestamp,
                "audio": {
                    "addressed": self._audio.addressed,
                    "transcript": self._audio.transcript,
                    "energy": round(self._audio.energy, 4),
                    "clap_detected": self._audio.clap_detected,
                },
                "vision": {
                    "screen_text": self._vision.screen_text[:200],
                    "faces_detected": self._vision.faces_detected,
                    "gestures": list(self._vision.gestures),
                    "active_window": self._vision.active_window,
                },
                "system": {
                    "cpu_pct": round(self._system.cpu_pct, 1),
                    "ram_pct": round(self._system.ram_pct, 1),
                    "battery_pct": round(self._system.battery_pct, 1),
                    "network_up": self._system.network_up,
                    "uptime_s": round(self._system.uptime_s, 0),
                },
                "text": self._text,
            }

    # ── Background poller for system stats ────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="multimodal-poll"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _poll_loop(self) -> None:
        """Continuously update system + active-window stats in the background."""
        while self._running:
            try:
                self._update_system_stats()
            except Exception as exc:  # noqa: BLE001
                logger.debug("multimodal poll error: %s", exc)
            time.sleep(self._poll_interval)

    def _update_system_stats(self) -> None:
        try:
            import psutil

            self.set_system(
                cpu=psutil.cpu_percent(interval=0.5),
                ram=psutil.virtual_memory().percent,
                battery=(psutil.sensors_battery().percent if psutil.sensors_battery() else 0.0),
                network_up=bool(psutil.net_if_addrs()),
                uptime=time.time() - psutil.boot_time(),
            )
        except ImportError:
            # Fallback: use /proc on Linux
            try:
                import os
                if os.name == "posix":
                    stat = open("/proc/stat").read()
                    parts = stat.split("\n")[0].split()
                    idle = int(parts[4])
                    total = sum(int(p) for p in parts[1:])
                    cpu = 100 * (1 - idle / total) if total else 0
                    self.set_system(cpu=cpu, ram=50.0, uptime=0.0)
            except Exception:
                pass
        # Best-effort active window title (Windows only)
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            fg = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(fg)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(fg, buf, length + 1)
            self.set_vision(active_window=buf.value)
        except Exception:
            pass

    # ── Combined query ────────────────────────────────────────────────────────
    def describe_for_llm(self) -> str:
        """Return a compact text description of the current perceptual state."""
        snap = self.perceive()
        parts: list[str] = []

        a = snap["audio"]
        if a["addressed"]:
            parts.append("The user is speaking to you.")
        if a["transcript"]:
            parts.append(f"Heard: '{a['transcript'][:80]}'")
        if a["clap_detected"]:
            parts.append("A clap was just detected.")
        if a["energy"] > 0.05:
            parts.append(f"Background audio level is moderate ({a['energy']:.2f}).")

        v = snap["vision"]
        if v["screen_text"]:
            parts.append(f"Screen shows: '{v['screen_text'][:60]}'")
        if v["active_window"]:
            parts.append(f"Active window: {v['active_window']}")
        if v["faces_detected"]:
            parts.append(f"{v['faces_detected']} face(s) visible on camera.")

        s = snap["system"]
        if s["cpu_pct"] > 80:
            parts.append(f"Warning: CPU at {s['cpu_pct']:.0f}%.")
        if s["battery_pct"] < 20:
            parts.append(f"Battery low: {s['battery_pct']:.0f}%.")
        if not s["network_up"]:
            parts.append("Network is down.")

        if not parts:
            return "Environment appears normal; no notable activity."
        return " | ".join(parts)


# Process-wide instance.
multimodal = MultimodalContext(poll_interval=2.0)
