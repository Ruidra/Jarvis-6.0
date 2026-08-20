"""
sentinel.py — JARVIS Sentinel: the always-on clap listener.

This is the piece that means **you never have to start JARVIS yourself**.

It is a tiny background process (a few MB of RAM, ~0.3 % CPU) that does one
thing: listen to the microphone for your clap pattern followed by the spoken
wake phrase. When it hears them it launches the full JARVIS and hands over,
so from your side the whole interaction is:

    *clap clap*  →  "wake up"  →  JARVIS boots and answers

While JARVIS is running the Sentinel goes quiet (it releases the microphone so
JARVIS can use it) and only starts listening again after JARVIS exits.

Run it directly::

    python sentinel.py                # foreground, with logs
    python sentinel.py --status       # is it running? what backend?
    python sentinel.py --test         # clap test mode: prints every clap heard
    python sentinel.py --stop         # stop a running sentinel

Or use the launchers: ``Sentinel.bat`` / ``install_autostart.bat``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE = LOG_DIR / "sentinel.pid"
LOG_FILE = LOG_DIR / "sentinel.log"
WAKE_REQUEST_FILE = LOG_DIR / ".wake_request"
STOP_FILE = LOG_DIR / ".sentinel_stop"

IS_WINDOWS = platform.system() == "Windows"

sys.path.insert(0, str(BASE_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# logging (tiny, no deps — the Sentinel must start instantly)
# ──────────────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# config
# ──────────────────────────────────────────────────────────────────────────────
def load_settings() -> dict:
    """Read wake settings from config/api_keys.json (with sane defaults)."""
    defaults = {
        "clap_count": 2,
        "clap_sensitivity": 1.0,
        "clap_window": 1.2,
        "clap_cooldown": 1.5,
        "wake_timeout": 12.0,
        "wake_words": ["wake up", "jarvis", "hey jarvis", "wake up jarvis"],
        "wake_require_clap": True,
        "sentinel_mic_index": None,
        "sentinel_launch_visible": True,
    }
    try:
        cfg = json.loads((BASE_DIR / "config" / "api_keys.json").read_text(encoding="utf-8"))
        for k in list(defaults):
            if k in cfg:
                defaults[k] = cfg[k]
    except Exception:
        pass
    return defaults


# ──────────────────────────────────────────────────────────────────────────────
# process helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if IS_WINDOWS:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def read_pid() -> int:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def already_running() -> bool:
    pid = read_pid()
    return pid > 0 and pid != os.getpid() and _pid_alive(pid)


def write_pid() -> None:
    try:
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass


def clear_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def jarvis_is_running() -> bool:
    """True when a JARVIS main.py process is alive (so we stay off the mic)."""
    try:
        import psutil  # type: ignore

        me = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if proc.info["pid"] in (me, os.getppid()):
                continue
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "main.py" in cmdline and "jarvis" in cmdline.replace("\\", "/"):
                return True
        return False
    except Exception:
        pass
    # psutil missing → fall back to a marker file written by our own launches.
    marker = LOG_DIR / ".jarvis_launched"
    try:
        if not marker.exists():
            return False
        pid = int(marker.read_text(encoding="utf-8").strip() or 0)
        if _pid_alive(pid):
            return True
        marker.unlink(missing_ok=True)
        return False
    except Exception:
        return False


def python_exe(gui: bool = True) -> str:
    """Prefer pythonw.exe on Windows so no console window pops up."""
    exe = Path(sys.executable)
    if gui and IS_WINDOWS:
        candidate = exe.with_name("pythonw.exe")
        if candidate.exists():
            return str(candidate)
    return str(exe)


def launch_jarvis(settings: dict) -> bool:
    """Start JARVIS (main.py) and record the PID so we know it's alive."""
    main_py = BASE_DIR / "main.py"
    if not main_py.exists():
        log(f"ERROR: {main_py} not found — cannot launch JARVIS.")
        return False
    try:
        WAKE_REQUEST_FILE.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

    cmd = [python_exe(gui=True), str(main_py)]
    kwargs: dict = {"cwd": str(BASE_DIR)}
    if IS_WINDOWS:
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **kwargs)
        (LOG_DIR / ".jarvis_launched").write_text(str(proc.pid), encoding="utf-8")
        log(f"JARVIS launched (pid {proc.pid}).")
        return True
    except Exception as exc:
        log(f"ERROR: failed to launch JARVIS: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# the listener
# ──────────────────────────────────────────────────────────────────────────────
class Sentinel:
    SAMPLE_RATE = 16000
    BLOCK = 1024

    def __init__(self, settings: dict, test_mode: bool = False) -> None:
        self.settings = settings
        self.test_mode = test_mode
        self.engine = None
        self._stop = False
        self._claps_seen = 0

    # -- engine ---------------------------------------------------------------
    def _build_engine(self):
        from core.wake import WakeEngine

        return WakeEngine(
            sample_rate=self.SAMPLE_RATE,
            phrases=self.settings["wake_words"],
            sensitivity=float(self.settings["clap_sensitivity"]),
            claps_required=int(self.settings["clap_count"]),
            clap_window=float(self.settings["clap_window"]),
            clap_cooldown=float(self.settings["clap_cooldown"]),
            arm_window=float(self.settings["wake_timeout"]),
            require_clap=bool(self.settings["wake_require_clap"]) and not self.test_mode,
        )

    # -- audio ----------------------------------------------------------------
    def run(self) -> int:
        try:
            import sounddevice as sd
        except Exception as exc:
            log(f"ERROR: sounddevice is required for the Sentinel: {exc}")
            return 2

        self.engine = self._build_engine()
        st = self.engine.status()
        log(
            f"Sentinel listening — clap x{st['claps_required']} then say "
            f"\"{st['phrases'][0]}\"  (phrase engine: {st['backend']})"
        )
        if not st["exact_phrase_match"]:
            log("Note: install Vosk for exact phrase matching → pip install vosk")
        if self.test_mode:
            log("TEST MODE: every clap pattern is printed, JARVIS is NOT launched.")

        device = self.settings.get("sentinel_mic_index")
        stream_kwargs = dict(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.BLOCK,
        )
        if device is not None:
            stream_kwargs["device"] = device

        paused_logged = False
        try:
            while not self._should_stop():
                # JARVIS already running → release the mic and idle.
                if not self.test_mode and jarvis_is_running():
                    if not paused_logged:
                        log("JARVIS is running — Sentinel paused (mic released).")
                        paused_logged = True
                    time.sleep(3.0)
                    continue
                if paused_logged:
                    log("JARVIS closed — Sentinel listening again.")
                    paused_logged = False
                    self.engine.disarm()

                try:
                    with sd.InputStream(**stream_kwargs) as stream:
                        self._listen(stream)
                except Exception as exc:
                    log(f"Microphone error ({exc}) — retrying in 5s.")
                    time.sleep(5.0)
        except KeyboardInterrupt:
            log("Sentinel stopped by user.")
        return 0

    def _listen(self, stream) -> None:
        """Inner loop: read blocks until JARVIS should be launched."""
        while not self._should_stop():
            if not self.test_mode and jarvis_is_running():
                return
            try:
                data, overflowed = stream.read(self.BLOCK)
            except Exception as exc:
                log(f"Audio read failed ({exc}).")
                return
            result = self.engine.feed(bytes(data))
            if not result:
                continue
            if result.armed:
                self._claps_seen += 1
                conf = result.clap.confidence if result.clap else 0.0
                log(f"Clap pattern #{self._claps_seen} heard ({conf:.0%}) — say \"wake up\".")
                if self.test_mode:
                    self.engine.disarm()
            if result.disarmed:
                log("No wake phrase heard — back to idle.")
            if result.awake:
                log(f"Wake phrase confirmed: \"{result.phrase}\" → starting JARVIS…")
                if self.test_mode:
                    continue
                launch_jarvis(self.settings)
                # Give JARVIS time to grab the microphone before we resume.
                time.sleep(12.0)
                return

    def _should_stop(self) -> bool:
        if self._stop:
            return True
        if STOP_FILE.exists():
            try:
                STOP_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            log("Stop file detected — shutting down Sentinel.")
            return True
        return False

    def stop(self, *_args) -> None:
        self._stop = True


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def cmd_status(settings: dict) -> int:
    pid = read_pid()
    running = pid > 0 and _pid_alive(pid)
    print(f"Sentinel : {'RUNNING (pid ' + str(pid) + ')' if running else 'stopped'}")
    print(f"JARVIS   : {'running' if jarvis_is_running() else 'stopped'}")
    print(f"Claps    : {settings['clap_count']}  (sensitivity {settings['clap_sensitivity']})")
    print(f"Phrases  : {', '.join(settings['wake_words'])}")
    try:
        from core.wake import WakePhraseDetector

        det = WakePhraseDetector(phrases=settings["wake_words"])
        print(f"Backend  : {det.backend} ({'exact words' if det.is_exact else 'loudness only'})")
    except Exception as exc:
        print(f"Backend  : unavailable ({exc})")
    print(f"Log file : {LOG_FILE}")
    return 0


def cmd_stop() -> int:
    pid = read_pid()
    if pid <= 0 or not _pid_alive(pid):
        print("Sentinel is not running.")
        clear_pid()
        return 0
    try:
        STOP_FILE.write_text("stop", encoding="utf-8")
    except Exception:
        pass
    deadline = time.time() + 8
    while time.time() < deadline:
        if not _pid_alive(pid):
            print("Sentinel stopped.")
            clear_pid()
            return 0
        time.sleep(0.5)
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        print("Sentinel terminated.")
    except Exception as exc:
        print(f"Could not stop Sentinel: {exc}")
        return 1
    clear_pid()
    return 0


def cmd_install_vosk() -> int:
    """One-time download of the offline speech model for exact phrase matching."""
    try:
        import vosk  # noqa: F401
    except Exception:
        print("Vosk is not installed. Run:  pip install vosk")
        return 1
    try:
        from core.wake import download_vosk_model, find_vosk_model

        existing = find_vosk_model()
        if existing:
            print(f"Vosk model already installed: {existing}")
            return 0
        print("Downloading the offline wake-phrase model (~40 MB, one time)…")
        path = download_vosk_model()
        print(f"Done: {path}")
        print('JARVIS will now match "wake up" exactly instead of any speech.')
        return 0
    except Exception as exc:
        print(f"Download failed: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="JARVIS Sentinel — clap + \"wake up\" auto-launcher."
    )
    parser.add_argument("--status", action="store_true", help="show Sentinel/JARVIS status")
    parser.add_argument("--stop", action="store_true", help="stop the running Sentinel")
    parser.add_argument("--test", action="store_true", help="clap test mode (does not launch JARVIS)")
    parser.add_argument("--force", action="store_true", help="start even if another Sentinel is recorded")
    parser.add_argument("--install-vosk", action="store_true",
                        help="download the offline model for exact wake-phrase matching")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.install_vosk:
        return cmd_install_vosk()
    if args.status:
        return cmd_status(settings)
    if args.stop:
        return cmd_stop()

    if already_running() and not args.force:
        log(f"Sentinel already running (pid {read_pid()}) — nothing to do.")
        return 0

    write_pid()
    sentinel = Sentinel(settings, test_mode=args.test)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, sentinel.stop)
        except Exception:
            pass
    try:
        return sentinel.run()
    finally:
        clear_pid()


if __name__ == "__main__":
    raise SystemExit(main())
