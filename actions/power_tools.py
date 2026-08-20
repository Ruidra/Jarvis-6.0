"""
actions/power_tools.py — JARVIS god-tier power tools.

One tool that covers the "why can't JARVIS just do that?" gaps: clipboard,
app install/removal, window management, process control, lightning-fast file
search, service control, scheduled tasks, environment variables, power state and
a quick system report.

Everything is defensive: each action degrades gracefully when an optional
dependency (pywin32, psutil, winget) is missing, and destructive actions
(install, uninstall, kill, shutdown, service control) require God Mode.

Entry point::

    power_tools(parameters={"action": "clipboard_get"})
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.power_tools")

IS_WINDOWS = platform.system() == "Windows"
_CLIP_HISTORY: list[str] = []
_MAX_HISTORY = 25

# Actions that can change or break things → God Mode required.
_DESTRUCTIVE = {
    "app_install", "app_uninstall", "app_upgrade", "kill_process",
    "service_start", "service_stop", "service_restart",
    "task_create", "task_delete", "env_set", "power",
}


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _god_mode() -> bool:
    try:
        from memory.config_manager import get_god_mode

        return bool(get_god_mode())
    except Exception:
        return False


def _run(cmd: list[str] | str, timeout: int = 60, shell: bool = False) -> tuple[int, str]:
    """Run a command and return (exit_code, combined_output)."""
    try:
        kwargs: dict[str, Any] = {
            "capture_output": True, "text": True, "timeout": timeout,
            "errors": "replace",
        }
        if IS_WINDOWS:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(cmd, shell=shell, **kwargs)
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return proc.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, f"Command timed out after {timeout}s."
    except FileNotFoundError:
        exe = cmd if isinstance(cmd, str) else cmd[0]
        return 127, f"'{exe}' is not available on this system."
    except Exception as exc:  # noqa: BLE001
        return 1, f"Command failed: {exc}"


def _trim(text: str, limit: int = 3000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…(+{len(text) - limit} more chars)"


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def _expand(path: str) -> Path:
    shortcuts = {
        "desktop": Path.home() / "Desktop",
        "downloads": Path.home() / "Downloads",
        "documents": Path.home() / "Documents",
        "pictures": Path.home() / "Pictures",
        "videos": Path.home() / "Videos",
        "music": Path.home() / "Music",
        "home": Path.home(),
        "temp": Path(os.environ.get("TEMP", "/tmp")),
    }
    key = (path or "").strip().lower()
    if key in shortcuts:
        return shortcuts[key]
    return Path(os.path.expandvars(os.path.expanduser(path or "."))).resolve()


# ──────────────────────────────────────────────────────────────────────────────
# clipboard
# ──────────────────────────────────────────────────────────────────────────────
def _clipboard_get() -> str:
    try:
        import pyperclip  # type: ignore

        text = pyperclip.paste()
    except Exception:
        if IS_WINDOWS:
            code, out = _run(
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"], timeout=15
            )
            text = out if code == 0 else ""
        else:
            text = ""
    text = text or ""
    if not text.strip():
        return "The clipboard is empty (or holds non-text data)."
    if text.strip() and (not _CLIP_HISTORY or _CLIP_HISTORY[-1] != text):
        _CLIP_HISTORY.append(text)
        del _CLIP_HISTORY[:-_MAX_HISTORY]
    return f"Clipboard ({len(text)} chars):\n{_trim(text)}"


def _clipboard_set(text: str, append: bool = False) -> str:
    if text is None:
        return "Nothing to copy."
    if append:
        current = ""
        try:
            import pyperclip  # type: ignore

            current = pyperclip.paste() or ""
        except Exception:
            pass
        text = (current + "\n" + text) if current else text
    try:
        import pyperclip  # type: ignore

        pyperclip.copy(text)
    except Exception:
        if not IS_WINDOWS:
            return "Clipboard write needs pyperclip: pip install pyperclip"
        # PowerShell fallback via a temp file (handles newlines and quotes).
        tmp = Path(os.environ.get("TEMP", ".")) / "_jarvis_clip.txt"
        tmp.write_text(text, encoding="utf-8")
        safe_path = str(tmp).replace("'", "''")
        code, out = _run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-Content -LiteralPath '{safe_path}' -Raw | Set-Clipboard"], timeout=15,
        )
        tmp.unlink(missing_ok=True)
        if code != 0:
            return f"Could not write the clipboard: {out}"
    _CLIP_HISTORY.append(text)
    del _CLIP_HISTORY[:-_MAX_HISTORY]
    return f"Copied {len(text)} characters to the clipboard."


def _clipboard_history() -> str:
    if not _CLIP_HISTORY:
        return "No clipboard history recorded in this session yet."
    lines = [f"Clipboard history ({len(_CLIP_HISTORY)} item(s), newest last):"]
    for i, item in enumerate(_CLIP_HISTORY, 1):
        preview = item.replace("\n", " ⏎ ")[:110]
        lines.append(f"  {i}. {preview}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# apps (winget)
# ──────────────────────────────────────────────────────────────────────────────
def _winget_available() -> bool:
    return shutil.which("winget") is not None


def _app_search(query: str) -> str:
    if not query:
        return "What app should I search for?"
    if not _winget_available():
        return "winget is not available on this PC (needs Windows 10 1809+ / App Installer)."
    code, out = _run(["winget", "search", query, "--accept-source-agreements"], timeout=90)
    if code != 0 and not out:
        return f"Search failed: {out}"
    return f"winget results for '{query}':\n{_trim(out, 2000)}"


def _app_install(query: str, exact_id: str = "") -> str:
    if not _winget_available():
        return "winget is not available on this PC."
    target = exact_id or query
    if not target:
        return "Which app should I install?"
    cmd = ["winget", "install", "--accept-package-agreements", "--accept-source-agreements",
           "--disable-interactivity"]
    cmd += (["--id", exact_id, "-e"] if exact_id else [target])
    code, out = _run(cmd, timeout=900)
    if code == 0:
        return f"Installed '{target}' successfully."
    return f"Install of '{target}' finished with code {code}:\n{_trim(out, 1200)}"


def _app_uninstall(query: str) -> str:
    if not _winget_available():
        return "winget is not available on this PC."
    if not query:
        return "Which app should I remove?"
    code, out = _run(
        ["winget", "uninstall", query, "--disable-interactivity",
         "--accept-source-agreements"], timeout=600,
    )
    if code == 0:
        return f"Uninstalled '{query}'."
    return f"Uninstall of '{query}' returned code {code}:\n{_trim(out, 1200)}"


def _app_upgrade(query: str = "") -> str:
    if not _winget_available():
        return "winget is not available on this PC."
    if query:
        cmd = ["winget", "upgrade", query, "--accept-package-agreements",
               "--accept-source-agreements", "--disable-interactivity"]
    else:
        cmd = ["winget", "upgrade", "--all", "--accept-package-agreements",
               "--accept-source-agreements", "--disable-interactivity"]
    code, out = _run(cmd, timeout=1800)
    scope = query or "all upgradable apps"
    return f"Upgrade of {scope} finished (code {code}):\n{_trim(out, 1200)}"


def _app_list_upgrades() -> str:
    if not _winget_available():
        return "winget is not available on this PC."
    code, out = _run(["winget", "upgrade", "--accept-source-agreements"], timeout=120)
    return f"Apps with available updates:\n{_trim(out, 2000)}" if out else "Everything is up to date."


# ──────────────────────────────────────────────────────────────────────────────
# windows / processes
# ──────────────────────────────────────────────────────────────────────────────
def _list_windows() -> str:
    titles: list[str] = []
    try:
        import pygetwindow as gw  # type: ignore

        titles = [t for t in gw.getAllTitles() if t and t.strip()]
    except Exception:
        if IS_WINDOWS:
            code, out = _run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-Process | Where-Object {$_.MainWindowTitle} | "
                 "Select-Object -ExpandProperty MainWindowTitle"], timeout=30,
            )
            if code == 0:
                titles = [l.strip() for l in out.splitlines() if l.strip()]
    if not titles:
        return "Could not enumerate open windows (install pygetwindow for best results)."
    lines = [f"Open windows ({len(titles)}):"]
    lines += [f"  • {t[:110]}" for t in titles[:40]]
    return "\n".join(lines)


def _window_action(action: str, title: str) -> str:
    if not title:
        return "Which window? Give me part of its title."
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return "Window control needs pygetwindow: pip install pygetwindow"
    try:
        matches = [w for w in gw.getAllWindows() if title.lower() in (w.title or "").lower()]
        if not matches:
            return f"No open window matches '{title}'."
        win = matches[0]
        if action == "window_focus":
            try:
                win.restore()
            except Exception:
                pass
            win.activate()
            return f"Focused '{win.title[:60]}'."
        if action == "window_minimize":
            win.minimize()
            return f"Minimised '{win.title[:60]}'."
        if action == "window_maximize":
            win.maximize()
            return f"Maximised '{win.title[:60]}'."
        if action == "window_close":
            win.close()
            return f"Closed '{win.title[:60]}'."
        return f"Unknown window action '{action}'."
    except Exception as exc:  # noqa: BLE001
        return f"Window action failed: {exc}"


def _top_processes(count: int = 10, sort: str = "cpu") -> str:
    try:
        import psutil  # type: ignore
    except Exception:
        code, out = _run(["tasklist"], timeout=30)
        return f"Running processes:\n{_trim(out, 1500)}" if code == 0 else "psutil is not installed."
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            cpu = p.cpu_percent(None)
            mem = (p.info.get("memory_info").rss if p.info.get("memory_info") else 0)
            procs.append((p.info["pid"], p.info.get("name") or "?", cpu, mem))
        except Exception:
            continue
    time.sleep(0.35)
    rescored = []
    for pid, name, _cpu, mem in procs:
        try:
            rescored.append((pid, name, psutil.Process(pid).cpu_percent(None), mem))
        except Exception:
            rescored.append((pid, name, 0.0, mem))
    key = 3 if sort.lower().startswith("mem") else 2
    rescored.sort(key=lambda t: t[key], reverse=True)
    lines = [f"Top {count} processes by {'memory' if key == 3 else 'CPU'}:"]
    for pid, name, cpu, mem in rescored[: max(1, count)]:
        lines.append(f"  • {name[:32]:32s} pid {pid:<7} cpu {cpu:5.1f}%  ram {_human_size(mem)}")
    return "\n".join(lines)


def _kill_process(name: str = "", pid: str = "") -> str:
    if pid:
        code, out = _run(
            ["taskkill", "/PID", str(pid), "/F"] if IS_WINDOWS else ["kill", "-9", str(pid)],
            timeout=30,
        )
        return f"Killed pid {pid}." if code == 0 else f"Could not kill pid {pid}: {_trim(out, 300)}"
    if not name:
        return "Which process should I stop? Give a name or PID."
    if IS_WINDOWS:
        exe = name if name.lower().endswith(".exe") else f"{name}.exe"
        code, out = _run(["taskkill", "/IM", exe, "/F"], timeout=30)
    else:
        code, out = _run(["pkill", "-f", name], timeout=30)
    return f"Stopped '{name}'." if code == 0 else f"Could not stop '{name}': {_trim(out, 300)}"


# ──────────────────────────────────────────────────────────────────────────────
# fast file search
# ──────────────────────────────────────────────────────────────────────────────
_SKIP_DIRS = {
    "$recycle.bin", "system volume information", "windows", "node_modules",
    "__pycache__", ".git", "appdata", "programdata",
}


def _find_files(pattern: str, path: str = "home", limit: int = 25,
                extension: str = "", newer_days: int = 0) -> str:
    if not pattern and not extension:
        return "What should I search for? Give a name pattern or an extension."
    root = _expand(path)
    if not root.exists():
        return f"'{root}' does not exist."
    needle = (pattern or "").lower()
    ext = extension.lower().lstrip(".")
    cutoff = time.time() - newer_days * 86400 if newer_days else 0
    hits: list[tuple[float, Path, int]] = []
    scanned = 0
    started = time.time()

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            scanned += 1
            low = fname.lower()
            if needle and needle not in low:
                continue
            if ext and not low.endswith("." + ext):
                continue
            fpath = Path(dirpath) / fname
            try:
                st = fpath.stat()
            except Exception:
                continue
            if cutoff and st.st_mtime < cutoff:
                continue
            hits.append((st.st_mtime, fpath, st.st_size))
            if len(hits) >= limit * 4:
                break
        # Keep it snappy: stop after 8 s or once we clearly have enough.
        if time.time() - started > 8.0 or len(hits) >= limit * 4:
            break

    if not hits:
        return f"No files matching '{pattern or ext}' under {root} ({scanned:,} files scanned)."
    hits.sort(reverse=True)
    lines = [f"Found {len(hits)} match(es) under {root} — newest first:"]
    for mtime, fpath, size in hits[:limit]:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        lines.append(f"  • {fpath}  ({_human_size(size)}, {when})")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# services / scheduled tasks / env / power
# ──────────────────────────────────────────────────────────────────────────────
def _services(query: str = "") -> str:
    if not IS_WINDOWS:
        code, out = _run(["systemctl", "list-units", "--type=service", "--no-pager"], timeout=30)
        return _trim(out, 2000) if code == 0 else "Service listing unavailable."
    cmd = ("Get-Service" + (f" | Where-Object {{$_.Name -like '*{query}*' -or "
                            f"$_.DisplayName -like '*{query}*'}}" if query else "")
           + " | Select-Object Status,Name,DisplayName | Format-Table -AutoSize")
    code, out = _run(["powershell", "-NoProfile", "-Command", cmd], timeout=60)
    return f"Services:\n{_trim(out, 2000)}" if code == 0 else f"Could not list services: {out}"


def _service_control(action: str, name: str) -> str:
    if not name:
        return "Which service?"
    verb = {"service_start": "start", "service_stop": "stop", "service_restart": "restart"}[action]
    if IS_WINDOWS:
        ps = {"start": "Start-Service", "stop": "Stop-Service", "restart": "Restart-Service"}[verb]
        code, out = _run(
            ["powershell", "-NoProfile", "-Command", f"{ps} -Name '{name}' -ErrorAction Stop"],
            timeout=90,
        )
    else:
        code, out = _run(["systemctl", verb, name], timeout=90)
    if code == 0:
        return f"Service '{name}' {verb}ed."
    return f"Could not {verb} '{name}': {_trim(out, 400)} (may need administrator rights)"


def _tasks_list(query: str = "") -> str:
    if not IS_WINDOWS:
        code, out = _run(["crontab", "-l"], timeout=20)
        return _trim(out, 1500) if code == 0 else "No cron entries found."
    code, out = _run(["schtasks", "/query", "/fo", "table"], timeout=60)
    if code != 0:
        return f"Could not list scheduled tasks: {_trim(out, 300)}"
    if query:
        keep = [l for l in out.splitlines() if query.lower() in l.lower()]
        out = "\n".join(keep) or f"No scheduled task matches '{query}'."
    return f"Scheduled tasks:\n{_trim(out, 2000)}"


def _task_create(name: str, command: str, when: str = "") -> str:
    if not IS_WINDOWS:
        return "Scheduled-task creation is implemented for Windows only."
    if not name or not command:
        return "I need a task name and a command to run."
    args = ["schtasks", "/create", "/tn", name, "/tr", command, "/f"]
    when = (when or "").strip()
    if when.lower() in ("logon", "login", "startup", ""):
        args += ["/sc", "onlogon"]
    else:
        args += ["/sc", "daily", "/st", when]
    code, out = _run(args, timeout=60)
    return (f"Scheduled task '{name}' created." if code == 0
            else f"Could not create the task: {_trim(out, 400)}")


def _task_delete(name: str) -> str:
    if not IS_WINDOWS:
        return "Scheduled-task deletion is implemented for Windows only."
    if not name:
        return "Which scheduled task should I delete?"
    code, out = _run(["schtasks", "/delete", "/tn", name, "/f"], timeout=60)
    return (f"Scheduled task '{name}' deleted." if code == 0
            else f"Could not delete the task: {_trim(out, 400)}")


def _env_get(name: str = "") -> str:
    if name:
        val = os.environ.get(name)
        return f"{name} = {val}" if val is not None else f"'{name}' is not set."
    keys = sorted(k for k in os.environ if not k.startswith("_"))
    lines = ["Environment variables:"]
    for k in keys[:60]:
        lines.append(f"  {k} = {str(os.environ.get(k, ''))[:90]}")
    return "\n".join(lines)


def _env_set(name: str, value: str, permanent: bool = True) -> str:
    if not name:
        return "Which variable should I set?"
    os.environ[name] = value or ""
    if permanent and IS_WINDOWS:
        code, out = _run(["setx", name, value or ""], timeout=30)
        if code != 0:
            return f"Set for this session, but permanent write failed: {_trim(out, 200)}"
        return f"{name} set to '{value}' (permanent — new terminals will see it)."
    return f"{name} set to '{value}' for the current session."


def _power(mode: str, delay: int = 0) -> str:
    mode = (mode or "").strip().lower()
    if not IS_WINDOWS:
        cmds = {"shutdown": ["shutdown", "-h", "now"], "restart": ["reboot"],
                "lock": ["loginctl", "lock-session"]}
        if mode not in cmds:
            return f"Unsupported power mode '{mode}' on this OS."
        code, out = _run(cmds[mode], timeout=20)
        return f"{mode} requested." if code == 0 else f"Failed: {_trim(out, 200)}"
    mapping = {
        "shutdown": ["shutdown", "/s", "/t", str(max(0, delay))],
        "restart": ["shutdown", "/r", "/t", str(max(0, delay))],
        "reboot": ["shutdown", "/r", "/t", str(max(0, delay))],
        "logoff": ["shutdown", "/l"],
        "cancel": ["shutdown", "/a"],
        "lock": ["rundll32.exe", "user32.dll,LockWorkStation"],
        "sleep": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
        "hibernate": ["shutdown", "/h"],
    }
    if mode not in mapping:
        return f"Unknown power mode '{mode}'. Use shutdown|restart|sleep|hibernate|lock|logoff|cancel."
    code, out = _run(mapping[mode], timeout=30)
    if code != 0:
        return f"Power command failed: {_trim(out, 300)}"
    when = f" in {delay}s" if delay and mode in ("shutdown", "restart", "reboot") else ""
    return f"{mode.capitalize()} requested{when}. Say 'cancel shutdown' to abort."


# ──────────────────────────────────────────────────────────────────────────────
# system report
# ──────────────────────────────────────────────────────────────────────────────
def _system_report() -> str:
    lines = [f"System: {platform.system()} {platform.release()} ({platform.machine()})",
             f"Host  : {platform.node()}",
             f"Python: {platform.python_version()}"]
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        boot = time.time() - psutil.boot_time()
        lines += [
            f"CPU   : {psutil.cpu_percent(interval=0.3)}% over {psutil.cpu_count(logical=True)} threads",
            f"RAM   : {_human_size(vm.used)} / {_human_size(vm.total)} ({vm.percent}%)",
            f"Uptime: {int(boot // 3600)}h {int((boot % 3600) // 60)}m",
        ]
        for part in psutil.disk_partitions(all=False)[:4]:
            try:
                u = psutil.disk_usage(part.mountpoint)
                lines.append(
                    f"Disk  : {part.mountpoint} {_human_size(u.used)}/{_human_size(u.total)} ({u.percent}%)"
                )
            except Exception:
                continue
        try:
            net = psutil.net_io_counters()
            lines.append(f"Net   : up {_human_size(net.bytes_sent)}  down {_human_size(net.bytes_recv)}")
        except Exception:
            pass
    except Exception:
        lines.append("(install psutil for CPU/RAM/disk details)")
    try:
        from core.brain import available_backends, stats

        b = available_backends()
        s = stats()
        lines.append(
            f"Brain : local={'up' if b['local'] else 'down'} "
            f"gemini={'ready' if b['gemini'] else 'missing'} "
            f"calls={s['calls']} cache_hits={s['cache_hits']}"
        )
    except Exception:
        pass
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────────
_HELP = (
    "power_tools actions:\n"
    "  clipboard_get | clipboard_set(text) | clipboard_append(text) | clipboard_history\n"
    "  app_search(query) | app_install(query|app_id) | app_uninstall(query) | "
    "app_upgrade(query?) | app_updates\n"
    "  list_windows | window_focus(title) | window_minimize(title) | "
    "window_maximize(title) | window_close(title)\n"
    "  processes(count,sort) | kill_process(name|pid)\n"
    "  find_files(pattern,path,extension,newer_days,limit)\n"
    "  services(query) | service_start/stop/restart(name)\n"
    "  tasks(query) | task_create(name,command,when) | task_delete(name)\n"
    "  env_get(name) | env_set(name,value) | power(mode,delay) | report"
)


def power_tools(parameters: dict | None = None, player: Any = None, **_kw) -> str:
    """Dispatch a power-tool action. Returns a human-readable result string."""
    p = dict(parameters or {})
    action = (p.get("action") or "").strip().lower()
    if not action or action in ("help", "info", "list"):
        return _HELP

    if action in _DESTRUCTIVE and not _god_mode():
        return (
            f"'{action}' can change or break things, so it needs God Mode. "
            "Say 'enable god mode' first, then ask me again."
        )

    text = p.get("text") if p.get("text") is not None else p.get("value", "")
    name = (p.get("name") or "").strip()
    query = (p.get("query") or p.get("app") or name or "").strip()
    title = (p.get("title") or p.get("window") or query).strip()

    try:
        if action == "clipboard_get":
            return _clipboard_get()
        if action in ("clipboard_set", "clipboard_copy"):
            return _clipboard_set(str(text or ""))
        if action == "clipboard_append":
            return _clipboard_set(str(text or ""), append=True)
        if action == "clipboard_history":
            return _clipboard_history()

        if action == "app_search":
            return _app_search(query)
        if action == "app_install":
            return _app_install(query, (p.get("app_id") or "").strip())
        if action == "app_uninstall":
            return _app_uninstall(query)
        if action == "app_upgrade":
            return _app_upgrade(query)
        if action in ("app_updates", "app_list_upgrades"):
            return _app_list_upgrades()

        if action in ("list_windows", "windows"):
            return _list_windows()
        if action in ("window_focus", "window_minimize", "window_maximize", "window_close"):
            return _window_action(action, title)

        if action in ("processes", "top"):
            return _top_processes(int(p.get("count") or 10), str(p.get("sort") or "cpu"))
        if action == "kill_process":
            return _kill_process(name or query, str(p.get("pid") or ""))

        if action in ("find_files", "search_files", "find"):
            return _find_files(
                pattern=str(p.get("pattern") or query or ""),
                path=str(p.get("path") or "home"),
                limit=int(p.get("limit") or 25),
                extension=str(p.get("extension") or ""),
                newer_days=int(p.get("newer_days") or 0),
            )

        if action == "services":
            return _services(query)
        if action in ("service_start", "service_stop", "service_restart"):
            return _service_control(action, name or query)

        if action in ("tasks", "tasks_list", "scheduled_tasks"):
            return _tasks_list(query)
        if action == "task_create":
            return _task_create(name, str(p.get("command") or ""), str(p.get("when") or ""))
        if action == "task_delete":
            return _task_delete(name or query)

        if action in ("env_get", "env"):
            return _env_get(name)
        if action == "env_set":
            return _env_set(name, str(p.get("value") or text or ""), bool(p.get("permanent", True)))

        if action == "power":
            return _power(str(p.get("mode") or query), int(p.get("delay") or 0))

        if action in ("report", "system_report", "status"):
            return _system_report()

        return f"Unknown action '{action}'.\n\n{_HELP}"
    except Exception as exc:  # noqa: BLE001 — a tool must never crash the assistant
        logger.exception("power_tools failed")
        return f"power_tools '{action}' failed: {exc}"
