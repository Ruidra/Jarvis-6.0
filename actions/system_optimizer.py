"""
actions/system_optimizer.py — JARVIS System Optimizer.

Cross-platform system maintenance and optimization:
- Clean temporary files and caches
- List and manage startup programs
- Show disk usage and largest files
- Empty recycle bin / trash
- Kill runaway processes
- Show installed software
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

_BASE = Path(__file__).resolve().parent.parent


def _platform_os() -> str:
    return {"Windows": "windows", "Darwin": "mac", "Linux": "linux"}.get(
        platform.system(), "linux"
    )


def _run(cmd: list[str] | str, capture: bool = True, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if _platform_os() == "windows" else 0,
            close_fds=True,
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if err and not out:
            return f"[stderr] {err[:500]}"
        return out[:3000]
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except FileNotFoundError:
        return f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"
    except Exception as e:
        return f"Error: {e}"


def _safe_rmtree(path: Path, max_size_mb: float = 500) -> tuple[int, int]:
    """Remove a directory tree, skipping files larger than max_size_mb."""
    freed = 0
    errors = 0
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                fp = Path(root) / f
                try:
                    size = fp.stat().st_size
                    if size > max_size_mb * 1024 * 1024:
                        continue
                    fp.unlink()
                    freed += size
                except Exception:
                    errors += 1
            try:
                Path(root).rmdir()
            except Exception:
                errors += 1
    except Exception:
        errors += 1
    return freed, errors


def system_optimizer(parameters: dict, player: Any = None, speak: Any = None) -> str:
    """Main dispatcher for system optimizer operations."""
    action = (parameters.get("action") or "info").lower().strip()

    if action == "clean_temp":
        return _clean_temp()
    if action == "clean_cache":
        return _clean_cache()
    if action == "disk_usage":
        return _disk_usage(parameters)
    if action == "largest_files":
        return _largest_files(parameters)
    if action == "startup_list":
        return _startup_list()
    if action == "startup_disable":
        return _startup_disable(parameters)
    if action == "processes":
        return _list_processes(parameters)
    if action == "kill_process":
        return _kill_process(parameters)
    if action == "empty_trash":
        return _empty_trash()
    if action == "installed_software":
        return _installed_software()
    if action == "info":
        return _system_info()
    return f"Unknown optimizer action: {action}. Use: clean_temp | clean_cache | disk_usage | largest_files | startup_list | startup_disable | processes | kill_process | empty_trash | installed_software | info"


def _system_info() -> str:
    os_type = _platform_os()
    lines = [f"OS: {platform.system()} {platform.release()} ({platform.machine()})"]
    try:
        import psutil
        lines.append(f"CPU: {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads")
        lines.append(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
        lines.append(f"RAM Used: {psutil.virtual_memory().percent}%")
        lines.append(f"Disk: {psutil.disk_usage('/').total / (1024**3):.1f} GB total")
    except ImportError:
        lines.append("psutil not installed — limited info available")
    return "\n".join(lines)


def _clean_temp() -> str:
    os_type = _platform_os()
    temp_dirs = []
    if os_type == "windows":
        temp_dirs = [
            Path(os.environ.get("TEMP", "")),
            Path(os.environ.get("TMP", "")),
            Path.home() / "AppData" / "Local" / "Temp",
        ]
    elif os_type == "mac":
        temp_dirs = [Path("/tmp"), Path.home() / "Library" / "Caches"]
    else:
        temp_dirs = [Path("/tmp"), Path("/var/tmp")]

    total_freed = 0
    total_errors = 0
    for d in temp_dirs:
        if not d.exists() or not d.is_dir():
            continue
        freed, errors = _safe_rmtree(d, max_size_mb=100)
        total_freed += freed
        total_errors += errors

    freed_mb = total_freed / (1024 * 1024)
    return f"Cleaned temp files: {freed_mb:.1f} MB freed, {total_errors} errors (files skipped/locked)."


def _clean_cache() -> str:
    os_type = _platform_os()
    cache_dirs = []
    if os_type == "windows":
        cache_dirs = [
            Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "INetCache",
            Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data" / "Default" / "Cache",
            Path.home() / "AppData" / "Local" / "Microsoft" / "Edge" / "User Data" / "Default" / "Cache",
        ]
    elif os_type == "mac":
        cache_dirs = [
            Path.home() / "Library" / "Caches",
            Path.home() / "Library" / "Caches" / "com.apple.Safari",
        ]
    else:
        cache_dirs = [
            Path.home() / ".cache",
            Path("/var/cache"),
        ]

    total_freed = 0
    total_errors = 0
    for d in cache_dirs:
        if not d.exists() or not d.is_dir():
            continue
        freed, errors = _safe_rmtree(d, max_size_mb=200)
        total_freed += freed
        total_errors += errors

    freed_mb = total_freed / (1024 * 1024)
    return f"Cleaned caches: {freed_mb:.1f} MB freed, {total_errors} errors."


def _disk_usage(parameters: dict) -> str:
    path = (parameters.get("path") or str(Path.home())).strip()
    try:
        usage = shutil.disk_usage(path)
        total = usage.total / (1024 ** 3)
        used = usage.used / (1024 ** 3)
        free = usage.free / (1024 ** 3)
        pct = (usage.used / usage.total) * 100
        return f"Disk usage for {path}:\n  Total: {total:.1f} GB\n  Used: {used:.1f} GB ({pct:.1f}%)\n  Free: {free:.1f} GB"
    except Exception as e:
        return f"Could not get disk usage: {e}"


def _largest_files(parameters: dict) -> str:
    path = (parameters.get("path") or str(Path.home())).strip()
    count = int(parameters.get("count", 10))
    count = max(1, min(count, 50))

    try:
        files = []
        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                try:
                    fp = Path(root) / f
                    size = fp.stat().st_size
                    files.append((size, fp))
                except Exception:
                    pass
        files.sort(key=lambda x: x[0], reverse=True)
        top = files[:count]
        lines = [f"Largest files in {path}:"]
        for size, fp in top:
            size_mb = size / (1024 * 1024)
            lines.append(f"  {size_mb:.1f} MB — {fp}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error scanning files: {e}"


def _startup_list() -> str:
    os_type = _platform_os()
    if os_type == "windows":
        out = _run(["wmic", "startup", "get", "Caption,Command,Location", "/format:list"])
        if not out or "No Instance" in out:
            return "No startup items found or WMIC returned no results."
        return out[:3000]
    elif os_type == "mac":
        out = _run(["osascript", "-e", 'tell application "System Events" to get the name of every login item'])
        if out:
            return "Login items:\n" + "\n".join(f"  • {l}" for l in out.splitlines() if l.strip())
        return "No login items found."
    else:
        out = _run(["systemctl", "list-unit-files", "--type=service", "--state=enabled"])
        if out:
            return "Enabled system services:\n" + out[:3000]
        return _run(["ls", "/etc/init.d/"]) or "Could not list startup items."


def _startup_disable(parameters: dict) -> str:
    name = (parameters.get("name") or "").strip()
    if not name:
        return "Startup item name is required to disable."

    os_type = _platform_os()
    if os_type != "windows":
        return "Startup management via voice is currently Windows-only."

    out = _run(["wmic", "startup", "where", f"Caption='{name}'", "delete"])
    if out and "successfully" in out.lower():
        return f"Startup item '{name}' disabled."
    out2 = _run(["reg", "delete", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run", "/v", name, "/f"])
    if out2 and "successfully" in out2.lower():
        return f"Startup item '{name}' removed from registry."
    return f"Could not disable startup item '{name}'. Try manual removal in Task Manager."


def _list_processes(parameters: dict) -> str:
    sort_by = (parameters.get("sort") or "cpu").lower().strip()
    count = int(parameters.get("count", 10))
    count = max(1, min(count, 30))

    try:
        import psutil
        procs = []
        for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
            try:
                info = p.info
                mem_mb = info.get('memory_info', type('obj', (0,), {'rss': 0})).rss / (1024 * 1024)
                procs.append({
                    'pid': info['pid'],
                    'name': info['name'],
                    'cpu': info.get('cpu_percent', 0) or 0,
                    'mem_mb': mem_mb,
                })
            except Exception:
                pass

        if sort_by == "memory":
            procs.sort(key=lambda x: x['mem_mb'], reverse=True)
        else:
            procs.sort(key=lambda x: x['cpu'], reverse=True)

        lines = [f"Top {count} processes by {sort_by.upper()}:"]
        for p in procs[:count]:
            lines.append(f"  PID {p['pid']:>6} — {p['name']:<25} CPU: {p['cpu']:>5.1f}%  MEM: {p['mem_mb']:>6.1f} MB")
        return "\n".join(lines)
    except ImportError:
        if _platform_os() == "windows":
            out = _run(["tasklist", "/fo", "csv", "/nh"])
            if out:
                lines = ["Running processes (top by memory):"]
                for line in out.splitlines()[:count]:
                    parts = line.strip('"').split('","')
                    if len(parts) >= 3:
                        lines.append(f"  {parts[0]:<30}  PID: {parts[1]}  MEM: {parts[2]}")
                return "\n".join(lines)
        return "psutil not installed. Run: pip install psutil"


def _kill_process(parameters: dict) -> str:
    name_or_pid = (parameters.get("name") or parameters.get("pid") or "").strip()
    if not name_or_pid:
        return "Process name or PID is required."

    try:
        pid = int(name_or_pid)
        import psutil
        p = psutil.Process(pid)
        p_name = p.name()
        p.terminate()
        return f"Terminated process: {p_name} (PID {pid})."
    except ValueError:
        pass

    try:
        import psutil
        killed = []
        for p in psutil.process_iter(['pid', 'name']):
            try:
                if name_or_pid.lower() in p.info['name'].lower():
                    p.terminate()
                    killed.append(f"{p.info['name']} (PID {p.info['pid']})")
            except Exception:
                pass
        if killed:
            return f"Terminated {len(killed)} process(es): " + ", ".join(killed)
        return f"No running process matching '{name_or_pid}' found."
    except ImportError:
        if _platform_os() == "windows":
            out = _run(["taskkill", "/F", "/IM", name_or_pid])
            return out or f"Attempted to kill '{name_or_pid}'."
        return "psutil not installed. Run: pip install psutil"


def _empty_trash() -> str:
    os_type = _platform_os()
    try:
        if os_type == "windows":
            out = _run(["powershell", "-Command", "Clear-RecycleBin -Force -ErrorAction SilentlyContinue"])
            if out:
                return f"Recycle Bin emptied.\n{out[:500]}"
            return "Recycle Bin emptied."
        elif os_type == "mac":
            out = _run(["osascript", "-e", 'tell application "Finder" to empty trash'])
            return out or "Trash emptied."
        else:
            out = _run(["rm", "-rf", str(Path.home() / ".local" / "share" / "Trash" / "files" / "*")])
            return out or "Trash emptied."
    except Exception as e:
        return f"Error emptying trash: {e}"


def _installed_software() -> str:
    os_type = _platform_os()
    if os_type == "windows":
        out = _run(["wmic", "product", "get", "Name,Version,Vendor", "/format:csv"])
        if out:
            lines = ["Installed software (partial list):"]
            for line in out.splitlines()[2:]:
                parts = line.strip().split(",")
                if len(parts) >= 3 and parts[1].strip():
                    lines.append(f"  {parts[1].strip()[:50]}  v{parts[2].strip()[:20]}  [{parts[0].strip()}]")
                if len(lines) > 30:
                    break
            return "\n".join(lines)
        return _run(["reg", "query", r"HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Uninstall", "/s", "/f", "*"]) or "Could not list installed software."
    elif os_type == "mac":
        return _run(["/usr/sbin/pkgutil", "--pkgs"]) or "Could not list installed packages."
    else:
        if shutil.which("dpkg"):
            return _run(["dpkg", "--get-selections"])[:3000]
        elif shutil.which("rpm"):
            return _run(["rpm", "-qa"])[:3000]
        return "Could not determine package manager."
