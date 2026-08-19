"""
actions/network_toolkit.py — JARVIS Network Toolkit.

Cross-platform network diagnostics and WiFi control:
- List saved WiFi profiles (Windows only)
- Connect to WiFi by SSID
- Check internet connectivity
- List active network connections
- Port scan / check port status
- Show network adapter info
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any

_BASE = Path(__file__).resolve().parent.parent


def _platform_os() -> str:
    return {"Windows": "windows", "Darwin": "mac", "Linux": "linux"}.get(
        platform.system(), "linux"
    )


def _run(cmd: list[str] | str, capture: bool = True, timeout: int = 15) -> str:
    """Run a command and return stdout string."""
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
        return out[:2000]
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except FileNotFoundError:
        return f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"
    except Exception as e:
        return f"Error: {e}"


def network_toolkit(parameters: dict, player: Any = None, speak: Any = None) -> str:
    """Main dispatcher for network toolkit operations."""
    action = (parameters.get("action") or "info").lower().strip()

    if action == "info":
        return _adapter_info()
    if action == "connections":
        return _active_connections()
    if action == "check_internet":
        return _check_internet()
    if action == "port_scan":
        return _port_scan(parameters)
    if action == "ping":
        return _ping(parameters)
    if action == "list_wifi":
        return _list_wifi()
    if action == "connect_wifi":
        return _connect_wifi(parameters)
    if action == "wifi_password":
        return _wifi_password(parameters)
    if action == "wifi_status":
        return _wifi_status()
    return f"Unknown network action: {action}. Use: info | connections | check_internet | port_scan | ping | list_wifi | connect_wifi | wifi_password | wifi_status"


def _adapter_info() -> str:
    os_type = _platform_os()
    if os_type == "windows":
        out = _run(["netsh", "interface", "show", "interface"])
        if not out or "Error" in out:
            return _run(["ipconfig", "/all"])
        return out
    elif os_type == "mac":
        return _run(["ifconfig"])
    else:
        return _run(["ip", "addr"]) or _run(["ifconfig"])


def _active_connections() -> str:
    os_type = _platform_os()
    if os_type == "windows":
        return _run(["netstat", "-ano"])
    else:
        return _run(["ss", "-tulpn"]) or _run(["netstat", "-tulpn"])


def _check_internet() -> str:
    """Check if internet is reachable by attempting to resolve and connect."""
    targets = [
        ("8.8.8.8", 53),
        ("1.1.1.1", 53),
    ]
    for host, port in targets:
        try:
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
            return f"Internet is reachable (verified via {host}:{port})."
        except Exception:
            continue
    return "Internet appears unreachable (DNS/connectivity check failed)."


def _ping(parameters: dict) -> str:
    host = (parameters.get("host") or "google.com").strip()
    count = str(parameters.get("count", "4"))
    os_type = _platform_os()
    if os_type == "windows":
        return _run(["ping", "-n", count, host])
    else:
        return _run(["ping", "-c", count, host])


def _port_scan(parameters: dict) -> str:
    host = (parameters.get("host") or "localhost").strip()
    ports_raw = parameters.get("ports", "22,80,443,3000,8000")
    if isinstance(ports_raw, str):
        try:
            ports = [int(p.strip()) for p in ports_raw.split(",") if p.strip()]
        except ValueError:
            return f"Invalid ports format: {ports_raw}"
    elif isinstance(ports_raw, list):
        ports = [int(p) for p in ports_raw]
    else:
        ports = [22, 80, 443, 3000, 8000]

    open_ports = []
    closed_ports = []
    for port in ports[:50]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                open_ports.append(str(port))
            else:
                closed_ports.append(str(port))
        except Exception:
            closed_ports.append(str(port))

    lines = [f"Port scan for {host}:"]
    if open_ports:
        lines.append(f"  Open: {', '.join(open_ports)}")
    if closed_ports:
        lines.append(f"  Closed/Filtered: {', '.join(closed_ports[:20])}")
    return "\n".join(lines)


def _list_wifi() -> str:
    os_type = _platform_os()
    if os_type != "windows":
        if os_type == "mac":
            out = _run(["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-s"])
            if out:
                return out
        return _run(["nmcli", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list"]) or \
               _run(["iwlist", "scan"])

    out = _run(["netsh", "wlan", "show", "profiles"])
    if not out or "not running" in out.lower():
        return "No WiFi profiles found or WiFi is not available."
    profiles = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("All User Profile") or line.startswith("    "):
            if ":" in line:
                profiles.append(line.split(":", 1)[1].strip().strip('"'))
    if not profiles:
        return "No saved WiFi profiles found."
    return "Saved WiFi profiles:\n" + "\n".join(f"  • {p}" for p in profiles)


def _connect_wifi(parameters: dict) -> str:
    ssid = (parameters.get("ssid") or "").strip()
    password = (parameters.get("password") or "").strip()
    if not ssid:
        return "SSID is required to connect to WiFi."

    os_type = _platform_os()
    if os_type == "windows":
        if not password:
            return f"Password required for '{ssid}'. Please provide the WiFi password."
        profile_xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{password}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>"""
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"wifi_{ssid}.xml"
        tmp.write_text(profile_xml, encoding="utf-8")
        _run(["netsh", "wlan", "add", "profile", "filename", str(tmp)])
        r = _run(["netsh", "wlan", "connect", f"name={ssid}"])
        try:
            tmp.unlink()
        except Exception:
            pass
        return r or f"Connecting to '{ssid}'..."

    elif os_type == "mac":
        if not password:
            return f"Password required for '{ssid}'."
        return _run(["networksetup", "-setairportnetwork", "en0", ssid, password])
    else:
        if not password:
            return f"Password required for '{ssid}'."
        _run(["nmcli", "dev", "wifi", "connect", ssid, "password", password])
        return f"Connecting to '{ssid}' via NetworkManager..."


def _wifi_password(parameters: dict) -> str:
    ssid = (parameters.get("ssid") or "").strip()
    if not ssid:
        return "SSID is required to retrieve the WiFi password."
    os_type = _platform_os()
    if os_type != "windows":
        return "WiFi password retrieval is only supported on Windows."
    out = _run(["netsh", "wlan", "show", "profile", f'name="{ssid}"', "key=clear"])
    if not out:
        return f"Could not retrieve password for '{ssid}'."
    for line in out.splitlines():
        line_s = line.strip()
        if line_s.lower().startswith("key content") or line_s.lower().startswith("content"):
            return f"WiFi password for '{ssid}': {line_s.split(':', 1)[1].strip()}"
    return f"Profile '{ssid}' found but password could not be extracted."


def _wifi_status() -> str:
    os_type = _platform_os()
    if os_type == "windows":
        out = _run(["netsh", "wlan", "show", "profiles"])
        if not out or "not running" in out.lower():
            return "WiFi is not available or not running."
        conn = _run(["netsh", "wlan", "show", "interfaces"])
        if conn:
            for line in conn.splitlines():
                line_s = line.strip()
                if line_s.startswith("SSID") or line_s.startswith("State") or line_s.startswith("Signal"):
                    pass
            return conn[:1500]
        return "WiFi adapter found but status could not be determined."
    elif os_type == "mac":
        return _run(["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"])
    else:
        return _run(["iwgetid", "-r"]) or _run(["nmcli", "-t", "-f", "active,ssid,signal", "dev", "wifi"])
