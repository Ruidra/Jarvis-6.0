"""
actions/workflow.py — JARVIS Workflow Automation Engine.

Record, save, replay, and manage desktop workflows. A workflow is a sequence of
actions (mouse, keyboard, app launch, wait) that can be replayed on demand.

Storage: memory/workflows.json
"""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Any

_BASE = Path(__file__).resolve().parent.parent
_WORKFLOW_PATH = _BASE / "memory" / "workflows.json"
_LOCK = threading.Lock()


def _load_workflows() -> dict:
    try:
        return json.loads(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_workflows(data: dict) -> None:
    _WORKFLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WORKFLOW_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _platform_os() -> str:
    import platform
    return {"Windows": "windows", "Darwin": "mac", "Linux": "linux"}.get(
        platform.system(), "linux"
    )


def workflow(parameters: dict, player: Any = None, speak: Any = None) -> str:
    """Main dispatcher for workflow operations."""
    action = (parameters.get("action") or "list").lower().strip()
    name = (parameters.get("name") or "").strip()

    if action == "list":
        return _list_workflows()
    if action == "save":
        return _save_workflow(parameters)
    if action == "delete":
        return _delete_workflow(name)
    if action == "run":
        return _run_workflow(name, parameters)
    if action == "record":
        return _record_workflow(parameters)
    if action == "info":
        return _workflow_info(name)
    return f"Unknown workflow action: {action}. Use: list | save | delete | run | record | info"


def _list_workflows() -> str:
    wfs = _load_workflows()
    if not wfs:
        return "No workflows saved. Say 'record a workflow' to create one."
    lines = [f"Saved workflows ({len(wfs)}):"]
    for k, v in wfs.items():
        steps = len(v.get("steps", []))
        desc = v.get("description", "No description")
        lines.append(f"  • {k}: {steps} steps — {desc}")
    return "\n".join(lines)


def _save_workflow(parameters: dict) -> str:
    name = (parameters.get("name") or "").strip()
    description = (parameters.get("description") or "").strip()
    steps = parameters.get("steps") or []
    if not name:
        return "Workflow name is required to save."
    if not steps:
        return "No steps provided. Pass a 'steps' array to save."

    with _LOCK:
        wfs = _load_workflows()
        wfs[name] = {
            "description": description or f"Workflow '{name}'",
            "steps": steps,
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "count": len(steps),
        }
        _save_workflows(wfs)
    return f"Workflow '{name}' saved with {len(steps)} steps."


def _delete_workflow(name: str) -> str:
    if not name:
        return "Workflow name is required to delete."
    with _LOCK:
        wfs = _load_workflows()
        if name not in wfs:
            return f"Workflow '{name}' not found."
        del wfs[name]
        _save_workflows(wfs)
    return f"Workflow '{name}' deleted."


def _workflow_info(name: str) -> str:
    if not name:
        return "Workflow name is required."
    wfs = _load_workflows()
    if name not in wfs:
        return f"Workflow '{name}' not found."
    wf = wfs[name]
    lines = [
        f"Workflow: {name}",
        f"Description: {wf.get('description', '')}",
        f"Steps: {len(wf.get('steps', []))}",
        f"Created: {wf.get('created', 'unknown')}",
        "Steps:",
    ]
    for i, step in enumerate(wf.get("steps", []), 1):
        stype = step.get("type", "?")
        detail = _format_step_detail(step)
        lines.append(f"  {i}. {stype}: {detail}")
    return "\n".join(lines)


def _format_step_detail(step: dict) -> str:
    stype = step.get("type", "")
    if stype in ("click", "double_click", "right_click"):
        return f"({step.get('x', '?')}, {step.get('y', '?')})"
    if stype == "type":
        return repr(step.get("text", ""))[:40]
    if stype == "hotkey":
        return step.get("keys", "")
    if stype == "open_app":
        return step.get("app_name", "")
    if stype == "wait":
        return f"{step.get('seconds', 1)}s"
    if stype == "scroll":
        return f"{step.get('amount', 3)} clicks {step.get('direction', 'down')}"
    return ", ".join(f"{k}={v}" for k, v in step.items() if k != "type")


def _run_workflow(name: str, parameters: dict) -> str:
    if not name:
        return "Workflow name is required to run."
    wfs = _load_workflows()
    if name not in wfs:
        return f"Workflow '{name}' not found."

    wf = wfs[name]
    steps = wf.get("steps", [])
    if not steps:
        return f"Workflow '{name}' is empty."

    speed = float(parameters.get("speed", 1.0))
    speed = max(0.1, min(speed, 5.0))

    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05 / speed
    except ImportError:
        return "PyAutoGUI not installed. Run: pip install pyautogui"

    results = []
    for i, step in enumerate(steps, 1):
        try:
            _exec_step(step, speed)
            results.append(f"Step {i}: OK")
        except Exception as e:
            results.append(f"Step {i}: FAILED — {e}")
            break

    ok_count = sum(1 for r in results if ": OK" in r)
    return f"Ran '{name}': {ok_count}/{len(steps)} steps completed.\n" + "\n".join(results)


def _exec_step(step: dict, speed: float) -> None:
    import pyautogui
    stype = (step.get("type") or "").lower().strip()
    wait_s = float(step.get("wait", 0)) / speed

    if stype in ("click", "double_click", "right_click"):
        x = int(step.get("x", 0))
        y = int(step.get("y", 0))
        if wait_s:
            time.sleep(wait_s)
        if stype == "double_click":
            pyautogui.doubleClick(x, y)
        elif stype == "right_click":
            pyautogui.rightClick(x, y)
        else:
            pyautogui.click(x, y)

    elif stype == "type":
        text = step.get("text", "")
        if wait_s:
            time.sleep(wait_s)
        pyautogui.typewrite(text, interval=0.02 / speed)

    elif stype == "hotkey":
        keys = step.get("keys", "")
        if wait_s:
            time.sleep(wait_s)
        if "+" in keys:
            parts = [k.strip() for k in keys.split("+")]
            pyautogui.hotkey(*parts)
        else:
            pyautogui.press(keys)

    elif stype == "open_app":
        app = step.get("app_name", "")
        if wait_s:
            time.sleep(wait_s)
        if app:
            _open_app(app)

    elif stype == "wait":
        time.sleep(float(step.get("seconds", 1)) / speed)

    elif stype == "scroll":
        x = int(step.get("x", pyautogui.position().x))
        y = int(step.get("y", pyautogui.position().y))
        amount = int(step.get("amount", 3))
        direction = step.get("direction", "down")
        if wait_s:
            time.sleep(wait_s)
        pyautogui.scroll(-amount if direction == "down" else amount, x=x, y=y)

    elif stype == "move":
        x = int(step.get("x", 0))
        y = int(step.get("y", 0))
        if wait_s:
            time.sleep(wait_s)
        pyautogui.moveTo(x, y, duration=0.1 / speed)

    elif stype == "screenshot":
        if wait_s:
            time.sleep(wait_s)
        from actions.screen_processor import _capture_screen
        img, _ = _capture_screen()
        if img:
            out = _BASE / "logs" / f"workflow_snap_{int(time.time())}.png"
            out.write_bytes(img)

    else:
        raise ValueError(f"Unknown step type: {stype}")


def _open_app(app_name: str) -> None:
    import subprocess
    os_type = _platform_os()
    app = app_name.lower()
    try:
        if os_type == "windows":
            subprocess.Popen(
                ["cmd", "/c", "start", "", app_name],
                creationflags=subprocess.CREATE_NO_WINDOW if os_type == "windows" else 0,
                close_fds=True,
            )
        elif os_type == "mac":
            subprocess.Popen(["open", "-a", app_name], close_fds=True)
        else:
            subprocess.Popen([app_name], close_fds=True)
    except Exception:
        pass


def _record_workflow(parameters: dict) -> str:
    """Start an interactive recording session (returns immediately with instructions)."""
    name = (parameters.get("name") or "").strip()
    description = (parameters.get("description") or "").strip()
    if not name:
        return "Please provide a workflow name. Example: record a workflow called 'morning_routine'"

    try:
        import pyautogui
        pyautogui.FAILSAFE = True
    except ImportError:
        return "PyAutoGUI not installed. Run: pip install pyautogui"

    duration = int(parameters.get("duration", 30))
    duration = max(5, min(duration, 300))

    state = {"recording": False, "steps": [], "start": 0, "name": name, "desc": description}
    _recording_state = state

    def _on_move(x, y):
        if not state["recording"]:
            return
        state["steps"].append({"type": "move", "x": x, "y": y, "ts": time.time()})

    def _on_click(x, y, button, pressed):
        if not state["recording"] or not pressed:
            return
        btn = "right_click" if button.name == "right" else "double_click" if button.name == "middle" else "click"
        state["steps"].append({"type": btn, "x": x, "y": y, "ts": time.time()})

    def _on_key(key):
        if not state["recording"]:
            return
        state["steps"].append({"type": "hotkey", "keys": str(key), "ts": time.time()})

    try:
        from pynput import mouse, keyboard
    except ImportError:
        return "pynput not installed. Run: pip install pynput"

    m_listener = mouse.Listener(on_move=_on_move, on_click=_on_click)
    k_listener = keyboard.Listener(on_press=_on_key)

    state["recording"] = True
    state["start"] = time.time()
    m_listener.start()
    k_listener.start()

    threading.Timer(duration, lambda: _stop_recording(state, m_listener, k_listener)).start()

    return (
        f"Recording workflow '{name}' for {duration} seconds. "
        f"Move the mouse, click, and type now. Recording will stop automatically."
    )


def _stop_recording(state: dict, m_listener, k_listener) -> None:
    state["recording"] = False
    try:
        m_listener.stop()
        k_listener.stop()
    except Exception:
        pass

    raw = state["steps"]
    if not raw:
        return

    steps = _simplify_steps(raw)
    name = state["name"]
    with _LOCK:
        wfs = _load_workflows()
        wfs[name] = {
            "description": state.get("desc") or f"Recorded workflow '{name}'",
            "steps": steps,
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "count": len(steps),
        }
        _save_workflows(wfs)


def _simplify_steps(raw: list[dict]) -> list[dict]:
    """Merge consecutive moves, deduplicate, keep meaningful actions."""
    simplified = []
    last_move: dict | None = None
    last_hotkey: dict | None = None

    for step in raw:
        stype = step.get("type")
        if stype == "move":
            if last_move and step.get("ts", 0) - last_move.get("ts", 0) < 0.1:
                last_move = step
                continue
            last_move = step
            simplified.append(step)
            continue
        last_move = None

        if stype == "hotkey":
            k = step.get("keys", "")
            if last_hotkey and last_hotkey.get("keys") == k and step.get("ts", 0) - last_hotkey.get("ts", 0) < 0.2:
                continue
            last_hotkey = step
            simplified.append(step)
            continue
        last_hotkey = None

        simplified.append(step)

    return simplified
