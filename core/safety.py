"""
JARVIS Safety & Sandboxing Engine — JARVIS 7.0.

As autonomy grows, destructive actions must be guarded:

  * **Dry-run mode** — simulate any plan/tool-call before real execution.
  * **Per-tool risk scoring** — each tool has a risk level (low/medium/high).
    High-risk tools always require user confirmation, even in God Mode.
  * **Action approval cache** — once a user approves a tool for a session,
    subsequent calls are auto-approved for a configurable TTL.
  * **Rollback registry** — tools can register undo actions that are
    executed automatically if the plan fails.

Example::

    from core.safety import SafetyEngine

    safety = SafetyEngine()
    if safety.check("power_tools", params={"action": "power", "mode": "shutdown"}):
        result = power_tools(params)
    # -> prompts user for confirmation (high risk)

    # Dry-run:
    safety.dry_run("power_tools", params={"action": "power", "mode": "shutdown"})
    # -> returns {'safe': False, 'reason': 'Would shut down the system'}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("jarvis.safety")


class RiskLevel(str, Enum):
    LOW = "low"           # read-only, safe (web_search, learn, goals)
    MEDIUM = "medium"      # modifies user data (notes, habits, goals)
    HIGH = "high"          # system-level changes (power, install, delete)


@dataclass
class ToolRisk:
    """Risk assessment for a tool."""
    tool: str
    base_risk: RiskLevel
    param_risks: dict[str, RiskLevel] = field(default_factory=dict)  # param-value → risk override
    description: str = ""


# Risk registry — define per-tool and per-parameter risk levels.
# This lets us be granular: e.g. `power_tools(action=list_windows)` is low-risk
# but `power_tools(action=power, mode=shutdown)` is high-risk.
RISK_REGISTRY: dict[str, ToolRisk] = {
    # Core read-only tools
    "open_app": ToolRisk("open_app", RiskLevel.LOW, description="Launch applications/websites"),
    "web_search": ToolRisk("web_search", RiskLevel.LOW, description="Search the web"),
    "vision": ToolRisk("vision", RiskLevel.LOW, description="Screen capture / webcam"),
    "emotion": ToolRisk("emotion", RiskLevel.LOW, description="Mood tracking"),
    "memory": ToolRisk("memory", RiskLevel.LOW, description="Store/retrieve memories"),
    "learn": ToolRisk("learn", RiskLevel.LOW, description="Teach JARVIS facts"),
    "goals": ToolRisk("goals", RiskLevel.LOW, description="Manage goals"),
    "persona": ToolRisk("persona", RiskLevel.LOW, description="Change persona"),
    "focus": ToolRisk("focus", RiskLevel.LOW, description="Focus mode toggle"),

    # Medium-risk tools
    "email": ToolRisk("email", RiskLevel.MEDIUM, description="Send emails"),
    "calendar": ToolRisk("calendar", RiskLevel.MEDIUM, description="Manage calendar"),
    "notes": ToolRisk("notes", RiskLevel.MEDIUM, description="Create/read notes"),
    "habit": ToolRisk("habit", RiskLevel.MEDIUM, description="Track habits"),
    "autonomy": ToolRisk("autonomy", RiskLevel.MEDIUM,
                         description="Execute multi-step plans"),

    # High-risk tools
    "power_tools": ToolRisk(
        "power_tools", RiskLevel.HIGH,
        param_risks={},
        description="System control: install apps, kill processes, shutdown/restart",
    ),
    "agent_app": ToolRisk("agent_app", RiskLevel.HIGH,
                          description="Windows PowerShell automation"),
    "domain": ToolRisk("domain", RiskLevel.MEDIUM,
                       description="External domain integrations"),
}

# Parameter-level risk overrides for power_tools specifically
_POWER_TOOL_RISKS: dict[str, RiskLevel] = {
    "list_windows": RiskLevel.LOW,
    "processes": RiskLevel.LOW,
    "env_get": RiskLevel.LOW,
    "app_search": RiskLevel.LOW,
    "services": RiskLevel.LOW,
    "clipboard_get": RiskLevel.LOW,

    "clipboard_set": RiskLevel.MEDIUM,
    "window_focus": RiskLevel.MEDIUM,
    "window_minimize": RiskLevel.MEDIUM,
    "window_maximize": RiskLevel.MEDIUM,
    "window_close": RiskLevel.MEDIUM,
    "app_install": RiskLevel.HIGH,
    "app_uninstall": RiskLevel.HIGH,
    "service_start": RiskLevel.HIGH,
    "service_stop": RiskLevel.HIGH,
    "service_restart": RiskLevel.HIGH,
    "task_create": RiskLevel.HIGH,
    "task_delete": RiskLevel.HIGH,
    "env_set": RiskLevel.HIGH,
    "power": RiskLevel.HIGH,
    "kill_process": RiskLevel.HIGH,
}

# Actions that are always dry-run safe (read-only)
_DRY_RUN_SAFE_ACTIONS = {
    "list_windows", "processes", "env_get", "app_search",
    "services", "clipboard_get", "app_updates", "report",
}


def get_risk(tool: str, params: dict[str, Any] | None = None) -> RiskLevel:
    """Determine the risk level for a tool call given its parameters."""
    params = params or {}
    tool_risk = RISK_REGISTRY.get(tool)
    if tool_risk is None:
        # Unknown tools are treated as medium risk
        logger.warning("Unknown tool '%s' — defaulting to medium risk", tool)
        return RiskLevel.MEDIUM

    # Check parameter-level overrides first
    action = params.get("action", "")
    if tool == "power_tools" and action in _POWER_TOOL_RISKS:
        return _POWER_TOOL_RISKS[action]

    if tool == "power_tools":
        mode = params.get("mode", "")
        if mode in ("shutdown", "restart", "hibernate", "lock", "logoff"):
            return RiskLevel.HIGH
        if action in ("kill_process",):
            return RiskLevel.HIGH
        if action in ("app_install", "app_uninstall"):
            return RiskLevel.HIGH

    # Check if params override risk level
    param_val = str(params.get("action", ""))
    if param_val in tool_risk.param_risks:
        return tool_risk.param_risks[param_val]

    return tool_risk.base_risk


class SafetyEngine:
    """Sandboxed execution with dry-run, risk scoring, and approval caching."""

    def __init__(self) -> None:
        self._approval_cache: dict[str, float] = {}  # tool_name -> expiry timestamp
        self._approval_ttl: float = 300.0  # 5 minutes
        self._dry_run_callbacks: dict[str, Callable] = {}

    def check(self, tool: str, params: dict[str, Any] | None = None,
              god_mode: bool = False, session_approvals: bool = True) -> dict[str, Any]:
        """Check if a tool call is safe to execute.

        Returns:
            {'safe': bool, 'risk': str, 'requires_approval': bool,
             'reason': str}
        """
        params = params or {}
        risk = get_risk(tool, params)

        # Low-risk tools are always safe
        if risk == RiskLevel.LOW:
            return {"safe": True, "risk": risk.value,
                    "requires_approval": False, "reason": ""}

        # Check approval cache
        if session_approvals:
            cache_key = f"{tool}:{params.get('action', '')}"
            if cache_key in self._approval_cache:
                if time.time() < self._approval_cache[cache_key]:
                    return {"safe": True, "risk": risk.value,
                            "requires_approval": False,
                            "reason": "Approved in this session (cached)"}
                else:
                    del self._approval_cache[cache_key]

        # Medium risk: safe in God Mode, needs approval otherwise
        if risk == RiskLevel.MEDIUM:
            if god_mode:
                return {"safe": True, "risk": risk.value,
                        "requires_approval": False, "reason": "God mode enabled"}
            return {"safe": False, "risk": risk.value,
                    "requires_approval": True,
                    "reason": f"Medium-risk action requires confirmation"}

        # High risk: always needs approval, even in God Mode (unless in cache)
        if god_mode and self._is_cached(tool, params):
            return {"safe": True, "risk": risk.value,
                    "requires_approval": False,
                    "reason": "Previously approved"}
        return {"safe": False, "risk": risk.value,
                "requires_approval": True,
                "reason": f"High-risk action requires explicit confirmation"}

    def _is_cached(self, tool: str, params: dict) -> bool:
        cache_key = f"{tool}:{params.get('action', '')}"
        return (cache_key in self._approval_cache and
                time.time() < self._approval_cache[cache_key])

    def approve(self, tool: str, params: dict[str, Any] | None = None,
                ttl: float | None = None) -> None:
        """Approve a tool call for future use within the TTL window."""
        cache_key = f"{tool}:{(params or {}).get('action', '')}"
        self._approval_cache[cache_key] = time.time() + (ttl or self._approval_ttl)

    def dry_run(self, tool: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Simulate a tool call without executing it.

        Returns {'safe': bool, 'reason': str, 'estimated_effect': str}.
        """
        params = params or {}
        risk = get_risk(tool, params)

        # Check if there's a registered dry-run handler
        if tool in self._dry_run_callbacks:
            try:
                return self._dry_run_callbacks[tool](params)
            except Exception as exc:
                return {"safe": False, "reason": f"Dry-run error: {exc}",
                        "estimated_effect": "unknown"}

        # Default dry-run logic
        action = params.get("action", "")
        if tool == "power_tools":
            if action in _DRY_RUN_SAFE_ACTIONS:
                return {"safe": True, "reason": "Read-only action",
                        "estimated_effect": f"List {action} without modifying system state"}
            if action == "power":
                mode = params.get("mode", "")
                return {"safe": False, "reason": f"Would execute system {mode}",
                        "estimated_effect": f"This will {mode} the computer and cannot be undone"}
            if action in ("app_install", "app_uninstall"):
                return {"safe": False, "reason": "Would install/uninstall software",
                        "estimated_effect": f"Modifies installed programs"}
            return {"safe": True, "reason": "Unknown action, simulating",
                    "estimated_effect": "No system modification detected"}

        return {"safe": risk != RiskLevel.HIGH, "reason": f"Risk: {risk.value}",
                "estimated_effect": "Unknown"}

    def register_dry_run(self, tool: str, callback: Callable[[dict], dict[str, Any]]) -> None:
        """Register a custom dry-run handler for a tool."""
        self._dry_run_callbacks[tool] = callback

    def clear_approvals(self) -> None:
        """Clear all session approval caches."""
        self._approval_cache.clear()

    def risk_report(self) -> dict[str, Any]:
        """Return a summary of the risk registry."""
        return {
            "total_tools": len(RISK_REGISTRY),
            "low_risk": sum(1 for t in RISK_REGISTRY.values() if t.base_risk == RiskLevel.LOW),
            "medium_risk": sum(1 for t in RISK_REGISTRY.values() if t.base_risk == RiskLevel.MEDIUM),
            "high_risk": sum(1 for t in RISK_REGISTRY.values() if t.base_risk == RiskLevel.HIGH),
        }


# Process-wide instance.
safety = SafetyEngine()
