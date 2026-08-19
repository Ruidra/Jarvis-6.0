"""
Jarvis Plugin — Utilities (offline power tools).

Fully offline helpers: strong password generation, a safe arithmetic
calculator, and common unit conversions (length / weight / temperature).

Args:
  action  : password | calc | convert   (default: calc)
  length  : password length (default 16)
  symbols : include symbols in password (default true)
  expr    : arithmetic expression for calc, e.g. "(2+3)*4"
  value   : number to convert
  from/to : units, e.g. from="km" to="mi", from="c" to="f"
"""

from __future__ import annotations

import ast
import logging
import operator
import secrets
import string

logger = logging.getLogger("jarvis.plugin.utils")

PLUGIN = {
    "name": "utils",
    "description": (
        "Offline utilities: generate a strong password, evaluate a math expression, "
        "or convert units (km↔mi, kg↔lb, C↔F, and more). Use for 'generate a password', "
        "'calculate (2+3)*4', 'convert 10 km to miles'."
    ),
    "triggers": ["password", "calculate", "convert", "how many", "unit"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action":  {"type": "STRING", "description": "password | calc | convert (default: calc)"},
            "length":  {"type": "INTEGER", "description": "Password length (default 16)."},
            "symbols": {"type": "BOOLEAN", "description": "Include symbols in password (default true)."},
            "expr":    {"type": "STRING", "description": "Arithmetic expression for calc, e.g. '(2+3)*4'."},
            "value":   {"type": "NUMBER", "description": "Value to convert."},
            "from":    {"type": "STRING", "description": "Source unit, e.g. 'km', 'kg', 'c'."},
            "to":      {"type": "STRING", "description": "Target unit, e.g. 'mi', 'lb', 'f'."},
        },
        "required": [],
    },
}

# to-SI factor maps for linear units (length->meters, mass->kg)
_LENGTH = {"m": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "mi": 1609.344,
           "yd": 0.9144, "ft": 0.3048, "in": 0.0254}
_MASS = {"kg": 1, "g": 0.001, "mg": 1e-6, "lb": 0.45359237, "oz": 0.0283495}

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numbers allowed")
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")
    return _ev(ast.parse(expr, mode="eval"))


def _password(length: int, symbols: bool) -> str:
    length = max(4, min(int(length or 16), 64))
    pool = string.ascii_letters + string.digits
    if symbols:
        pool += "!@#$%^&*()-_=+[]{};:,.<>?"
    return "".join(secrets.choice(pool) for _ in range(length))


def _convert(value: float, frm: str, to: str) -> float | None:
    frm, to = frm.lower().strip(), to.lower().strip()
    if frm in _LENGTH and to in _LENGTH:
        return value * _LENGTH[frm] / _LENGTH[to]
    if frm in _MASS and to in _MASS:
        return value * _MASS[frm] / _MASS[to]
    # temperature
    if {frm, to} <= {"c", "f", "k"}:
        if frm == "c" and to == "f":
            return value * 9 / 5 + 32
        if frm == "f" and to == "c":
            return (value - 32) * 5 / 9
        if frm == "c" and to == "k":
            return value + 273.15
        if frm == "k" and to == "c":
            return value - 273.15
        if frm == "f" and to == "k":
            return (value - 32) * 5 / 9 + 273.15
        if frm == "k" and to == "f":
            return (value - 273.15) * 9 / 5 + 32
    return None


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    action = (args.get("action") or "calc").lower().strip()

    if action == "password":
        try:
            pw = _password(args.get("length", 16), bool(args.get("symbols", True)))
        except Exception as e:  # noqa: BLE001
            return f"Couldn't generate a password: {e}"
        return f"🔐 Generated password: {pw}\n(Store it in a password manager — I won't remember it.)"

    if action == "convert":
        try:
            val = float(args.get("value", 0))
            res = _convert(val, args.get("from", ""), args.get("to", ""))
        except Exception:
            return "I couldn't perform that conversion. Check the units (e.g. 10 km to mi)."
        if res is None:
            return ("I don't support that unit pair yet. Try length (km/mi), "
                    "mass (kg/lb), or temperature (C/F/K).")
        return f"📐 {val:g} {args.get('from')} = {res:g} {args.get('to')}"

    # default: calc
    expr = (args.get("expr") or "").strip()
    if not expr:
        return "What should I calculate? e.g. 'calculate (2+3)*4'."
    try:
        result = _safe_eval(expr)
    except Exception as e:  # noqa: BLE001
        return f"I couldn't evaluate '{expr}': {e}"
    return f"🧮 {expr} = {result:g}"
