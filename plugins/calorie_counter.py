"""
Jarvis Plugin — Calorie Counter & Food Logger.

A focused nutrition tracker for logging food, tracking daily calorie/macronutrient
intake against a goal, and reviewing history. Uses a built-in food database of
common items (expandable by the user) plus the ability to add custom foods.

Triggers (spoken): "log food", "track calories", "what did I eat", "calorie count",
"my macros", "food log".

State lives in ``memory/nutrition.json`` (atomic, via core.json_store).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date
from pathlib import Path

from core.json_store import JsonStore, read_json, atomic_write_json

logger = logging.getLogger("jarvis.plugin.nutrition")

# ── Built-in food database (name → {calories, protein, carbs, fat, serving}) ──
# Values are per 100g unless otherwise noted. Easily extendable via "add food".
_FOOD_DB: dict[str, dict] = {
    "rice":           {"cal": 130, "p": 2.7, "c": 28, "f": 0.3, "serving": "1 cup cooked (~150g)"},
    "chicken breast": {"cal": 165, "p": 31, "c": 0, "f": 3.6, "serving": "100g"},
    "egg":            {"cal": 70, "p": 6, "c": 0.6, "f": 5, "serving": "1 large"},
    "banana":         {"cal": 89, "p": 1.1, "c": 23, "f": 0.3, "serving": "100g"},
    "apple":          {"cal": 52, "p": 0.3, "c": 14, "f": 0.2, "serving": "100g"},
    "bread":          {"cal": 265, "p": 9, "c": 49, "f": 3.2, "serving": "1 slice (~50g)"},
    "pasta":          {"cal": 131, "p": 5, "c": 25, "f": 1.1, "serving": "100g cooked"},
    "salmon":         {"cal": 206, "p": 22, "c": 0, "f": 13, "serving": "100g"},
    "beef":           {"cal": 250, "p": 26, "c": 0, "f": 15, "serving": "100g"},
    "broccoli":       {"cal": 34, "p": 2.8, "c": 7, "f": 0.4, "serving": "100g"},
    "milk":           {"cal": 42, "p": 3.4, "c": 5, "f": 1, "serving": "100ml"},
    "protein powder": {"cal": 400, "p": 80, "c": 10, "f": 5, "serving": "30g scoop"},
    "olive oil":      {"cal": 884, "p": 0, "c": 0, "f": 100, "serving": "1 tbsp (~14g)"},
    "avocado":        {"cal": 160, "p": 2, "c": 9, "f": 15, "serving": "100g"},
    "nuts":           {"cal": 576, "p": 20, "c": 21, "f": 49, "serving": "100g"},
    "water":          {"cal": 0, "p": 0, "c": 0, "f": 0, "serving": "any amount"},
    "coffee":         {"cal": 2, "p": 0.3, "c": 0.5, "f": 0, "serving": "1 cup (250ml)"},
    "yogurt":         {"cal": 59, "p": 3.5, "c": 5, "f": 0.4, "serving": "100g"},
    "oatmeal":        {"cal": 68, "p": 2.4, "c": 12, "f": 1.1, "serving": "100g"},
    "pizza":          {"cal": 266, "p": 11, "c": 33, "f": 10, "serving": "100g"},
    "burger":         {"cal": 250, "p": 18, "c": 29, "f": 9, "serving": "1 patty"},
    "chips":          {"cal": 536, "p": 6, "c": 53, "f": 34, "serving": "100g"},
    "soda":           {"cal": 42, "p": 0, "c": 10, "f": 0, "serving": "1 can (330ml)"},
    "salad":          {"cal": 15, "p": 1, "c": 3, "f": 0.2, "serving": "100g lettuce"},
    "tomato":         {"cal": 18, "p": 0.9, "c": 3.9, "f": 0.2, "serving": "100g"},
    "cheese":          {"cal": 402, "p": 25, "c": 1.3, "f": 33, "serving": "100g"},
    "peanut butter":   {"cal": 588, "p": 25, "c": 20, "f": 50, "serving": "100g"},
    "almonds":         {"cal": 579, "p": 21, "c": 21, "f": 49, "serving": "100g"},
}

# Common serving multipliers (from base 100g values)
_SERVING_MULTIPLIERS = {
    "g": 0.01, "gram": 0.01, "grams": 0.01,
    "kg": 1.0,
    "mg": 0.00001,
    "oz": 0.02835, "ounce": 0.02835, "ounces": 0.02835,
    "lb": 0.4536, "pound": 0.4536, "pounds": 0.4536,
    "cup": 0.25, "cups": 0.25,
    "tbsp": 1.0, "tablespoon": 1.0, "tablespoons": 1.0,
    "tsp": 0.33, "teaspoon": 0.33, "teaspoons": 0.33,
    "slice": 0.2, "slices": 0.2,
    "scoop": 1.0, "scoops": 1.0,
    "can": 3.3, "cans": 3.3,
    "piece": 1.0, "pieces": 1.0, "pcs": 1.0,
    "": 1.0,
}

PLUGIN = {
    "name": "calorie_counter",
    "description": (
        "Food and calorie tracker. Log what you eat, track daily calories and "
        "macros against a goal, and review your food history. "
        "Use when the user says 'log food', 'track calories', 'I ate', 'what did I "
        "eat today', 'my calorie count', 'set a goal', or 'show my food log'."
    ),
    "triggers": [
        "log food", "track calories", "calorie count", "calorie counter",
        "i ate", "food log", "my macros", "what did i eat", "nutrition",
    ],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action":    {"type": "STRING",
                           "description": "log | delete | today | summary | goal | add_food | list (default: log)"},
            "food":      {"type": "STRING", "description": "Food item name (for log / add_food)."},
            "quantity":  {"type": "STRING", "description": "Amount e.g. '2 cups', '150g', '1 apple'."},
            "calories":  {"type": "INTEGER", "description": "Override calories (for add_food)."},
            "meal":      {"type": "STRING",
                           "description": "Meal: breakfast | lunch | dinner | snack (default: snack)."},
            "goal":      {"type": "INTEGER",
                           "description": "Daily calorie goal e.g. 2000 (for goal action)."},
            "id":        {"type": "STRING", "description": "Entry id (for delete action)."},
            "date":      {"type": "STRING", "description": "Date YYYY-MM-DD (defaults to today)."},
        },
        "required": [],
    },
}


def _store() -> JsonStore:
    base = Path(__file__).resolve().parent.parent
    return JsonStore(base / "memory" / "nutrition.json")


def _today() -> str:
    return date.today().isoformat()


def _parse_quantity(qty: str | None) -> float:
    """Parse '2 cups', '150g', '1 apple' → grams (or multiplier of base entry)."""
    if not qty:
        return 1.0
    qty = qty.lower().strip()
    # Look for a number
    m = re.match(r"(\d+(?:\.\d+)?)", qty)
    if not m:
        return 1.0
    num = float(m.group(1))
    rest = qty[m.end():].strip()
    # Check for unit keywords
    for unit, mult in _SERVING_MULTIPLIERS.items():
        if unit and unit in rest:
            if unit in ("g", "gram", "grams", "kg", "mg", "oz", "ounce", "ounces",
                        "lb", "pound", "pounds"):
                # Direct gram conversion
                if unit in ("g", "gram", "grams"):
                    return num
                elif unit in ("kg",):
                    return num * 1000
                elif unit in ("mg",):
                    return num / 1000
                elif unit in ("oz", "ounce", "ounces"):
                    return num * 28.35
                elif unit in ("lb", "pound", "pounds"):
                    return num * 453.6
            else:
                # Use multiplier relative to 100g base
                return num * mult * 100
    # No unit found — assume 1 serving of the food
    return num


def _lookup_food(name: str) -> dict | None:
    """Find a food in the DB (case-insensitive, fuzzy match)."""
    name = name.lower().strip()
    if name in _FOOD_DB:
        return _FOOD_DB[name]
    # Fuzzy: check if any DB entry is contained in the query or vice versa
    for key, val in _FOOD_DB.items():
        if key in name or name in key:
            return val
    return None


def _add_custom_food(name: str, calories: int, serving: str = "1 serving") -> str:
    """Add a custom food to the in-memory database (persists via user config)."""
    if not name or not calories:
        return "Give me a food name and calorie count, e.g. 'add food: mango, 60 kcal'."
    _db = _load_custom_db()
    _db[name.lower().strip()] = {
        "cal": calories, "p": 0, "c": 0, "f": 0, "serving": serving,
    }
    _save_custom_db(_db)
    return f"✅ Added '{name}' ({calories} kcal per serving) to your food database."


def _db_path() -> Path:
    return _store().path.parent / "food_db.json"


def _load_custom_db() -> dict:
    """Load user-added foods merged with the built-in DB."""
    base = dict(_FOOD_DB)
    try:
        if _db_path().exists():
            import json
            custom = json.loads(_db_path().read_text(encoding="utf-8"))
            if isinstance(custom, dict):
                base.update(custom)
    except Exception:
        pass
    return base


def _save_custom_db(db: dict) -> None:
    try:
        _db_path().parent.mkdir(parents=True, exist_ok=True)
        import json
        _db_path().write_text(
            json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("food db save failed: %s", e)


def _calc_macros(food_entry: dict, grams: float) -> dict:
    """Scale macronutrients to the logged amount."""
    scale = grams / 100.0
    return {
        "cal": round(food_entry.get("cal", 0) * scale),
        "p": round(food_entry.get("p", 0) * scale, 1),
        "c": round(food_entry.get("c", 0) * scale, 1),
        "f": round(food_entry.get("f", 0) * scale, 1),
    }


def _log_food(food: str, quantity: str, meal: str, user_date: str | None) -> str:
    """Log a food entry."""
    db = _load_custom_db()
    food_entry = _lookup_food(food)
    if not food_entry:
        return (f"I don't know '{food}' yet. Tell me its calories: "
                f"'add food: {food}, 100 kcal'.")

    grams = _parse_quantity(quantity)

    # If quantity has no unit, assume it's a count of the serving size
    macros = _calc_macros(food_entry, grams)

    store = _store()
    state = read_json(store.path, {}) or {}
    state.setdefault("entries", [])
    today = user_date or _today()

    entry_id = f"{len(state['entries']) + 1:04d}"
    state["entries"].append({
        "id": entry_id,
        "date": today,
        "food": food,
        "quantity": quantity or f"{grams:.0f}g",
        "grams": round(grams, 1),
        "calories": macros["cal"],
        "protein": macros["p"],
        "carbs": macros["c"],
        "fat": macros["f"],
        "meal": meal,
        "ts": time.time(),
    })
    atomic_write_json(store.path, state)

    return (f"✅ Logged: {quantity or f'{grams:.0f}g'} of {food} "
            f"({macros['cal']} kcal) to {meal}.")


def _today_summary(target_date: str | None) -> str:
    """Show today's or a specific date's food log + calorie summary."""
    store = _store()
    state = read_json(store.path, {}) or {}
    entries = state.get("entries", [])
    day = target_date or _today()

    day_entries = [e for e in entries if e.get("date") == day]
    if not day_entries:
        d = day if day != _today() else "today"
        return f"Nothing logged {d}."

    total_cal = sum(e.get("calories", 0) for e in day_entries)
    total_p = sum(e.get("protein", 0) for e in day_entries)
    total_c = sum(e.get("carbs", 0) for e in day_entries)
    total_f = sum(e.get("fat", 0) for e in day_entries)

    goal = state.get("daily_goal", 2000)
    remaining = goal - total_cal

    lines = [f"🍽️ Food log for {day}:"]
    for e in day_entries:
        lines.append(
            f"  • [{e['id']}] {e['meal']} — {e.get('quantity', '?')}"
            f" {e.get('food', '')}: {e.get('calories', 0)} kcal"
        )
    lines.append("")
    lines.append(f"Total: {total_cal} kcal")
    lines.append(f"  Protein: {total_p}g  Carbs: {total_c}g  Fat: {total_f}g")
    if remaining > 0:
        lines.append(f"  Goal: {goal} kcal — {remaining} kcal remaining")
    else:
        lines.append(f"  Goal: {goal} kcal — {abs(remaining)} kcal over!")
    return "\n".join(lines)


def _delete(entry_id: str) -> str:
    if not entry_id:
        return "Which entry id should I delete? Use 'my food log' to see ids."
    store = _store()
    state = read_json(store.path, {}) or {}
    entries = state.get("entries", [])
    for i, e in enumerate(entries):
        if e.get("id") == entry_id:
            food = entries.pop(i).get("food", "")
            atomic_write_json(store.path, state)
            return f"🗑️ Deleted '{food}' (id {entry_id})."
    return f"I couldn't find entry id {entry_id}."


def _list_foods() -> str:
    db = _load_custom_db()
    lines = ["🍎 Your food database:"]
    for name in sorted(db.keys()):
        info = db[name]
        lines.append(f"  • {name}: {info.get('cal', 0)} kcal — {info.get('serving', '')}")
    return "\n".join(lines)


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    action = (args.get("action") or "log").lower().strip()
    user = (ctx.get("user_name") or "sir").title()

    if action == "add_food":
        return _add_custom_food(
            args.get("food", ""),
            args.get("calories") or 0,
            args.get("quantity", "1 serving"),
        )

    if action == "goal":
        goal = args.get("goal")
        if goal:
            store = _store()
            state = read_json(store.path, {}) or {}
            state["daily_goal"] = int(goal)
            atomic_write_json(store.path, state)
            return f"✅ Daily calorie goal set to {goal} kcal."
        # Show current goal
        store = _store()
        state = read_json(store.path, {}) or {}
        current = state.get("daily_goal", 2000)
        return f"Your current daily goal is {current} kcal. Tell me a new number to change it."

    if action == "today":
        return _today_summary(args.get("date"))

    if action == "summary":
        return _today_summary(args.get("date"))

    if action == "delete":
        return _delete((args.get("id") or "").strip())

    if action == "list":
        return _list_foods()

    # default: log food
    food = (args.get("food") or "").strip()
    quantity = (args.get("quantity") or "").strip()
    meal = (args.get("meal") or "snack").strip().lower()
    user_date = args.get("date") or None

    if not food:
        return (f"What did you eat, {user}? e.g. 'I ate 200g chicken breast for lunch'.")

    return _log_food(food, quantity, meal, user_date)
