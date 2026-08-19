# Jarvis L — Code Quality Review

Scope: all 51 files / ~16,700 lines of Python. Two modules were fully
refactored as a demonstration (`memory/config_manager.py` and
`memory/memory_manager.py` — same public API, drop-in replacements).
Everything below is a prioritized list of what's worth fixing next, and why.

## What was actually changed

- **`memory/config_manager.py`** — the three setters (`save_api_keys`,
  `save_assistant_config`, `save_brief_enabled`) each reimplemented the same
  read-JSON → merge → write-JSON logic with a `try/except: data = {}`
  swallow. Collapsed into one `_update_config()` helper. Added a module
  logger instead of silent failure, type hints, and docstrings.
- **`memory/memory_manager.py`** — `format_memory_for_prompt` repeated the
  same 5-line "if section: append heading, loop, extract value, title-case
  key" block once per category (identity/preferences/projects/relationships/
  wishes/notes), ~50 lines of copy-pasted logic. Extracted to a
  `_format_section()` helper driven by a small config table. Also
  deduplicated the JSON read/write paths used by `save_memory`,
  `save_session_summary`, and `pop_last_session` into `_read_json`/
  `_write_json`, and replaced `print()` diagnostics with `logging`.

Both were verified against the original behavior with a functional smoke
test (see commands below) before being swapped in.

## Codebase-wide patterns worth fixing

### 1. `print()` instead of `logging` (26 files)
Every diagnostic message is `print(f"[Tag] emoji message")`. This means:
no log levels, no way to silence noisy modules, no way to redirect to a
file for the packaged desktop app, and no timestamps. A drop-in fix:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Mic started")      # instead of print("[JARVIS] 🎤 Mic started")
logger.warning("...")           # instead of print("... ⚠️ ...")
logger.error("...")             # instead of print("... ❌ ...")
```

Configure once in `main.py` with `logging.basicConfig(...)`, and the
`[Tag]` prefixes can become `logger = logging.getLogger("jarvis.vision")`
per module — free filtering by subsystem.

### 2. Broad `except Exception` (353 occurrences)
Almost every I/O call, network call, or JSON parse is wrapped in a bare
`except Exception as e: print(...)`. This is safe against crashes but hides
real bugs (e.g. a `TypeError` from a code change looks identical to a
network timeout in the logs). Where practical, narrow these to the
exceptions actually expected — `(OSError, json.JSONDecodeError)` for file
I/O, `requests.RequestException` for network calls — and let genuinely
unexpected exceptions surface (or log with `logger.exception(...)` so the
traceback isn't lost).

### 3. Duplicated read-modify-write JSON logic
The pattern fixed in `config_manager.py` (load file, mutate one field,
write file, with a repeated try/except around the load) shows up in
several other places, e.g. `actions/reminder.py`,
`actions/background_monitor.py`, and parts of `dashboard/server.py`. Worth
a shared tiny utility (`core/json_store.py`) with `read(path, default)` /
`update(path, **fields)` so every module isn't reinventing file locking.

### 4. Two very large files
- `main.py` — 1,535 lines, one `JarvisLive` class handling session
  lifecycle, mic capture, audio playback, vision, tool dispatch, proactive
  checks, background monitoring, and reconnect/backoff logic.
- `ui.py` — 3,338 lines, the PyQt6 HUD.

Neither is "wrong," but both mix several responsibilities that could be
split without changing behavior — e.g. pulling the reconnect/backoff loop,
the background-monitor poller, and the proactive-check scheduler out of
`JarvisLive` into their own small classes in `actions/`, each taking a
callback into the main session. That would make each piece independently
testable and make `main.py` readable top-to-bottom instead of requiring
you to hold the whole file in your head. Given the size and the live
audio/tool-call surface involved, this is worth doing as its own focused
pass rather than folding into a general cleanup — happy to do it as a
follow-up if useful.

### 5. Magic numbers without named constants
Timeouts, cooldowns, and size limits are scattered as literals (`20-minute
cooldown`, `2200` char memory budget, `380` char value truncation — the
last two are now named constants in `memory_manager.py`). Worth doing the
same in `actions/proactive.py`, `actions/background_monitor.py`, and
`actions/screen_processor.py`, where cooldown/interval values currently
appear inline.

### 6. Inconsistent type hints
Newer-looking files (`core/llm_client.py`, `memory/memory_manager.py`
before this pass) use hints fairly consistently; older ones
(`actions/open_app.py`, `actions/desktop.py`) have almost none. Not
urgent, but adding hints to function signatures in `actions/` would make
the tool-dispatch contract in `main.py` (which calls these by name with a
JSON-decoded `args` dict) much easier to verify at a glance.

## Suggested order of follow-up work

1. Add `logging.basicConfig` in `main.py` and do a search-and-replace of
   `print(f"[Tag] ...")` → `logger.info/warning/error(...)` — mechanical,
   low-risk, high value for anyone debugging a deployed instance.
2. Extract the shared JSON read-modify-write helper and point the
   remaining duplicated call sites at it.
3. Narrow the highest-traffic `except Exception` blocks (network calls in
   `actions/web_search.py`, `actions/flight_finder.py`,
   `core/llm_client.py`) to specific exception types.
4. Split `main.py`'s `JarvisLive` into smaller collaborators, as a
   dedicated pass with test coverage around reconnect/backoff behavior
   before and after.

## How the two refactored files were verified

```python
# both modules were exercised end-to-end against a temp copy of memory/
# before being swapped in, e.g.:
cm.save_assistant_config('Friday', 'Tony')
assert cm.get_assistant_name() == 'Friday'

mm.update_memory({'identity': {'name': {'value': 'Tony'}}})
assert 'Name: Tony' in mm.format_memory_for_prompt(mm.load_memory())
```

No public function signature or return value changed — everything else in
the codebase that imports from these two modules keeps working unmodified.
