"""Shared pytest setup: keep tests out of the user's real ``memory/`` directory.

Several modules build their storage paths from ``core.security.get_base_dir()``,
which resolves to the repo root.  Importing them under pytest therefore made the
suite read *and write* live user data — a full run was observed to modify
``memory/learned.json``, ``memory/mood_journal.json`` and
``memory/vectors/chroma.sqlite3``.

Two consequences, both bad:

* The user's mood journal and learned facts were rewritten by a test run.
* Tests became order-dependent.  ``Learner`` shares one chromadb collection
  (``jarvis_learned``) rooted at ``memory/vectors``, and every ``Learner()``
  re-seeds it from whatever ``learned.json`` had accumulated.  Recall is
  top-k, so enough accumulated junk pushes the fact a test just taught out of
  the results -- ``test_learner_teach_and_recall`` failed in a full run while
  passing in isolation.

Fix: point those modules at a throwaway directory.  Config reading is left
alone deliberately.  ``core.security.get_base_dir`` is restored immediately
after the redirected modules are imported, so ``safe_read_config`` still finds
the real ``config/`` and no secrets are copied anywhere.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

# Make the repo root importable regardless of where pytest was invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# One throwaway base dir for the whole session. Fresh per run, so no state
# survives between runs; kept for the whole session so the (slow) chromadb
# client is only built once.
_FAKE_BASE = Path(tempfile.mkdtemp(prefix="jarvis-test-base-")).resolve()
(_FAKE_BASE / "memory").mkdir(parents=True, exist_ok=True)

# Modules that build a path under ``memory/``. Each does
# ``from core.security import get_base_dir``, binding the function object into
# its own namespace at import time -- so they must be imported *while* the
# patch is in place to pick it up.
_REDIRECTED = (
    "core.db",
    "core.emotion_engine",
    "core.goals",
    "core.learning",
    "core.self_improve",
    "core.vector_memory",
)


def _fake_base_dir() -> Path:
    return _FAKE_BASE


def _redirect_data_dirs() -> None:
    import core.security as security

    original = security.get_base_dir
    security.get_base_dir = _fake_base_dir  # type: ignore[assignment]
    try:
        for name in _REDIRECTED:
            # A fresh import binds the patch. An already-imported module (e.g.
            # pulled in as a side effect) keeps a stale binding, so overwrite
            # its attribute directly as well.
            module = importlib.import_module(name)
            if getattr(module, "get_base_dir", None) is original:
                module.get_base_dir = _fake_base_dir  # type: ignore[attr-defined]
    finally:
        # Restore before any test runs: config/ must stay real.
        security.get_base_dir = original  # type: ignore[assignment]


# Runs at conftest import, i.e. before pytest collects (and therefore imports)
# any test module. This ordering matters: core/vector_memory.py builds a
# module-level ``vector_db = VectorMemory()`` singleton at import time, which
# resolves its persist_dir once and keeps it.
_redirect_data_dirs()


def pytest_report_header(config) -> str:
    return f"jarvis: data dirs redirected to {_FAKE_BASE}"
