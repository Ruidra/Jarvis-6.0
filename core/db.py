"""
JARVIS Structured Database — JARVIS 7.0.

Replaces ad-hoc JSON files for relational data (goals, habits, memories,
tool_calls, sessions) with a proper SQLite schema. This enables real
analytics — e.g. "show me my mood trend vs sleep" — instead of hand-rolled
aggregation over flat JSON.

Tables:
  * **users**         — per-user preferences / settings.
  * **memories**      — episodic + semantic + procedural memories with
                        embedding, importance score, last_accessed_at.
  * **goals**         — persistent goals with status, due date, category.
  * **habit_logs**    — daily habit completion records.
  * **tool_calls**    — every agent/tool invocation, with success/failure
                        and duration for performance analysis.
  * **sessions**      — conversation sessions for context compression.
  * **reviews**       — nightly self-review results.

Example::

    from core.db import db

    session_id = db.create_session("voice")
    mem_id = db.store_memory(
        "User prefers dark mode in all apps",
        mem_type="semantic",
        importance=0.8,
        embedding=[0.1, 0.2, ...],     # 384-dim vector
    )
    results = db.search_memories("dark mode")
    habit_id = db.add_habit(user_id=1, name="morning run")
    db.log_habit(habit_id, date="2026-08-21", completed=True)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from core.security import get_base_dir

logger = logging.getLogger("jarvis.db")

_DB_PATH = get_base_dir() / "memory" / "jarvis.db"


class Database:
    """Thread-safe SQLite wrapper with WAL mode and schema migration."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.path = Path(db_path) if db_path else _DB_PATH
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.executescript(_SCHEMA)
            self._migrate()
            conn.commit()

    def _migrate(self) -> None:
        """Apply any schema migrations needed."""
        try:
            cols = [row[1] for row in self._connect().execute("PRAGMA table_info(tool_calls)")]
            if "error_message" not in cols:
                self._connect().execute("ALTER TABLE tool_calls ADD COLUMN error_message TEXT")
        except sqlite3.OperationalError:
            pass

    # ── Sessions ───────────────────────────────────────────────────────────────
    def create_session(self, mode: str = "text") -> int:
        conn = self._connect()
        with self._lock:
            cur = conn.execute(
                "INSERT INTO sessions (mode, started_at) VALUES (?, ?)",
                (mode, time.time()),
            )
            return cur.lastrowid

    def end_session(self, session_id: int) -> None:
        with self._lock:
            self._connect().execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (time.time(), session_id),
            )

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        row = self._connect().execute(
            "SELECT id, mode, started_at, ended_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "mode": row[1], "started_at": row[2], "ended_at": row[3]}

    # ── Memories ───────────────────────────────────────────────────────────────
    def store_memory(
        self,
        content: str,
        mem_type: str = "semantic",
        importance: float = 0.5,
        embedding: list[float] | None = None,
        session_id: int | None = None,
        source: str = "",
    ) -> int:
        """Store a memory with optional embedding for similarity search."""
        conn = self._connect()
        emb_json = json.dumps(embedding) if embedding else None
        with self._lock:
            cur = conn.execute(
                """INSERT INTO memories
                   (content, mem_type, importance, embedding, session_id, source, created_at, accessed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (content, mem_type, importance, emb_json, session_id, source,
                 time.time(), time.time()),
            )
            return cur.lastrowid

    def search_memories(self, query_embedding: list[float] | None = None,
                         text_query: str = "",
                         mem_type: str | None = None,
                         top_k: int = 10) -> list[dict[str, Any]]:
        """Search memories by embedding similarity or text match."""
        conn = self._connect()
        results: list[dict[str, Any]] = []

        if query_embedding:
            # Vector similarity search (if FAISS index available, use that;
            # otherwise brute-force cosine in Python)
            from core.vector_memory import vector_db
            ids, scores = vector_db.search(query_embedding, top_k=top_k)

            if ids:
                placeholders = ",".join("?" * len(ids))
                rows = conn.execute(
                    f"""SELECT id, content, mem_type, importance, accessed_at, source, created_at
                          FROM memories WHERE id IN ({placeholders})
                          ORDER BY importance DESC""",
                    list(ids),
                ).fetchall()
                for row, score in zip(rows, scores):
                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "type": row[2],
                        "importance": row[3],
                        "accessed_at": row[4],
                        "source": row[5],
                        "created_at": row[6],
                        "score": float(score),
                    })
                    # Update accessed_at
                    conn.execute(
                        "UPDATE memories SET accessed_at = ? WHERE id = ?",
                        (time.time(), row[0]),
                    )
        elif text_query:
            pattern = f"%{text_query}%"
            query = """SELECT id, content, mem_type, importance, accessed_at, source, created_at
                       FROM memories WHERE content LIKE ?"""
            params: list = [pattern]
            if mem_type:
                query += " AND mem_type = ?"
                params.append(mem_type)
            query += " ORDER BY importance DESC LIMIT ?"
            params.append(top_k)
            rows = conn.execute(query, params).fetchall()
            for row in rows:
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "type": row[2],
                    "importance": row[3],
                    "accessed_at": row[4],
                    "source": row[5],
                    "created_at": row[6],
                })

        # Sort by combined importance + recency
        results.sort(key=lambda r: r.get("score", 0) * r.get("importance", 0.5), reverse=True)
        return results[:top_k]

    def decay_memories(self, half_life_days: float = 30.0) -> int:
        """Decay memory importance over time using exponential decay."""
        now = time.time()
        with self._lock:
            cur = self._connect().execute(
                """UPDATE memories
                   SET importance = importance * power(0.5,
                       (JULIANDAY('now') - JULIANDAY(created_at, 'unixepoch')) / ?)
                   WHERE mem_type != 'procedural'""",
                (half_life_days,),
            )
            return cur.rowcount

    def forget_memory(self, memory_id: int) -> bool:
        with self._lock:
            cur = self._connect().execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def list_memories(self, mem_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._connect()
        if mem_type:
            rows = conn.execute(
                "SELECT id, content, mem_type, importance, source, created_at "
                "FROM memories WHERE mem_type = ? ORDER BY created_at DESC LIMIT ?",
                (mem_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, content, mem_type, importance, source, created_at "
                "FROM memories ORDER BY created_at DESC LIMIT ?",
                limit,
            ).fetchall()
        return [{"id": r[0], "content": r[1], "type": r[2], "importance": r[3],
                 "source": r[4], "created_at": r[5]} for r in rows]

    # ── Goals ──────────────────────────────────────────────────────────────────
    def add_goal(self, text: str, due: str = "", category: str = "goal",
                 user_id: int = 1) -> int:
        with self._lock:
            cur = self._connect().execute(
                """INSERT INTO goals (user_id, text, due, category, created, status)
                   VALUES (?, ?, ?, ?, ?, 'open')""",
                (user_id, text, due, category, time.strftime("%Y-%m-%d")),
            )
            return cur.lastrowid

    def list_goals(self, only_open: bool = True, category: str | None = None) -> list[dict[str, Any]]:
        conn = self._connect()
        query = "SELECT id, text, due, category, created, status FROM goals WHERE user_id = 1"
        params: list = [1]
        if only_open:
            query += " AND status = 'open'"
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY created DESC"
        rows = conn.execute(query, params).fetchall()
        return [{"id": r[0], "text": r[1], "due": r[2], "category": r[3],
                 "created": r[4], "status": r[5]} for r in rows]

    def complete_goal(self, goal_id: int) -> bool:
        with self._lock:
            cur = self._connect().execute(
                "UPDATE goals SET status = 'completed', completed_at = ? WHERE id = ?",
                (time.time(), goal_id),
            )
            return cur.rowcount > 0

    # ── Habits ─────────────────────────────────────────────────────────────────
    def add_habit(self, name: str, user_id: int = 1) -> int:
        with self._lock:
            cur = self._connect().execute(
                "INSERT INTO habits (user_id, name, created_at) VALUES (?, ?, ?)",
                (user_id, name, time.strftime("%Y-%m-%d")),
            )
            return cur.lastrowid

    def list_habits(self, user_id: int = 1) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            "SELECT id, name, created_at FROM habits WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

    def log_habit(self, habit_id: int, date: str = "", completed: bool = True,
                  user_id: int = 1) -> int:
        date = date or time.strftime("%Y-%m-%d")
        with self._lock:
            cur = self._connect().execute(
                """INSERT OR REPLACE INTO habit_logs (habit_id, user_id, date, completed, logged_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (habit_id, user_id, date, completed, time.time()),
            )
            return cur.lastrowid

    def get_habit_streak(self, habit_id: int, user_id: int = 1) -> int:
        """Return the current streak (consecutive days completed)."""
        rows = self._connect().execute(
            """SELECT date, completed FROM habit_logs
               WHERE habit_id = ? AND user_id = ?
               ORDER BY date DESC LIMIT 30""",
            (habit_id, user_id),
        ).fetchall()
        streak = 0
        for date_str, completed in rows:
            if completed:
                streak += 1
            else:
                break
        return streak

    # ── Tool calls ─────────────────────────────────────────────────────────────
    def log_tool_call(self, session_id: int, tool_name: str, args: dict[str, Any],
                       success: bool, duration_ms: float, error: str = "") -> int:
        with self._lock:
            cur = self._connect().execute(
                """INSERT INTO tool_calls
                   (session_id, tool_name, args, success, duration_ms, error_message, called_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, tool_name, json.dumps(args), success,
                 duration_ms, error, time.time()),
            )
            return cur.lastrowid

    def get_tool_stats(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            f"""SELECT tool_name, AVG(duration_ms) as avg_ms,
                       SUM(CASE WHEN success THEN 0 ELSE 1 END) as failures,
                       COUNT(*) as total
                FROM tool_calls GROUP BY tool_name ORDER BY total DESC LIMIT {limit}""",
        ).fetchall()
        return [{"tool": r[0], "avg_ms": round(r[1], 1),
                 "failures": r[2], "total": r[3]} for r in rows]

    # ── Self-review ────────────────────────────────────────────────────────────
    def save_review(self, date: str, review_text: str, suggestions: list[dict] | None = None) -> int:
        with self._lock:
            cur = self._connect().execute(
                """INSERT INTO reviews (date, review_text, suggestions, created_at)
                   VALUES (?, ?, ?, ?)""",
                (date, review_text,
                 json.dumps(suggestions or []), time.time()),
            )
            return cur.lastrowid

    def get_reviews(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            "SELECT date, review_text, suggestions, created_at FROM reviews ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"date": r[0], "review": r[1],
                 "suggestions": json.loads(r[2]) if r[2] else [],
                 "created_at": r[3]} for r in rows]

    # ── Raw query ──────────────────────────────────────────────────────────────
    def query(self, sql: str, params: tuple | None = None) -> list[tuple]:
        with self._lock:
            cur = self._connect().execute(sql, params or ())
            return cur.fetchall()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL DEFAULT 'User',
    preferences   TEXT    DEFAULT '{}',
    created_at    REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY,
    mode          TEXT    NOT NULL,
    started_at    REAL,
    ended_at      REAL
);

CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY,
    content       TEXT NOT NULL,
    mem_type      TEXT    NOT NULL DEFAULT 'semantic',
    importance    REAL    DEFAULT 0.5,
    embedding     TEXT,
    session_id    INTEGER REFERENCES sessions(id),
    source        TEXT,
    created_at    REAL,
    accessed_at   REAL
);

CREATE TABLE IF NOT EXISTS goals (
    id            INTEGER PRIMARY KEY,
    user_id       INTEGER DEFAULT 1,
    text          TEXT NOT NULL,
    due           TEXT,
    category      TEXT DEFAULT 'goal',
    created       TEXT,
    completed_at  REAL,
    status        TEXT DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS habits (
    id            INTEGER PRIMARY KEY,
    user_id       INTEGER DEFAULT 1,
    name          TEXT NOT NULL,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS habit_logs (
    habit_id      INTEGER,
    user_id       INTEGER DEFAULT 1,
    date          TEXT NOT NULL,
    completed     INTEGER DEFAULT 0,
    logged_at     REAL,
    PRIMARY KEY (habit_id, user_id, date)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id            INTEGER PRIMARY KEY,
    session_id    INTEGER,
    tool_name     TEXT NOT NULL,
    args          TEXT,
    success       INTEGER NOT NULL,
    duration_ms   REAL,
    error_message TEXT,
    called_at     REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY,
    date          TEXT NOT NULL,
    review_text   TEXT,
    suggestions   TEXT,
    created_at    REAL
);
"""

# Process-wide instance.
db = Database()
