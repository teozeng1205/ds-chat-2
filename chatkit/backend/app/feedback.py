"""Per-message feedback store.

SQLite-backed append-only log of thumbs-up/thumbs-down on assistant
messages, with an optional free-text comment. Powers future evals
(signal for eval rubric tuning) and weekly review dashboards.

POST /chatkit/feedback (main.py) is the write side. Read-side
summaries live here (`recent`, `summary_by_thread`).

Verdict is normalized to +1 (thumbs up) or -1 (thumbs down); any
other value is rejected.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


DEFAULT_FEEDBACK_DB_ENV = "DS_CHAT_FEEDBACK_DB"
DEFAULT_FEEDBACK_DB_FILENAME = "ds-chat-feedback.sqlite"
MAX_COMMENT_LEN = 2000


def default_feedback_db_path() -> Path:
    env = os.environ.get(DEFAULT_FEEDBACK_DB_ENV)
    if env:
        return Path(env).expanduser().resolve()
    backend_root = Path(__file__).resolve().parents[1]
    data_dir = backend_root / "app" / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (data_dir / DEFAULT_FEEDBACK_DB_FILENAME).resolve()


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        thread_id   TEXT NOT NULL,
        message_id  TEXT,
        verdict     INTEGER NOT NULL CHECK (verdict IN (-1, 1)),
        comment     TEXT,
        user_id     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_feedback_thread ON feedback(thread_id)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at)",
]


@dataclass(frozen=True)
class FeedbackEntry:
    id: int
    created_at: str
    thread_id: str
    message_id: Optional[str]
    verdict: int
    comment: Optional[str]
    user_id: Optional[str]


class FeedbackStore:
    """Thread-safe append-only SQLite feedback log."""

    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()

    def record(
        self,
        *,
        thread_id: str,
        verdict: int,
        message_id: str | None = None,
        comment: str | None = None,
        user_id: str | None = None,
    ) -> int:
        if verdict not in (-1, 1):
            raise ValueError("verdict must be -1 or +1")
        if not thread_id:
            raise ValueError("thread_id is required")
        clipped = (comment or "")[:MAX_COMMENT_LEN]
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO feedback(thread_id, message_id, verdict, comment, user_id)
                VALUES(?, ?, ?, ?, ?)
                """,
                (thread_id, message_id, int(verdict), clipped or None, user_id),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def recent(self, limit: int = 50) -> list[FeedbackEntry]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, created_at, thread_id, message_id, verdict, comment, user_id
                FROM feedback
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [FeedbackEntry(*r) for r in rows]

    def summary_by_thread(self, thread_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*)                                 AS total,
                    COALESCE(SUM(CASE WHEN verdict = 1 THEN 1 ELSE 0 END),0) AS up,
                    COALESCE(SUM(CASE WHEN verdict = -1 THEN 1 ELSE 0 END),0) AS down
                FROM feedback WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
        total, up, down = int(row[0]), int(row[1]), int(row[2])
        return {"total": total, "up": up, "down": down}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── Singleton ──

_SINGLETON: FeedbackStore | None = None
_SINGLETON_LOCK = threading.Lock()


def get_feedback_store() -> FeedbackStore:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = FeedbackStore(default_feedback_db_path())
        return _SINGLETON


__all__ = ["FeedbackStore", "FeedbackEntry", "get_feedback_store", "default_feedback_db_path"]
