"""Per-user / per-thread persistent KV memory.

Lets the agent stash small facts that outlive a single turn:
  - scope="user"    default customer, team, preferred tables, custom
                    instructions. Persists across threads.
  - scope="thread"  scratch pad for the current investigation. Scoped
                    to one conversation.

No auth yet, so `user` scope uses a shared "default" identifier until
the auth module lands. That swap is one-line (`_user_id(ctx)`).

Usage pattern:
    from app.memory import get_memory_store
    store = get_memory_store()
    store.put(scope="user", scope_id=_user_id, key="team", value="B6")
    val = store.get(scope="user", scope_id=_user_id, key="team")
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


Scope = Literal["user", "thread"]

DEFAULT_MEMORY_DB_ENV = "DS_CHAT_MEMORY_DB"
DEFAULT_MEMORY_DB_FILENAME = "ds-chat-memory.sqlite"

# Used as the scope_id for `scope="user"` calls until we have auth.
DEFAULT_USER_ID = "default"


def default_memory_db_path() -> Path:
    env = os.environ.get(DEFAULT_MEMORY_DB_ENV)
    if env:
        return Path(env).expanduser().resolve()
    backend_root = Path(__file__).resolve().parents[1]
    data_dir = backend_root / "app" / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (data_dir / DEFAULT_MEMORY_DB_FILENAME).resolve()


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS memory (
        scope       TEXT NOT NULL,
        scope_id    TEXT NOT NULL,
        key         TEXT NOT NULL,
        value       TEXT NOT NULL,
        updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        PRIMARY KEY (scope, scope_id, key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory(scope, scope_id)",
]


class MemoryStore:
    """Thread-safe SQLite-backed KV store."""

    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()

    def put(self, *, scope: Scope, scope_id: str, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory(scope, scope_id, key, value, updated_at)
                VALUES(?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                ON CONFLICT(scope, scope_id, key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                (scope, scope_id, key, value),
            )
            self._conn.commit()

    def get(self, *, scope: Scope, scope_id: str, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM memory WHERE scope = ? AND scope_id = ? AND key = ?",
                (scope, scope_id, key),
            ).fetchone()
        return row[0] if row else None

    def list(self, *, scope: Scope, scope_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, value, updated_at
                FROM memory
                WHERE scope = ? AND scope_id = ?
                ORDER BY key
                """,
                (scope, scope_id),
            ).fetchall()
        return [{"key": r[0], "value": r[1], "updated_at": r[2]} for r in rows]

    def delete(self, *, scope: Scope, scope_id: str, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memory WHERE scope = ? AND scope_id = ? AND key = ?",
                (scope, scope_id, key),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── Process-wide singleton ──

_SINGLETON: MemoryStore | None = None
_SINGLETON_LOCK = threading.Lock()


def get_memory_store() -> MemoryStore:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = MemoryStore(default_memory_db_path())
        return _SINGLETON


__all__ = ["MemoryStore", "Scope", "DEFAULT_USER_ID", "get_memory_store", "default_memory_db_path"]
