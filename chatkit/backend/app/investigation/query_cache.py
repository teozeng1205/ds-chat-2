"""Query-result cache for read-only SQL.

Keyed by a stable hash of (sql, workgroup, extra tags). TTL-expires at
15 minutes by default. Payloads are stored as JSON blobs in SQLite;
small result dicts (the shape execute_sql already returns) fit
comfortably.

Wired into `execute_sql` in app/tools/investigation_tools.py — a cache
hit returns the cached preview with `cached: True` and no dataset_id
(datasets are per-thread-ephemeral).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)


DEFAULT_CACHE_DB_ENV = "DS_CHAT_QUERY_CACHE_DB"
DEFAULT_CACHE_DB_FILENAME = "ds-chat-query-cache.sqlite"
DEFAULT_TTL_SECONDS = 900  # 15 minutes


def default_cache_db_path() -> Path:
    env = os.environ.get(DEFAULT_CACHE_DB_ENV)
    if env:
        return Path(env).expanduser().resolve()
    backend_root = Path(__file__).resolve().parents[2]
    data_dir = backend_root / "app" / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (data_dir / DEFAULT_CACHE_DB_FILENAME).resolve()


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS query_cache (
        key            TEXT PRIMARY KEY,
        sql            TEXT NOT NULL,
        workgroup      TEXT,
        created_at     REAL NOT NULL,
        expires_at     REAL NOT NULL,
        payload_json   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_query_cache_expires ON query_cache(expires_at)",
]


def _normalize_sql(sql: str) -> str:
    """Normalize whitespace so cosmetically-different queries share a key."""
    return " ".join(sql.split()).strip().lower()


def make_key(sql: str, workgroup: str | None, extra: Iterable[str] | None = None) -> str:
    """Stable hash key for (sql, workgroup, extra tags)."""
    parts = [_normalize_sql(sql), (workgroup or "").strip().lower()]
    if extra:
        parts.extend(sorted(str(x) for x in extra))
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:40]


@dataclass(frozen=True)
class CacheHit:
    payload: dict[str, Any]
    created_at: float
    expires_at: float
    key: str

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.created_at)


class QueryCache:
    """TTL-bounded SQL result cache. Thread-safe; small payloads only.

    Callers decide whether to consult this cache — we don't hook into
    `execute_sql` automatically so existing behavior is untouched until
    the follow-up commit wires it in.
    """

    def __init__(self, db_path: Path, default_ttl_s: int = DEFAULT_TTL_SECONDS) -> None:
        self._db_path = db_path
        self._ttl = int(default_ttl_s)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()

    def get(
        self,
        sql: str,
        workgroup: str | None = None,
        extra: Iterable[str] | None = None,
        *,
        now: float | None = None,
    ) -> Optional[CacheHit]:
        key = make_key(sql, workgroup, extra)
        moment = now if now is not None else time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json, created_at, expires_at FROM query_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        payload_json, created_at, expires_at = row
        if expires_at <= moment:
            # Expired — leave it for purge_expired() to clean up.
            return None
        try:
            payload = json.loads(payload_json)
        except Exception:
            return None
        return CacheHit(payload=payload, created_at=float(created_at), expires_at=float(expires_at), key=key)

    def put(
        self,
        sql: str,
        payload: dict[str, Any],
        workgroup: str | None = None,
        extra: Iterable[str] | None = None,
        *,
        ttl_s: int | None = None,
        now: float | None = None,
    ) -> str:
        key = make_key(sql, workgroup, extra)
        moment = now if now is not None else time.time()
        ttl = int(ttl_s if ttl_s is not None else self._ttl)
        expires = moment + ttl
        serialized = json.dumps(payload, ensure_ascii=True, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO query_cache(key, sql, workgroup, created_at, expires_at, payload_json)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    created_at   = excluded.created_at,
                    expires_at   = excluded.expires_at,
                    payload_json = excluded.payload_json
                """,
                (key, sql, workgroup, moment, expires, serialized),
            )
            self._conn.commit()
        return key

    def invalidate(self, sql: str, workgroup: str | None = None, extra: Iterable[str] | None = None) -> bool:
        key = make_key(sql, workgroup, extra)
        with self._lock:
            cur = self._conn.execute("DELETE FROM query_cache WHERE key = ?", (key,))
            self._conn.commit()
        return cur.rowcount > 0

    def purge_expired(self, *, now: float | None = None) -> int:
        moment = now if now is not None else time.time()
        with self._lock:
            cur = self._conn.execute("DELETE FROM query_cache WHERE expires_at <= ?", (moment,))
            self._conn.commit()
        return cur.rowcount

    def stats(self) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()
        return {"entries": int(row[0])}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── Module-level singleton ──

_SINGLETON: QueryCache | None = None
_SINGLETON_LOCK = threading.Lock()


def get_query_cache() -> QueryCache:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = QueryCache(default_cache_db_path())
        return _SINGLETON
