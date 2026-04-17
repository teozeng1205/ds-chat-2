"""Append-only cost + token telemetry.

Every LLM call records a single row: model, tokens in/out, dollars,
thread_id, trace_id. Aggregation queries power the $/tokens display in
the SessionStateBar, per-thread budget enforcement, and eval/regression
dashboards.

This module is self-contained. Importing it does NOT auto-install
anything. Callers (server.py + tool wrappers) record events via
`record_tokens(...)` and read aggregates via `thread_totals(...)` /
`trace_totals(...)`.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


DEFAULT_COST_DB_ENV = "DS_CHAT_COST_DB"
DEFAULT_COST_DB_FILENAME = "ds-chat-cost.sqlite"


def default_cost_db_path() -> Path:
    env = os.environ.get(DEFAULT_COST_DB_ENV)
    if env:
        return Path(env).expanduser().resolve()
    backend_root = Path(__file__).resolve().parents[1]
    data_dir = backend_root / "app" / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (data_dir / DEFAULT_COST_DB_FILENAME).resolve()


# ── Pricing table ──
#
# US-east OpenAI prices as of Apr 2026, in dollars per 1M tokens.
# Update here when pricing changes.

@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


# NOTE: these are indicative and may drift; keep them close to reality
# because this number is surfaced to users and used for budget caps.
_PRICE_TABLE: dict[str, ModelPrice] = {
    "gpt-5.4":              ModelPrice(input_per_mtok=1.25, output_per_mtok=10.00),
    "gpt-5.4-mini":         ModelPrice(input_per_mtok=0.25, output_per_mtok=2.00),
    # legacy identifiers — map to prior prices so old threads still score
    "gpt-5.2":              ModelPrice(input_per_mtok=1.25, output_per_mtok=10.00),
    "gpt-5-mini":           ModelPrice(input_per_mtok=0.25, output_per_mtok=2.00),
    # embeddings — output_per_mtok is unused but kept for table uniformity
    "text-embedding-3-large": ModelPrice(input_per_mtok=0.13, output_per_mtok=0.00),
    "text-embedding-3-small": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.00),
    "whisper-1":              ModelPrice(input_per_mtok=0.006 * 1000, output_per_mtok=0.00),
    #                          ↑ whisper is priced per-minute; this is a rough tokens-equivalent placeholder
}


def price_for(model: str) -> Optional[ModelPrice]:
    return _PRICE_TABLE.get(model)


def dollars_for(model: str, input_tokens: int, output_tokens: int) -> float:
    p = price_for(model)
    if p is None:
        return 0.0
    return round(
        (input_tokens / 1_000_000.0) * p.input_per_mtok
        + (output_tokens / 1_000_000.0) * p.output_per_mtok,
        6,
    )


# ── Store ──

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS cost_events (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        thread_id      TEXT,
        trace_id       TEXT,
        model          TEXT NOT NULL,
        input_tokens   INTEGER NOT NULL DEFAULT 0,
        output_tokens  INTEGER NOT NULL DEFAULT 0,
        total_tokens   INTEGER NOT NULL DEFAULT 0,
        dollars        REAL NOT NULL DEFAULT 0.0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cost_thread ON cost_events(thread_id)",
    "CREATE INDEX IF NOT EXISTS idx_cost_trace  ON cost_events(trace_id)",
]


class CostStore:
    """Thread-safe append-only cost store.

    Writes are synchronous but cheap (single INSERT). The lock protects
    the connection, which is opened once and reused.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
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
        model: str,
        input_tokens: int,
        output_tokens: int,
        thread_id: str | None = None,
        trace_id: str | None = None,
    ) -> float:
        """Record a single LLM usage event. Returns dollars charged."""
        input_tokens = max(0, int(input_tokens or 0))
        output_tokens = max(0, int(output_tokens or 0))
        total = input_tokens + output_tokens
        dollars = dollars_for(model, input_tokens, output_tokens)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cost_events(thread_id, trace_id, model, input_tokens, output_tokens, total_tokens, dollars)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (thread_id, trace_id, model, input_tokens, output_tokens, total, dollars),
            )
            self._conn.commit()
        return dollars

    def thread_totals(self, thread_id: str) -> dict[str, float | int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens),0)  AS input_tokens,
                    COALESCE(SUM(output_tokens),0) AS output_tokens,
                    COALESCE(SUM(total_tokens),0)  AS total_tokens,
                    COALESCE(SUM(dollars),0.0)     AS dollars,
                    COUNT(*)                       AS events
                FROM cost_events
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
        return {
            "input_tokens": int(row[0]),
            "output_tokens": int(row[1]),
            "total_tokens": int(row[2]),
            "dollars": round(float(row[3]), 6),
            "events": int(row[4]),
        }

    def trace_totals(self, trace_id: str) -> dict[str, float | int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens),0),
                    COALESCE(SUM(output_tokens),0),
                    COALESCE(SUM(total_tokens),0),
                    COALESCE(SUM(dollars),0.0),
                    COUNT(*)
                FROM cost_events
                WHERE trace_id = ?
                """,
                (trace_id,),
            ).fetchone()
        return {
            "input_tokens": int(row[0]),
            "output_tokens": int(row[1]),
            "total_tokens": int(row[2]),
            "dollars": round(float(row[3]), 6),
            "events": int(row[4]),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── Module-level singleton ──

_SINGLETON: CostStore | None = None
_SINGLETON_LOCK = threading.Lock()


def get_cost_store() -> CostStore:
    """Lazily construct the default-path CostStore (idempotent)."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = CostStore(default_cost_db_path())
        return _SINGLETON


def record_tokens(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    thread_id: str | None = None,
    trace_id: str | None = None,
) -> float:
    """Convenience wrapper around `get_cost_store().record(...)`.

    Returns the dollars charged. Safe to call from anywhere; never raises.
    """
    try:
        return get_cost_store().record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thread_id=thread_id,
            trace_id=trace_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("cost record failed: %s", exc)
        return 0.0
