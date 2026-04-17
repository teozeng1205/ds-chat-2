"""Persistent SQLite tracing processor for the OpenAI Agents SDK.

Captures every trace and span into a local SQLite file so runs can be
inspected after the fact and so the UI can cite a trace_id with each
assistant message.

Usage — at startup, call:

    from .tracing import install_sqlite_tracing
    install_sqlite_tracing()

The processor is registered via `agents.add_trace_processor` and persists
asynchronously from a background thread. The agent runtime is never
blocked by disk I/O.

This module is intentionally self-contained: importing it does NOT
register anything. Call `install_sqlite_tracing()` explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from agents import add_trace_processor
from agents.tracing import Span, Trace, get_current_trace
from agents.tracing.processor_interface import TracingProcessor

log = logging.getLogger(__name__)


DEFAULT_TRACE_DB_ENV = "DS_CHAT_TRACE_DB"
DEFAULT_TRACE_DB_FILENAME = "ds-chat-traces.sqlite"


def default_trace_db_path() -> Path:
    """Resolve the trace SQLite path.

    Priority:
      1. `$DS_CHAT_TRACE_DB` (explicit override)
      2. Co-located with the persistent thread store under `app/.data/`
    """
    env = os.environ.get(DEFAULT_TRACE_DB_ENV)
    if env:
        return Path(env).expanduser().resolve()

    backend_root = Path(__file__).resolve().parents[1]
    data_dir = backend_root / "app" / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (data_dir / DEFAULT_TRACE_DB_FILENAME).resolve()


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS traces (
        trace_id      TEXT PRIMARY KEY,
        name          TEXT,
        started_at    TEXT,
        ended_at      TEXT,
        payload_json  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS spans (
        span_id       TEXT PRIMARY KEY,
        trace_id      TEXT,
        parent_id     TEXT,
        kind          TEXT,
        started_at    TEXT,
        ended_at      TEXT,
        error_json    TEXT,
        payload_json  TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)",
]


def _ensure_schema(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        for stmt in _SCHEMA:
            conn.execute(stmt)
        conn.commit()


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=True, default=str)
    except Exception:
        return json.dumps({"_unserializable": repr(obj)[:400]})


def _span_kind(span: Span[Any]) -> str:
    data = getattr(span, "span_data", None)
    if data is None:
        return "unknown"
    return type(data).__name__


class SQLiteTracingProcessor(TracingProcessor):
    """TracingProcessor that appends traces + spans to a SQLite file.

    Writes happen on a background worker thread from a bounded queue.
    Dropping events under backpressure is preferred over blocking the
    agent; drops are logged.
    """

    def __init__(self, db_path: Path, queue_size: int = 2000) -> None:
        self._db_path = db_path
        self._q: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._drops = 0
        _ensure_schema(db_path)
        self._worker = threading.Thread(
            target=self._run, name="sqlite-trace-worker", daemon=True
        )
        self._worker.start()

    # ── TracingProcessor interface ──

    def on_trace_start(self, trace: Trace) -> None:
        self._enqueue(("trace_start", _trace_snapshot(trace)))

    def on_trace_end(self, trace: Trace) -> None:
        self._enqueue(("trace_end", _trace_snapshot(trace)))

    def on_span_start(self, span: Span[Any]) -> None:
        self._enqueue(("span_start", _span_snapshot(span)))

    def on_span_end(self, span: Span[Any]) -> None:
        self._enqueue(("span_end", _span_snapshot(span)))

    def shutdown(self, timeout: float | None = None) -> None:
        self._stop.set()
        # Drain any pending events with a short join
        self._worker.join(timeout=timeout if timeout is not None else 2.0)

    def force_flush(self) -> None:
        # Best effort: wait for the queue to drain briefly.
        self._q.join()

    # ── Internals ──

    def _enqueue(self, item: tuple[str, Any]) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._drops += 1
            if self._drops % 100 == 1:
                log.warning("SQLiteTracingProcessor queue full; dropped %d events", self._drops)

    def _run(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            while not self._stop.is_set() or not self._q.empty():
                try:
                    kind, payload = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    self._apply(conn, kind, payload)
                except Exception as exc:  # noqa: BLE001
                    log.warning("trace persist failed (%s): %s", kind, exc)
                finally:
                    self._q.task_done()
            conn.commit()
        finally:
            conn.close()

    def _apply(self, conn: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> None:
        if kind in ("trace_start", "trace_end"):
            conn.execute(
                """
                INSERT INTO traces(trace_id, name, started_at, ended_at, payload_json)
                VALUES(:trace_id, :name, :started_at, :ended_at, :payload_json)
                ON CONFLICT(trace_id) DO UPDATE SET
                    name         = COALESCE(excluded.name, traces.name),
                    started_at   = COALESCE(traces.started_at, excluded.started_at),
                    ended_at     = COALESCE(excluded.ended_at, traces.ended_at),
                    payload_json = excluded.payload_json
                """,
                payload,
            )
        else:  # span_start / span_end
            conn.execute(
                """
                INSERT INTO spans(span_id, trace_id, parent_id, kind, started_at, ended_at, error_json, payload_json)
                VALUES(:span_id, :trace_id, :parent_id, :kind, :started_at, :ended_at, :error_json, :payload_json)
                ON CONFLICT(span_id) DO UPDATE SET
                    started_at   = COALESCE(spans.started_at, excluded.started_at),
                    ended_at     = COALESCE(excluded.ended_at, spans.ended_at),
                    error_json   = COALESCE(excluded.error_json, spans.error_json),
                    payload_json = excluded.payload_json
                """,
                payload,
            )
        conn.commit()


def _trace_snapshot(trace: Trace) -> dict[str, Any]:
    exported = {}
    try:
        exported = trace.export() or {}
    except Exception:
        pass
    return {
        "trace_id": getattr(trace, "trace_id", None),
        "name": getattr(trace, "name", None),
        "started_at": exported.get("started_at") if isinstance(exported, dict) else None,
        "ended_at": exported.get("ended_at") if isinstance(exported, dict) else None,
        "payload_json": _json(exported),
    }


def _span_snapshot(span: Span[Any]) -> dict[str, Any]:
    exported = {}
    try:
        exported = span.export() or {}
    except Exception:
        pass
    error = getattr(span, "error", None)
    error_json: Optional[str] = None
    if error is not None:
        try:
            error_json = _json(error if isinstance(error, dict) else {"repr": repr(error)})
        except Exception:
            error_json = _json({"repr": repr(error)[:400]})
    return {
        "span_id": getattr(span, "span_id", None),
        "trace_id": getattr(span, "trace_id", None),
        "parent_id": getattr(span, "parent_id", None),
        "kind": _span_kind(span),
        "started_at": getattr(span, "started_at", None),
        "ended_at": getattr(span, "ended_at", None),
        "error_json": error_json,
        "payload_json": _json(exported),
    }


# ── Installation + small query helpers ──

_INSTALLED: SQLiteTracingProcessor | None = None
_LOCK = threading.Lock()


def install_sqlite_tracing(db_path: Path | None = None) -> SQLiteTracingProcessor:
    """Register the SQLite tracing processor with the agents SDK (idempotent)."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED is not None:
            return _INSTALLED
        resolved = db_path or default_trace_db_path()
        processor = SQLiteTracingProcessor(resolved)
        add_trace_processor(processor)
        _INSTALLED = processor
        log.info("SQLite tracing installed at %s", resolved)
        return processor


def current_trace_id() -> str | None:
    """Return the trace_id of the currently-running trace, if any."""
    try:
        trace = get_current_trace()
    except Exception:
        return None
    return getattr(trace, "trace_id", None) if trace else None
