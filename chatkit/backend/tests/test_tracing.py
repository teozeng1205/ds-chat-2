"""Unit tests for app.tracing.SQLiteTracingProcessor.

We don't spin up a full agent run here; we call the TracingProcessor
methods directly with simple stand-in trace/span objects and confirm
rows land in the SQLite file. That's enough to validate the contract
with the SDK.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

from app.tracing import SQLiteTracingProcessor


class _FakeTrace(SimpleNamespace):
    def export(self):
        return {"name": self.name, "started_at": self.started_at, "ended_at": self.ended_at}


class _FakeSpanData(SimpleNamespace):
    pass


class _FakeSpan(SimpleNamespace):
    def export(self):
        return {"kind": type(self.span_data).__name__, "started_at": self.started_at, "ended_at": self.ended_at}


def _drain(processor: SQLiteTracingProcessor, timeout: float = 2.0) -> None:
    """Poll until the queue drains (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline and not processor._q.empty():
        time.sleep(0.05)


def test_processor_persists_trace_and_span(tmp_path: Path) -> None:
    db = tmp_path / "traces.sqlite"
    p = SQLiteTracingProcessor(db)

    trace = _FakeTrace(trace_id="tr-1", name="test", started_at="t0", ended_at=None)
    p.on_trace_start(trace)

    span_data = _FakeSpanData(foo="bar")
    span = _FakeSpan(
        span_id="sp-1",
        trace_id="tr-1",
        parent_id=None,
        span_data=span_data,
        started_at="s0",
        ended_at=None,
        error=None,
    )
    p.on_span_start(span)

    span.ended_at = "s1"
    p.on_span_end(span)

    trace.ended_at = "t1"
    p.on_trace_end(trace)

    _drain(p)
    p.shutdown(timeout=2.0)

    with sqlite3.connect(db) as conn:
        traces = conn.execute("SELECT trace_id, name, started_at, ended_at FROM traces").fetchall()
        spans = conn.execute(
            "SELECT span_id, trace_id, parent_id, kind, started_at, ended_at FROM spans"
        ).fetchall()

    assert traces == [("tr-1", "test", "t0", "t1")]
    assert spans == [("sp-1", "tr-1", None, "_FakeSpanData", "s0", "s1")]


def test_processor_upsert_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "traces.sqlite"
    p = SQLiteTracingProcessor(db)

    trace = _FakeTrace(trace_id="tr-x", name="t", started_at="t0", ended_at=None)
    p.on_trace_start(trace)
    p.on_trace_start(trace)  # duplicate start
    trace.ended_at = "t1"
    p.on_trace_end(trace)

    _drain(p)
    p.shutdown(timeout=2.0)

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM traces").fetchone()
    assert rows[0] == 1
