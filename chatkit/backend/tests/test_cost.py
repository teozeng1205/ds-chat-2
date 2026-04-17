"""Unit tests for app.cost."""

from __future__ import annotations

from pathlib import Path

from app.cost import CostStore, dollars_for, price_for


def test_price_table_covers_current_and_legacy_models() -> None:
    assert price_for("gpt-5.4") is not None
    assert price_for("gpt-5.4-mini") is not None
    assert price_for("gpt-5.2") is not None  # legacy
    assert price_for("gpt-5-mini") is not None  # legacy
    assert price_for("text-embedding-3-large") is not None
    assert price_for("totally-not-a-model") is None


def test_dollars_for_known_model() -> None:
    # gpt-5.4-mini: $0.25 in / $2 out per 1M tokens
    # 1000 in + 2000 out  → 0.25 * 0.001 + 2.0 * 0.002 = 0.00025 + 0.004 = 0.00425
    assert dollars_for("gpt-5.4-mini", 1000, 2000) == 0.00425


def test_dollars_for_unknown_model_is_zero() -> None:
    assert dollars_for("unknown-model", 1_000_000, 1_000_000) == 0.0


def test_store_records_and_aggregates(tmp_path: Path) -> None:
    store = CostStore(tmp_path / "cost.sqlite")

    d1 = store.record(model="gpt-5.4-mini", input_tokens=1000, output_tokens=2000,
                      thread_id="T1", trace_id="R1")
    d2 = store.record(model="gpt-5.4", input_tokens=500, output_tokens=1000,
                      thread_id="T1", trace_id="R1")
    d3 = store.record(model="gpt-5.4-mini", input_tokens=10, output_tokens=0,
                      thread_id="T2", trace_id="R2")

    t1 = store.thread_totals("T1")
    r1 = store.trace_totals("R1")
    t2 = store.thread_totals("T2")

    assert t1["events"] == 2
    assert t1["input_tokens"] == 1500
    assert t1["output_tokens"] == 3000
    assert t1["total_tokens"] == 4500
    assert t1["dollars"] == round(d1 + d2, 6)

    assert r1["events"] == 2
    assert r1["dollars"] == round(d1 + d2, 6)

    assert t2["events"] == 1
    assert t2["total_tokens"] == 10
    assert t2["dollars"] == round(d3, 6)

    store.close()


def test_unknown_model_still_records_tokens(tmp_path: Path) -> None:
    store = CostStore(tmp_path / "cost.sqlite")
    store.record(model="mystery", input_tokens=42, output_tokens=7, thread_id="T")
    totals = store.thread_totals("T")
    assert totals["total_tokens"] == 49
    assert totals["dollars"] == 0.0
    store.close()


def test_negative_tokens_clamped_to_zero(tmp_path: Path) -> None:
    store = CostStore(tmp_path / "cost.sqlite")
    store.record(model="gpt-5.4-mini", input_tokens=-10, output_tokens=-5, thread_id="T")
    totals = store.thread_totals("T")
    assert totals["input_tokens"] == 0
    assert totals["output_tokens"] == 0
    store.close()
