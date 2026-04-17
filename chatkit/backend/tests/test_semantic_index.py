"""Unit tests for app.investigation.semantic_index."""

from __future__ import annotations

import math
from pathlib import Path

from app.investigation.semantic_index import SemanticIndex, tokenize


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def test_tokenize_lowercases_and_filters_short() -> None:
    assert tokenize("SELECT * FROM prod.analytics.market_level_anomalies_v3") == [
        "select", "from", "prod", "analytics", "market_level_anomalies_v3",
    ]


def test_upsert_and_search_cosine(tmp_path: Path) -> None:
    idx = SemanticIndex(tmp_path / "idx.sqlite")

    idx.upsert("a", "alpha table about anomalies", _unit([1.0, 0.0, 0.0, 0.0]), kind="table")
    idx.upsert("b", "bravo doc explains monitoring", _unit([0.0, 1.0, 0.0, 0.0]), kind="doc")
    idx.upsert("c", "charlie dataset has pricing", _unit([0.0, 0.0, 1.0, 0.0]), kind="table")

    hits = idx.search(_unit([0.9, 0.1, 0.0, 0.0]), top_k=3)
    assert [h.id for h in hits][0] == "a"
    assert hits[1].id == "b"
    idx.close()


def test_search_filters_by_kind(tmp_path: Path) -> None:
    idx = SemanticIndex(tmp_path / "idx.sqlite")
    idx.upsert("a", "alpha", _unit([1.0, 0.0]), kind="table")
    idx.upsert("d", "delta", _unit([1.0, 0.0]), kind="doc")
    hits = idx.search(_unit([1.0, 0.0]), top_k=5, kind="doc")
    assert [h.id for h in hits] == ["d"]
    idx.close()


def test_dim_mismatch_raises(tmp_path: Path) -> None:
    idx = SemanticIndex(tmp_path / "idx.sqlite")
    idx.upsert("a", "alpha", [1.0, 0.0, 0.0])
    try:
        idx.upsert("b", "bravo", [1.0, 0.0])
    except ValueError:
        pass
    else:  # pragma: no cover
        assert False, "expected ValueError on dim mismatch"
    idx.close()


def test_hybrid_blends_cosine_with_lexical(tmp_path: Path) -> None:
    idx = SemanticIndex(tmp_path / "idx.sqlite")
    idx.upsert("a", "market_level_anomalies by customer", _unit([1.0, 0.0]))
    idx.upsert("b", "generic other doc", _unit([1.0, 0.0]))
    pure = idx.search(_unit([1.0, 0.0]), top_k=2)
    hybrid = idx.hybrid_search(
        _unit([1.0, 0.0]),
        lexical_terms=["market_level_anomalies", "customer"],
        top_k=2,
    )
    assert [h.id for h in hybrid] == ["a", "b"]
    assert {h.id for h in pure} == {"a", "b"}
    idx.close()


def test_upsert_updates_existing_row(tmp_path: Path) -> None:
    idx = SemanticIndex(tmp_path / "idx.sqlite")
    idx.upsert("a", "old text", _unit([1.0, 0.0]))
    idx.upsert("a", "new text", _unit([0.0, 1.0]))
    hits = idx.search(_unit([0.0, 1.0]), top_k=1)
    assert hits[0].id == "a" and hits[0].text == "new text"
    assert idx.count() == 1
    idx.close()


def test_delete_and_clear(tmp_path: Path) -> None:
    idx = SemanticIndex(tmp_path / "idx.sqlite")
    idx.upsert("a", "x", _unit([1.0, 0.0]))
    idx.upsert("b", "y", _unit([0.0, 1.0]))
    assert idx.delete("a") is True
    assert idx.count() == 1
    idx.clear()
    assert idx.count() == 0
    idx.close()


def test_metadata_roundtrip(tmp_path: Path) -> None:
    idx = SemanticIndex(tmp_path / "idx.sqlite")
    idx.upsert("a", "x", _unit([1.0, 0.0]), metadata={"source": "tables.md", "line": 12})
    hits = idx.search(_unit([1.0, 0.0]), top_k=1)
    assert hits[0].metadata == {"source": "tables.md", "line": 12}
    idx.close()
