"""Unit tests for `_pipeline_context` — the search_kb overlay that
attaches 1-hop lineage to KB hits that resolve to graph nodes."""

from __future__ import annotations

from pathlib import Path

from app.pipelines.canonicalize import Edge, Node
from app.pipelines.graph_store import GraphStore
from app.tools import investigation_tools as it


def _seed_graph(tmp_path: Path) -> Path:
    db = tmp_path / "graph.sqlite"
    store = GraphStore(db)
    nodes = [
        Node(kind="stage", name="market-level-generator",
             aliases=("mlg",), metadata={"repo": "ds-priceeye-analytics"},
             source="config:x"),
        Node(kind="redshift_table", name="analytics.market_level_anomalies_v4",
             aliases=(), metadata={}, source="config:x"),
        Node(kind="redshift_table", name="analytics.market_level_analysis_v2",
             aliases=(), metadata={}, source="config:x"),
        Node(kind="s3_prefix", name="s3-atp-3victors-3vprod-use1-anomaly-datasets/market-level/v4",
             aliases=(), metadata={}, source="config:x"),
    ]
    edges = [
        # stage reads analysis table
        Edge(source_id="stage:market-level-generator",
             target_id="redshift_table:analytics.market_level_analysis_v2",
             rel="reads", source="config:x"),
        # stage writes anomalies table + S3 prefix
        Edge(source_id="stage:market-level-generator",
             target_id="redshift_table:analytics.market_level_anomalies_v4",
             rel="writes", source="config:x"),
        Edge(source_id="stage:market-level-generator",
             target_id="s3_prefix:s3-atp-3victors-3vprod-use1-anomaly-datasets/market-level/v4",
             rel="writes", source="config:x"),
    ]
    store.upsert(nodes, edges)
    store.close()
    return db


def test_pipeline_context_resolves_candidate_table(tmp_path: Path, monkeypatch) -> None:
    db = _seed_graph(tmp_path)
    monkeypatch.setenv("DS_CHAT_PIPELINE_GRAPH_DB", str(db))

    ctx = it._pipeline_context(
        candidate_tables=["analytics.market_level_anomalies_v4"],
        semantic_hits=None,
    )
    assert "redshift_table:analytics.market_level_anomalies_v4" in ctx
    entry = ctx["redshift_table:analytics.market_level_anomalies_v4"]
    # Upstream produced_by is the stage
    assert entry.get("produced_by") == ["stage:market-level-generator"]


def test_pipeline_context_resolves_alias_from_semantic_hits(tmp_path: Path, monkeypatch) -> None:
    db = _seed_graph(tmp_path)
    monkeypatch.setenv("DS_CHAT_PIPELINE_GRAPH_DB", str(db))

    ctx = it._pipeline_context(
        candidate_tables=[],
        semantic_hits=[{"id": "MLG", "metadata": {}}],
    )
    # Alias MLG resolves to stage:market-level-generator
    assert "stage:market-level-generator" in ctx
    entry = ctx["stage:market-level-generator"]
    assert entry["kind"] == "stage"
    assert set(entry.get("writes_to") or []) >= {
        "redshift_table:analytics.market_level_anomalies_v4"
    }


def test_pipeline_context_missing_db_returns_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DS_CHAT_PIPELINE_GRAPH_DB", str(tmp_path / "does_not_exist.sqlite"))
    assert it._pipeline_context(candidate_tables=["foo"], semantic_hits=None) == {}


def test_pipeline_context_empty_graph_returns_empty(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "empty.sqlite"
    GraphStore(db).close()
    monkeypatch.setenv("DS_CHAT_PIPELINE_GRAPH_DB", str(db))
    assert it._pipeline_context(candidate_tables=["foo"], semantic_hits=None) == {}


def test_pipeline_context_honors_max_entries(tmp_path: Path, monkeypatch) -> None:
    db = _seed_graph(tmp_path)
    monkeypatch.setenv("DS_CHAT_PIPELINE_GRAPH_DB", str(db))
    ctx = it._pipeline_context(
        candidate_tables=[
            "analytics.market_level_anomalies_v4",
            "analytics.market_level_analysis_v2",
            "market-level-generator",
        ],
        semantic_hits=None,
        max_entries=2,
    )
    assert len(ctx) <= 2
