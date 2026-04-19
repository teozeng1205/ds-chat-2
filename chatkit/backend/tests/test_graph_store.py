"""Unit tests for app.pipelines.graph_store."""

from __future__ import annotations

from pathlib import Path

from app.pipelines.canonicalize import Edge, Node, node_id
from app.pipelines.graph_store import GraphStore


def _mk_store(tmp_path: Path) -> GraphStore:
    return GraphStore(db_path=tmp_path / "g.sqlite")


# ── upsert ────────────────────────────────────────────────────────────


def test_upsert_idempotent(tmp_path: Path) -> None:
    s = _mk_store(tmp_path)
    n = Node(kind="stage", name="comp-pos", source="test")
    e = Edge(source_id="stage:comp-pos", target_id="redshift_table:x",
             rel="writes", source="test")
    for _ in range(3):
        s.upsert([n], [e])
    stats = s.stats()
    assert stats["total_nodes"] == 1
    assert stats["total_edges"] == 1
    s.close()


def test_upsert_merges_metadata_on_conflict(tmp_path: Path) -> None:
    s = _mk_store(tmp_path)
    s.upsert([Node(kind="stage", name="x", metadata={"repo": "r1"}, source="s1")], [])
    s.upsert([Node(kind="stage", name="x", metadata={"other": "v"}, source="s2")], [])
    hit = s.get_node(node_id("stage", "x"))
    assert hit is not None
    # Latest write wins for the metadata payload (we serialize whole dict)
    assert hit.metadata == {"other": "v"}
    s.close()


# ── neighbors (BFS) ───────────────────────────────────────────────────


def _priceeye_like_store(tmp_path: Path) -> GraphStore:
    """Build a tiny graph matching the shape Pass 1 produces for the
    priceeye analytics chain (comp-pos → MLA → MLG)."""
    s = _mk_store(tmp_path)
    nodes = [
        Node(kind="stage", name="comp-pos", source="t"),
        Node(kind="stage", name="mla", source="t"),
        Node(kind="stage", name="mlg", source="t"),
        Node(kind="s3_prefix", name="dco-bucket/v1", source="t"),
        Node(kind="s3_prefix", name="comp-pos-bucket/v2", source="t"),
        Node(kind="s3_prefix", name="anomaly-bucket/market/v4", source="t"),
        Node(kind="redshift_table", name="prod.analytics.market_level_anomalies_v4", source="t"),
    ]
    edges = [
        # comp-pos reads dco, writes comp-pos-bucket/v2
        Edge("stage:comp-pos",      "s3_prefix:dco-bucket/v1",            "reads",  source="t"),
        Edge("stage:comp-pos",      "s3_prefix:comp-pos-bucket/v2",       "writes", source="t"),
        # mla reads comp-pos-bucket/v2, writes anomaly-bucket/market/v4
        Edge("stage:mla",           "s3_prefix:comp-pos-bucket/v2",       "reads",  source="t"),
        Edge("stage:mla",           "s3_prefix:anomaly-bucket/market/v4", "writes", source="t"),
        # mlg writes both the anomaly bucket AND the redshift table
        Edge("stage:mlg",           "s3_prefix:anomaly-bucket/market/v4", "writes", source="t"),
        Edge("stage:mlg",           "redshift_table:prod.analytics.market_level_anomalies_v4",
             "writes", source="t"),
    ]
    s.upsert(nodes, edges)
    return s


def test_neighbors_upstream_from_final_table(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    out = s.neighbors(
        node_id("redshift_table", "prod.analytics.market_level_anomalies_v4"),
        direction="upstream", depth=3,
    )
    reached = {n.id for n in out["nodes"]}
    # Upstream from the final table → MLG (writer)
    assert node_id("stage", "mlg") in reached
    s.close()


def test_neighbors_downstream_from_source_bucket(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    out = s.neighbors(
        node_id("s3_prefix", "dco-bucket/v1"), direction="downstream", depth=5,
    )
    reached = {n.id for n in out["nodes"]}
    # dco bucket → comp-pos (reads it) → comp-pos-bucket → mla → anomaly bucket
    assert node_id("stage", "comp-pos") in reached
    assert node_id("stage", "mla") in reached
    assert node_id("s3_prefix", "anomaly-bucket/market/v4") in reached
    s.close()


def test_neighbors_depth_caps_traversal(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    deep = s.neighbors(node_id("s3_prefix", "dco-bucket/v1"),
                       direction="downstream", depth=10)["nodes"]
    shallow = s.neighbors(node_id("s3_prefix", "dco-bucket/v1"),
                          direction="downstream", depth=1)["nodes"]
    assert len(deep) >= len(shallow)
    s.close()


def test_neighbors_missing_node_returns_empty(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    out = s.neighbors("stage:does-not-exist", direction="both", depth=3)
    assert out["origin"] is None
    assert out["nodes"] == []
    assert out["edges"] == []
    s.close()


# ── resolve (alias / contains lookup) ─────────────────────────────────


def test_resolve_exact_id(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    assert s.resolve("stage:comp-pos") == "stage:comp-pos"
    s.close()


def test_resolve_by_alias(tmp_path: Path) -> None:
    # Use a node whose alias appears in the shipped aliases.yaml
    s = _mk_store(tmp_path)
    s.upsert([Node(kind="stage", name="competitive-position", source="t")], [])
    assert s.resolve("DCO") is None  # no DCO node
    # But "comp-pos" (an alias of competitive-position) resolves to the real node
    assert s.resolve("comp-pos") == "stage:competitive-position"
    s.close()


def test_resolve_by_contains_on_table_name(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    # Agent might ask for "market_level_anomalies_v4"
    found = s.resolve("market_level_anomalies_v4")
    assert found == node_id("redshift_table",
                            "prod.analytics.market_level_anomalies_v4")
    s.close()


def test_resolve_missing_returns_none(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    assert s.resolve("totally-unrelated-string") is None
    s.close()


# ── stats ────────────────────────────────────────────────────────────


def test_stats_counts(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    stats = s.stats()
    assert stats["total_nodes"] == 7
    assert stats["total_edges"] == 6
    assert stats["by_kind"]["stage"] == 3
    assert stats["by_rel"]["writes"] == 4
    s.close()


def test_clear_wipes_rows(tmp_path: Path) -> None:
    s = _priceeye_like_store(tmp_path)
    s.clear()
    stats = s.stats()
    assert stats["total_nodes"] == 0
    assert stats["total_edges"] == 0
    s.close()
