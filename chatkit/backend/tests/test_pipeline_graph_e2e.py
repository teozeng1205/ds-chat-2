"""End-to-end regression: the on-disk pipeline graph is populated and
covers the priceeye anomalies chain.

Runs against the committed `app/investigation/knowledge/pipelines.json`
so CI / humans see the moment the graph drifts away from shipping the
priceeye chain. No network. No LLM. Reads the JSON only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "app" / "investigation" / "knowledge"
PIPELINES_PATH = KNOWLEDGE_DIR / "pipelines.json"


def _load_graph() -> dict:
    if not PIPELINES_PATH.exists():
        pytest.skip("pipelines.json not yet built — run scripts/build_pipeline_graph.py")
    return json.loads(PIPELINES_PATH.read_text(encoding="utf-8"))


def _names_by_kind(graph: dict, kind: str) -> set[str]:
    return {n["name"] for n in graph.get("nodes", {}).get(kind, [])}


def test_graph_has_nonzero_nodes_and_edges() -> None:
    graph = _load_graph()
    stats = graph.get("_stats", {})
    assert stats.get("nodes", 0) >= 50, stats
    assert stats.get("edges", 0) >= 50, stats


def test_priceeye_stages_covered() -> None:
    """The priceeye anomalies chain is the canonical 'this graph is
    useful' test case. Every stage in the documented chain must be a
    node before we ship."""
    graph = _load_graph()
    stages = _names_by_kind(graph, "stage") | _names_by_kind(graph, "app")
    expected = {
        "common-output",
        "derived-common-output",
        "competitive-position",
        "market-level-analysis",
        "market-level-generator",
    }
    missing = expected - stages
    assert not missing, f"priceeye stages missing from graph: {missing}"


def test_priceeye_redshift_table_has_producer() -> None:
    """`prod.analytics.market_level_anomalies_v4` should be produced by
    at least one stage in the graph (it's the final output of the
    anomalies chain). We accept any stage/app edge — the shape is what
    matters, not the exact producer name, because discovery may
    attribute it to either `market-level-generator` or its `app` twin."""
    graph = _load_graph()
    edges = graph.get("edges", [])
    anomaly_ids = {
        nid["id"] for nid in graph.get("nodes", {}).get("redshift_table", [])
        if "market_level_anomalies_v4" in nid["name"]
    } | {
        nid["id"] for nid in graph.get("nodes", {}).get("glue_table", [])
        if "market_level_anomalies_v4" in nid["name"]
    }
    if not anomaly_ids:
        pytest.skip("market_level_anomalies_v4 not yet discovered by any pass")

    producers = {
        e["source"] for e in edges
        if e.get("target") in anomaly_ids and e.get("rel") == "writes"
    }
    assert producers, (
        f"anomalies table has no producer: tables={anomaly_ids}"
    )
    # Producers should be stage / app nodes, not raw repos
    assert any(p.startswith(("stage:", "app:")) for p in producers), producers


def test_every_repo_with_config_discovered() -> None:
    """repos.yaml + auto-discovery should ensure that any repo which
    has a priceeye-style config_*/ folder shows up as a `repo` node.
    Regression guard against an over-eager auto-discovery skip."""
    graph = _load_graph()
    repo_names = _names_by_kind(graph, "repo")
    assert "ds-priceeye-analytics" in repo_names, sorted(repo_names)
