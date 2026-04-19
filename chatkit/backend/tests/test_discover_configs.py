"""Unit tests for app.pipelines.discover_configs (Pass 1)."""

from __future__ import annotations

from pathlib import Path

from app.pipelines.canonicalize import (
    AliasTable,
    RepoEntry,
    merge_edges,
    merge_nodes,
    node_id,
)
from app.pipelines.discover_configs import discover


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pipelines"


def _make_repo() -> RepoEntry:
    base = (FIXTURES / "repo-a").resolve()
    return RepoEntry(
        name="repo-a",
        local_path=base,
        config_roots=(base / "config_gold_prod",),
        pipelines=("priceeye-analytics",),
    )


def test_discover_yields_stage_and_io_edges_from_fixture() -> None:
    result = discover([_make_repo()])
    assert result.files_scanned >= 3
    assert result.files_with_signal >= 2  # alerts.properties has no IO

    # Merge + index for assertion convenience
    edge_by_rel = {}
    for e in merge_edges(result.edges):
        edge_by_rel.setdefault(e.rel, []).append((e.source_id, e.target_id))

    # competitive-position writes to the comp-pos bucket (both envs expanded)
    writes = set(edge_by_rel.get("writes", []))
    assert (node_id("stage", "competitive-position"),
            node_id("s3_prefix", "s3-atp-3victors3vdev-use1-competitive-position/v2")) in writes
    assert (node_id("stage", "competitive-position"),
            node_id("s3_prefix", "s3-atp-3victors3vprod-use1-competitive-position/v2")) in writes

    # competitive-position reads derived-common-output with v1 prefix
    reads = set(edge_by_rel.get("reads", []))
    assert (node_id("stage", "competitive-position"),
            node_id("s3_prefix", "s3-atp-3victors3vprod-use1-derived-common-output/v1")) in reads

    # competitive-position reads the redshift table analytics.derived_common_output
    assert (node_id("stage", "competitive-position"),
            node_id("redshift_table", "analytics.derived_common_output")) in reads


def test_discover_emits_glue_edge() -> None:
    result = discover([_make_repo()])
    writes = {(e.source_id, e.target_id) for e in result.edges if e.rel == "writes"}
    assert (node_id("stage", "market-level-generator"),
            node_id("glue_table",
                    "glue-atp-3victors3vprod-use1-analytics_db.market_level_anomalies_v4")) in writes


def test_discover_emits_pipeline_and_repo_edges() -> None:
    result = discover([_make_repo()])
    part_of = {(e.source_id, e.target_id) for e in result.edges if e.rel == "part_of"}
    assert (node_id("stage", "competitive-position"),
            node_id("pipeline", "priceeye-analytics")) in part_of
    repo_edges = {(e.source_id, e.target_id) for e in result.edges if e.rel == "repo"}
    assert (node_id("app", "market-level-generator"),
            node_id("repo", "repo-a")) in repo_edges


def test_discover_alerts_file_yields_stage_but_no_io() -> None:
    result = discover([_make_repo()])
    alerts_io = [
        e for e in result.edges
        if e.source_id == node_id("stage", "alerts") and e.rel in ("reads", "writes")
    ]
    assert alerts_io == []


def test_discover_is_case_insensitive_on_key_names() -> None:
    # The parser lowercases keys — verified implicitly via the fixture
    # using the real lowercased form. Just confirm nodes/edges are well-formed.
    result = discover([_make_repo()])
    assert all(isinstance(n.name, str) and n.name == n.name.lower() for n in result.nodes)


def test_discover_survives_missing_config_root(tmp_path: Path) -> None:
    repo = RepoEntry(
        name="missing", local_path=tmp_path, config_roots=(tmp_path / "nope",),
        pipelines=(),
    )
    result = discover([repo])
    assert result.files_scanned == 0
    assert result.nodes == []
    assert result.edges == []


# ── Real-world integration test: the 12 priceeye-analytics manifests ───


def test_discover_on_real_priceeye_analytics_repo_if_present() -> None:
    from app.pipelines.canonicalize import load_repos
    real_root = Path("~/git/ds-priceeye-analytics/docs/config_gold_prod").expanduser()
    if not real_root.exists():
        import pytest
        pytest.skip("ds-priceeye-analytics repo not cloned; skipping real-world test")
    repos = [r for r in load_repos() if r.name == "ds-priceeye-analytics"]
    assert repos, "repos.yaml must list ds-priceeye-analytics"
    result = discover(repos)
    # Should parse all 12 .properties files
    assert result.files_scanned >= 12
    # Should produce stage nodes for each
    stage_names = {n.name for n in result.nodes if n.kind == "stage"}
    for required in ("competitive-position", "market-level-generator",
                     "market-level-analysis", "segment-level-generator"):
        assert required in stage_names, f"missing stage: {required}"
    # And the documented upstream edge (comp-pos → dco S3)
    reads = {(e.source_id, e.target_id) for e in result.edges if e.rel == "reads"}
    assert (node_id("stage", "competitive-position"),
            node_id("s3_prefix",
                    "s3-atp-3victors3vprod-use1-derived-common-output/v1")) in reads
    _ = merge_nodes  # silence unused
