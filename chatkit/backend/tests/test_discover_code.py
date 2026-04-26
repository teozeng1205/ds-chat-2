"""Unit tests for app.pipelines.discover_code (Pass 3)."""

from __future__ import annotations

from pathlib import Path

from app.pipelines.canonicalize import (
    RepoEntry,
    merge_edges,
    node_id,
)
from app.pipelines.discover_code import discover


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pipelines" / "code-repo"


def _make_repo() -> RepoEntry:
    return RepoEntry(
        name="code-repo",
        local_path=FIXTURES.resolve(),
        config_roots=(),       # Pass 3 ignores config_roots; it walks the whole repo
        pipelines=("test-pipeline",),
    )


def test_discover_extracts_s3_literal_writes_and_reads() -> None:
    result = discover([_make_repo()])
    edges = merge_edges(result.edges)
    by_target = {e.target_id: e.rel for e in edges}

    # From run.py (attributed to the market-level-generator stage because the
    # folder name matches a known alias in the shipped aliases.yaml)
    # s3 writes to anomaly-datasets and reads from competitive-position
    assert node_id("s3_prefix", "s3-atp-3victors3vprod-use1-anomaly-datasets/market-level/v4") \
        in by_target


def test_discover_file_in_known_stage_folder_attributes_to_stage() -> None:
    result = discover([_make_repo()])
    stage_writes = {
        e.target_id
        for e in result.edges
        if e.source_id == node_id("stage", "market-level-generator") and e.rel == "writes"
    }
    # run.py writes to the anomaly-datasets bucket; should be attributed to
    # the market-level-generator stage
    assert any("anomaly-datasets" in t for t in stage_writes)


def test_discover_detects_redshift_unload_to_s3() -> None:
    result = discover([_make_repo()])
    # UNLOAD … TO s3://... → writes S3
    writes = {(e.source_id, e.target_id, e.rel) for e in result.edges}
    expected_target = node_id("s3_prefix",
                              "s3-atp-3victors3vprod-use1-anomaly-datasets/market-level/unload")
    assert any(t == expected_target for _, t, _ in writes)


def test_discover_detects_redshift_copy_from_s3() -> None:
    result = discover([_make_repo()])
    # COPY analytics.competitive_position FROM 's3://...' → writes the table + reads the s3
    writes = {(e.source_id, e.target_id) for e in result.edges if e.rel == "writes"}
    reads  = {(e.source_id, e.target_id) for e in result.edges if e.rel == "reads"}
    assert any(t == node_id("redshift_table", "analytics.competitive_position") for _, t in writes)
    assert any("competitive-position/v2" in t for _, t in reads)


def test_discover_skips_glue_asset_and_config_server_noise() -> None:
    result = discover([_make_repo()])
    targets = {e.target_id for e in result.edges}
    for noise in targets:
        assert "aws-glue-assets" not in noise
        assert "config-server-" not in noise


def test_discover_skips_files_under_tests_dir() -> None:
    result = discover([_make_repo()])
    targets = {e.target_id for e in result.edges}
    for t in targets:
        assert "should-not-appear" not in t


def test_discover_detects_insert_into_and_update() -> None:
    result = discover([_make_repo()])
    writes = {(e.source_id, e.target_id) for e in result.edges if e.rel == "writes"}
    # The SQL fixture has INSERT INTO analytics.daily_summary + UPDATE on same
    summary_target = node_id("redshift_table", "analytics.daily_summary")
    assert any(t == summary_target for _, t in writes)
    # And SQL `FROM prod.monitoring.provider_combined_audit` → reads
    reads = {e.target_id for e in result.edges if e.rel == "reads"}
    assert node_id("redshift_table", "prod.monitoring.provider_combined_audit") in reads


def test_discover_survives_missing_repo_path(tmp_path: Path) -> None:
    repo = RepoEntry(name="gone", local_path=tmp_path / "nope", config_roots=(), pipelines=())
    result = discover([repo])
    assert result.files_scanned == 0
    assert result.nodes == []
