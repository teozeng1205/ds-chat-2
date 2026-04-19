"""Unit tests for app.pipelines.canonicalize."""

from __future__ import annotations

from pathlib import Path

from app.pipelines.canonicalize import (
    AliasTable,
    Edge,
    Node,
    canonical_glue_table,
    canonical_redshift_table,
    canonical_s3_prefix,
    expand_environment,
    load_repos,
    merge_edges,
    merge_nodes,
    node_id,
)


# ── Alias table ────────────────────────────────────────────────────────


def test_alias_table_resolves_case_and_underscore_variants(tmp_path: Path) -> None:
    p = tmp_path / "aliases.yaml"
    p.write_text("""
derived-common-output:
  - dco
  - derived_common_output
  - "Derived Common Output"

competitive-position:
  - comp-pos
  - competitive_position
""", encoding="utf-8")
    t = AliasTable.load(p)
    assert t.resolve("DCO") == "derived-common-output"
    assert t.resolve("derived_common_output") == "derived-common-output"
    assert t.resolve("Derived Common Output") == "derived-common-output"
    assert t.resolve("comp-pos") == "competitive-position"
    # Unknown → normalized (underscores / spaces → hyphens, lowercased)
    assert t.resolve("Some New Thing") == "some-new-thing"
    assert t.resolve("another_name") == "another-name"


def test_alias_table_handles_missing_file(tmp_path: Path) -> None:
    t = AliasTable.load(tmp_path / "nope.yaml")
    assert t.resolve("Foo") == "foo"


def test_alias_table_matches_ships_with_priceeye_names() -> None:
    # Uses the project-shipped aliases.yaml
    t = AliasTable.load()
    assert t.resolve("DCO") == "derived-common-output"
    assert t.resolve("MLG") == "market-level-generator"
    assert t.resolve("MLA") == "market-level-analysis"


# ── Env expansion ──────────────────────────────────────────────────────


def test_expand_environment_produces_one_per_env() -> None:
    assert expand_environment("s3-atp-3victors${environment}-use1-foo") == [
        "s3-atp-3victors3vdev-use1-foo",
        "s3-atp-3victors3vprod-use1-foo",
    ]


def test_expand_environment_noop_when_no_placeholder() -> None:
    assert expand_environment("plain-bucket") == ["plain-bucket"]


def test_expand_environment_case_insensitive_placeholder() -> None:
    assert expand_environment("${ENVIRONMENT}-suffix") == [
        "3vdev-suffix",
        "3vprod-suffix",
    ]


# ── S3 / Redshift / Glue normalizers ───────────────────────────────────


def test_canonical_s3_prefix_strips_scheme_and_slashes() -> None:
    assert canonical_s3_prefix("s3://My-Bucket/", "/v4/") == "my-bucket/v4"


def test_canonical_s3_prefix_empty_prefix() -> None:
    assert canonical_s3_prefix("My-Bucket") == "my-bucket"


def test_canonical_redshift_table_lowercases_and_strips_quotes() -> None:
    assert canonical_redshift_table('"Prod.Analytics.Market_Level_Anomalies_V4"') == \
        "prod.analytics.market_level_anomalies_v4"


def test_canonical_glue_table() -> None:
    assert canonical_glue_table("Glue_DB", "Market_Level_Anomalies_v4") == \
        "glue_db.market_level_anomalies_v4"


# ── Node ID ────────────────────────────────────────────────────────────


def test_node_id_format() -> None:
    assert node_id("app", "Competitive-Position") == "app:competitive-position"
    assert node_id("s3_prefix", "MyBucket/V4") == "s3_prefix:mybucket/v4"


# ── Merge helpers ──────────────────────────────────────────────────────


def test_merge_nodes_unions_aliases_and_metadata() -> None:
    a = Node(kind="app", name="x", aliases=("xa",), metadata={"repo": "r1"}, source="s1")
    b = Node(kind="app", name="x", aliases=("xb",), metadata={"repo": "r1", "desc": "hi"}, source="s2")
    m = merge_nodes(a, b)
    assert set(m.aliases) == {"xa", "xb"}
    assert m.metadata == {"repo": "r1", "desc": "hi"}
    assert set(m.source.split(", ")) == {"s1", "s2"}


def test_merge_edges_dedupes_and_unions_sources() -> None:
    e1 = Edge("app:a", "redshift_table:t", "writes", source="s1")
    e2 = Edge("app:a", "redshift_table:t", "writes", source="s2")
    e3 = Edge("app:a", "redshift_table:t", "reads",  source="s3")   # different rel, kept
    out = merge_edges([e1, e2, e3])
    # The writes edge merges, reads stays separate → two edges total
    assert len(out) == 2
    writes = next(e for e in out if e.rel == "writes")
    assert set(writes.source.split(", ")) == {"s1", "s2"}


# ── Repos ──────────────────────────────────────────────────────────────


def test_load_repos_reads_shipped_yaml() -> None:
    repos = load_repos()
    names = {r.name for r in repos}
    assert "ds-priceeye-analytics" in names
    # At least one repo declares a config_root so discover_configs has
    # somewhere to scan
    assert any(r.config_roots for r in repos)
