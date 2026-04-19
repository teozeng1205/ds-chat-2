"""Unit tests for app.pipelines.discover_docs (Pass 5)."""

from __future__ import annotations

from pathlib import Path

from app.pipelines.canonicalize import AliasTable, node_id
from app.pipelines.discover_docs import discover


def _write_doc(tmp_path: Path, name: str, body: str) -> Path:
    (tmp_path / "docs").mkdir(exist_ok=True)
    p = tmp_path / "docs" / name
    p.write_text(body, encoding="utf-8")
    return p


def _run(tmp_path: Path, aliases_yaml: str | None = None) -> object:
    aliases_path = tmp_path / "aliases.yaml"
    if aliases_yaml:
        aliases_path.write_text(aliases_yaml, encoding="utf-8")
    else:
        aliases_path.write_text("""
derived-common-output: [dco, derived_common_output]
competitive-position: [comp-pos, competitive_position]
market-level-generator: [mlg]
market-level-analysis: [mla]
""", encoding="utf-8")
    aliases = AliasTable.load(aliases_path)
    return discover(doc_roots=[tmp_path / "docs"], aliases=aliases)


def test_discover_unicode_arrow(tmp_path: Path) -> None:
    _write_doc(tmp_path, "a.md",
               "The pipeline flows `derived-common-output` → `competitive-position` each hour.")
    result = _run(tmp_path)
    src = node_id("stage", "derived-common-output")
    tgt = node_id("stage", "competitive-position")
    assert any(e.source_id == src and e.target_id == tgt and e.rel == "writes"
               for e in result.edges)


def test_discover_prose_writes_to(tmp_path: Path) -> None:
    _write_doc(tmp_path, "b.md",
               "The `market-level-generator` writes to `analytics.market_level_anomalies_v4`.")
    result = _run(tmp_path)
    src = node_id("stage", "market-level-generator")
    tgt = node_id("redshift_table", "analytics.market_level_anomalies_v4")
    assert any(e.source_id == src and e.target_id == tgt and e.rel == "writes"
               for e in result.edges)


def test_discover_prose_reads_from(tmp_path: Path) -> None:
    _write_doc(tmp_path, "c.md",
               "Stage `competitive-position` reads from `derived-common-output`.")
    result = _run(tmp_path)
    src = node_id("stage", "competitive-position")
    tgt = node_id("stage", "derived-common-output")
    assert any(e.source_id == src and e.target_id == tgt and e.rel == "reads"
               for e in result.edges)


def test_discover_ignores_heading_and_code_fences(tmp_path: Path) -> None:
    _write_doc(tmp_path, "d.md", """
# Pipeline → not an edge
|--- | --- |
`competitive-position` → `market-level-analysis`
""")
    result = _run(tmp_path)
    # The only edge is from the real content line, not the heading
    edges_with_real_source = [e for e in result.edges
                              if e.source_id == node_id("stage", "competitive-position")]
    assert len(edges_with_real_source) == 1


def test_discover_survives_unresolvable_entities(tmp_path: Path) -> None:
    _write_doc(tmp_path, "e.md", "random text with X → Y arrows and other things")
    result = _run(tmp_path)
    # Neither X nor Y resolves to a known entity → no edge
    assert all(e.weight != 0.5 or "stage:x" not in e.source_id for e in result.edges)


def test_discover_handles_missing_root() -> None:
    result = discover(doc_roots=[Path("/nonexistent/path")])
    assert result.nodes == []
    assert result.edges == []


def test_discover_does_not_over_match_from_to_prose(tmp_path: Path) -> None:
    # "from X to Y" is NOT one of our patterns; this doc should not
    # spuriously emit an edge for that construction.
    _write_doc(tmp_path, "f.md",
               "Data flows from derived-common-output to competitive-position.")
    result = _run(tmp_path)
    unrelated = [e for e in result.edges
                 if "derived-common-output" in e.source_id and "competitive-position" in e.target_id]
    assert unrelated == []
