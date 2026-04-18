"""Unit tests for app.investigation.knowledge.build_embeddings.

Focus: chunking pipeline. We exercise run_build in dry-run mode so
no OpenAI call is made.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.investigation.knowledge import build_embeddings as be


def test_chunk_markdown_splits_by_heading(tmp_path: Path, monkeypatch) -> None:
    md = tmp_path / "demo.md"
    md.write_text(
        "# Top\nintro\n\n## Section A\naaa\n\n## Section B\nbbb\n",
        encoding="utf-8",
    )
    # Point KNOWLEDGE_DIR at tmp so relative-path metadata resolves
    monkeypatch.setattr(be, "KNOWLEDGE_DIR", tmp_path)
    chunks = be._chunk_markdown(md, kind="doc")
    assert [c.metadata["heading"] for c in chunks] == ["Top", "Section A", "Section B"]
    assert all(c.kind == "doc" for c in chunks)
    assert all(c.metadata["source"] == "demo.md" for c in chunks)


def test_chunk_common_codes_emits_one_per_entry(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "common_codes.json").write_text(json.dumps({
        "providers": [{"code": "QL2", "name": "QL2 Software", "aliases": ["QL2"]}],
        "sites": ["AV"],
        "customers": [{"code": "B6", "name": "JetBlue", "aliases": ["JetBlue"]}],
    }), encoding="utf-8")
    monkeypatch.setattr(be, "KNOWLEDGE_DIR", tmp_path)
    chunks = be._chunk_common_codes()
    ids = [c.id for c in chunks]
    assert "code:providers:QL2" in ids
    assert "code:customers:B6" in ids
    assert "code:sites:AV" in ids
    by_kind = {c.kind for c in chunks}
    assert by_kind == {"code"}


def test_run_build_dry_run_yields_summary(tmp_path: Path, monkeypatch) -> None:
    # Minimal fake knowledge tree
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "d1.md").write_text("# D1\nhello\n", encoding="utf-8")
    (tmp_path / "tables.md").write_text("# Tables\nbody\n", encoding="utf-8")
    (tmp_path / "common_codes.json").write_text(json.dumps({
        "providers": [], "sites": [], "customers": [],
    }), encoding="utf-8")
    (tmp_path / "sql_best_practices.md").write_text("# SQL\nbody\n", encoding="utf-8")

    monkeypatch.setattr(be, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(be, "DOCS_DIR", tmp_path / "docs")

    summary = be.run_build(dry_run=True)
    assert summary["dry_run"] is True
    assert summary["chunks_total"] >= 3
    # Codes kind produces 0 chunks since the arrays are empty
    assert "doc" in summary["chunks_by_kind"]
    assert "tables" in summary["chunks_by_kind"]


def test_run_build_respects_kinds_filter(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "tables.md").write_text("# Tables\nbody\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "d1.md").write_text("# D1\nhello\n", encoding="utf-8")
    (tmp_path / "common_codes.json").write_text(json.dumps({
        "providers": [], "sites": [], "customers": [],
    }), encoding="utf-8")
    (tmp_path / "sql_best_practices.md").write_text("# S\n", encoding="utf-8")
    monkeypatch.setattr(be, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(be, "DOCS_DIR", tmp_path / "docs")
    summary = be.run_build(["tables"], dry_run=True)
    assert set(summary["chunks_by_kind"]) == {"tables"}


def test_main_cli_dry_run(capsys, tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "d1.md").write_text("# D1\nhello\n", encoding="utf-8")
    (tmp_path / "tables.md").write_text("# T\n", encoding="utf-8")
    (tmp_path / "common_codes.json").write_text(json.dumps({
        "providers": [], "sites": [], "customers": [],
    }), encoding="utf-8")
    (tmp_path / "sql_best_practices.md").write_text("# S\n", encoding="utf-8")
    monkeypatch.setattr(be, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(be, "DOCS_DIR", tmp_path / "docs")
    rc = be.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["dry_run"] is True
    assert parsed["chunks_total"] >= 2
