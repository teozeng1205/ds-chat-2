"""Tests for app.skills loader + classifier + renderer."""

from __future__ import annotations

from pathlib import Path

from app.skills import (
    DEFAULT_SKILLS_DIR,
    Skill,
    SkillRegistry,
    choose_skills,
    render_skills,
)
from app.skills import _parse_frontmatter  # noqa: PLC2701 — worth testing directly


def test_frontmatter_parses_scalars_and_inline_lists() -> None:
    text = """---
name: alpha
description: hello
keywords: [a, b, c]
tier: high
---
body goes here
"""
    fields, body = _parse_frontmatter(text)
    assert fields["name"] == "alpha"
    assert fields["description"] == "hello"
    assert fields["keywords"] == ("a", "b", "c")
    assert fields["tier"] == "high"
    assert body.strip() == "body goes here"


def test_frontmatter_parses_block_list() -> None:
    text = """---
name: bravo
keywords:
  - one
  - two
  - three
description: demo
---
body
"""
    fields, _ = _parse_frontmatter(text)
    assert fields["keywords"] == ("one", "two", "three")


def test_registry_loads_from_disk(tmp_path: Path) -> None:
    (tmp_path / "_README.md").write_text("ignored file", encoding="utf-8")  # should be skipped
    (tmp_path / "alpha.md").write_text("""---
name: alpha
description: A
keywords: [sql, redshift]
tier: high
---
Alpha body.
""", encoding="utf-8")
    (tmp_path / "bravo.md").write_text("""---
name: bravo
description: B
keywords: [bash, shell]
---
Bravo body.
""", encoding="utf-8")

    reg = SkillRegistry.load(tmp_path)
    assert {s.name for s in reg.skills} == {"alpha", "bravo"}
    assert reg.by_name("alpha").tier == "high"
    assert reg.by_name("bravo").tier == "normal"


def test_choose_skills_by_keyword_overlap(tmp_path: Path) -> None:
    (tmp_path / "sql.md").write_text("""---
name: sql
description: Use SQL when...
keywords: [sql, redshift, partition, query]
tier: high
---
SQL body
""", encoding="utf-8")
    (tmp_path / "shell.md").write_text("""---
name: shell
description: Use bash when...
keywords: [bash, shell, install, npm]
---
Shell body
""", encoding="utf-8")

    reg = SkillRegistry.load(tmp_path)

    hits = choose_skills("run a redshift query for sales_date", reg, k=2)
    names = [h.name for h in hits]
    assert names[0] == "sql"

    hits = choose_skills("install an npm package", reg, k=2)
    assert hits[0].name == "shell"

    # no overlap → empty
    assert choose_skills("xyzzy nothing matches", reg, k=2) == []


def test_choose_skills_respects_k_cap(tmp_path: Path) -> None:
    # 3 skills all matching
    for n in ("a", "b", "c"):
        (tmp_path / f"{n}.md").write_text(f"""---
name: {n}
description: X
keywords: [common]
---
body
""", encoding="utf-8")

    reg = SkillRegistry.load(tmp_path)
    hits = choose_skills("common word", reg, k=2)
    assert len(hits) == 2


def test_render_skills_wraps_each_in_tags() -> None:
    s = Skill(name="alpha", description="d", keywords=("k",), body="hello world", tier="normal")
    out = render_skills([s])
    assert '<skill name="alpha">' in out
    assert "</skill>" in out
    assert "hello world" in out


# ── Smoke-test the shipped skills directory ──


def test_default_skills_dir_is_loadable() -> None:
    reg = SkillRegistry.load(DEFAULT_SKILLS_DIR)
    # We shipped sql_investigation and pipeline_ops in the same commit
    names = {s.name for s in reg.skills}
    assert "sql_investigation" in names
    assert "pipeline_ops" in names
