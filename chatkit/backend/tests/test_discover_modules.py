"""Unit tests for app.pipelines.discover_modules (Pass 1b)."""

from __future__ import annotations

from pathlib import Path

from app.pipelines.canonicalize import AliasTable, RepoEntry
from app.pipelines.discover_modules import discover


def _repo(tmp_path: Path, name: str = "priceeye-v2") -> RepoEntry:
    return RepoEntry(
        name=name,
        local_path=tmp_path,
        config_roots=(tmp_path,),
        pipelines=(),
    )


def test_discovers_maven_submodules(tmp_path: Path) -> None:
    # Simulate priceeye-v2's source/<module>/src/main/java/ layout
    for module in ("common-output-generator", "common-audit-generator", "daily-billing"):
        (tmp_path / "source" / module / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "source" / module / "pom.xml").write_text("<project/>", encoding="utf-8")

    result = discover([_repo(tmp_path)], aliases=AliasTable.load())
    stage_names = sorted({n.name for n in result.nodes if n.kind == "stage"})
    assert "common-output-generator" in stage_names
    assert "common-audit-generator" in stage_names
    assert "daily-billing" in stage_names

    # Every stage gets a matching app node
    app_names = {n.name for n in result.nodes if n.kind == "app"}
    assert app_names >= set(stage_names)


def test_skips_common_utility_dirs(tmp_path: Path) -> None:
    # `common`, `dao`, `utils`, `lib`, `src` etc. are plumbing, not stages
    for noise in ("common", "dao", "utils", "src"):
        (tmp_path / "source" / noise / "src" / "main" / "java").mkdir(parents=True)
    result = discover([_repo(tmp_path)], aliases=AliasTable.load())
    names = {n.name for n in result.nodes}
    assert "common" not in names
    assert "dao" not in names


def test_requires_module_marker(tmp_path: Path) -> None:
    # A folder under source/ with NO pom.xml / src/ / pyproject.toml
    # should NOT become a stage — most likely a docs/notes folder.
    (tmp_path / "source" / "some-folder").mkdir(parents=True)
    (tmp_path / "source" / "some-folder" / "NOTES.md").write_text("docs", encoding="utf-8")
    result = discover([_repo(tmp_path)], aliases=AliasTable.load())
    names = {n.name for n in result.nodes}
    assert "some-folder" not in names


def test_covers_multiple_parent_layouts(tmp_path: Path) -> None:
    # packages/, services/, lambdas/ all count
    (tmp_path / "packages" / "api-gateway").mkdir(parents=True)
    (tmp_path / "packages" / "api-gateway" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "services" / "scheduler-svc").mkdir(parents=True)
    (tmp_path / "services" / "scheduler-svc" / "Dockerfile").write_text("FROM scratch", encoding="utf-8")
    (tmp_path / "lambdas" / "webhook-handler").mkdir(parents=True)
    (tmp_path / "lambdas" / "webhook-handler" / "pyproject.toml").write_text("[project]", encoding="utf-8")

    result = discover([_repo(tmp_path, name="misc")], aliases=AliasTable.load())
    names = {n.name for n in result.nodes}
    assert {"api-gateway", "scheduler-svc", "webhook-handler"} <= names


def test_survives_missing_repo_path(tmp_path: Path) -> None:
    repo = RepoEntry(name="gone", local_path=tmp_path / "nope",
                     config_roots=(), pipelines=())
    result = discover([repo], aliases=AliasTable.load())
    assert result.nodes == []
    assert result.modules_scanned == 0


def test_module_metadata_carries_repo_and_path(tmp_path: Path) -> None:
    (tmp_path / "source" / "widget-gen").mkdir(parents=True)
    (tmp_path / "source" / "widget-gen" / "pom.xml").write_text("<project/>", encoding="utf-8")
    result = discover([_repo(tmp_path, name="widget-repo")], aliases=AliasTable.load())
    stage = next(n for n in result.nodes if n.kind == "stage" and n.name == "widget-gen")
    assert stage.metadata["repo"] == "widget-repo"
    assert stage.metadata["module_path"] == "source/widget-gen"
    assert stage.metadata.get("discovered_via") == "module_layout"
