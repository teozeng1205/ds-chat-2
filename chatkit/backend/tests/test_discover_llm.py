"""Unit tests for app.pipelines.discover_llm (Pass 4).

Uses a fake OpenAI client so no real API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.pipelines.canonicalize import AliasTable, RepoEntry, node_id
from app.pipelines.discover_llm import discover


class _FakeResponsesClient:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.calls = 0

    @property
    def responses(self):
        outer = self

        class _R:
            def create(self, **kwargs):
                outer.calls += 1
                class _Resp:
                    output_text = json.dumps(outer._payload)
                return _Resp()

        return _R()


def _make_repo(tmp_path: Path, name: str, *, readme: str | None = None) -> RepoEntry:
    d = tmp_path / name
    d.mkdir()
    if readme:
        (d / "README.md").write_text(readme, encoding="utf-8")
    (d / "src").mkdir()
    (d / "src" / "main.py").write_text("# noop", encoding="utf-8")
    return RepoEntry(
        name=name, local_path=d, config_roots=(), pipelines=(),
    )


def test_discover_emits_stage_and_reads_writes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "widget-svc", readme="Widget service.")
    payload = {
        "apps": [{
            "name": "market-level-generator",
            "inputs": ["analytics.derived_common_output"],
            "outputs": ["s3://bucket-x/v4", "analytics.market_level_anomalies_v4"],
            "triggers": ["cron(0 * * * ? *)"],
            "purpose": "Generates anomalies"
        }]
    }
    client = _FakeResponsesClient(payload)
    result = discover(
        repos=[repo], aliases=AliasTable.load(),
        client_factory=lambda: client,
        cache_dir=tmp_path / "cache",
    )
    assert result.repos_with_signal == 1
    assert any(e.rel == "reads" for e in result.edges)
    assert any(e.rel == "writes" for e in result.edges)
    # Stage node created
    assert any(
        n.id == node_id("stage", "market-level-generator") for n in result.nodes
    )


def test_discover_uses_cache_on_second_call(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "alpha", readme="Alpha pipeline")
    payload = {"apps": [{"name": "alpha-stage", "inputs": [], "outputs": [], "triggers": []}]}
    client = _FakeResponsesClient(payload)
    cache_dir = tmp_path / "cache"

    discover(repos=[repo], aliases=AliasTable.load(),
             client_factory=lambda: client, cache_dir=cache_dir)
    discover(repos=[repo], aliases=AliasTable.load(),
             client_factory=lambda: client, cache_dir=cache_dir)
    # Two runs but exactly one LLM call thanks to the content-hash cache
    assert client.calls == 1


def test_discover_handles_bad_json_gracefully(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "beta", readme="Beta.")

    class _Bad:
        calls = 0

        @property
        def responses(self):
            class _R:
                def create(_self, **kwargs):
                    class _Resp:
                        output_text = "not json at all — nothing here"
                    return _Resp()
            return _R()

    result = discover(
        repos=[repo], aliases=AliasTable.load(),
        client_factory=lambda: _Bad(),
        cache_dir=tmp_path / "cache",
    )
    assert result.repos_with_signal == 0
    assert result.nodes == [] and result.edges == []


def test_discover_skips_missing_repo(tmp_path: Path) -> None:
    repo = RepoEntry(name="ghost", local_path=tmp_path / "does-not-exist",
                     config_roots=(), pipelines=())
    result = discover(
        repos=[repo], aliases=AliasTable.load(),
        client_factory=lambda: _FakeResponsesClient({"apps": []}),
        cache_dir=tmp_path / "cache",
    )
    assert result.nodes == []


def test_discover_respects_max_repos(tmp_path: Path) -> None:
    repos = [_make_repo(tmp_path, f"r{i}", readme=f"r{i}") for i in range(5)]
    client = _FakeResponsesClient({"apps": []})
    result = discover(
        repos=repos, aliases=AliasTable.load(),
        client_factory=lambda: client,
        cache_dir=tmp_path / "cache",
        max_repos=2,
    )
    assert result.repos_scanned == 2
