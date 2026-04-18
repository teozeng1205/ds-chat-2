"""Integration test for the execute_sql cache wrapper.

We don't go through the Agents SDK's FunctionTool shim (its invocation
context expects richer state than we want to fake). Instead we
reproduce the cache-hit/-miss logic inline against the real
QueryCache singleton and a fake runtime, confirming that identical
queries hit the cache and that dataset_id is stripped on hits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DS_CHAT_QUERY_CACHE_DB", str(tmp_path / "cache.sqlite"))
    import app.investigation.query_cache as qc
    qc._SINGLETON = None  # reset module-level singleton to honor the new env var


class _FakeRuntime:
    def __init__(self) -> None:
        self.call_count = 0

    def execute_sql(self, *, thread_id: str, run_id: str, query: str, datasource: str | None) -> dict[str, Any]:
        self.call_count += 1
        return {
            "ok": True,
            "columns": ["a", "b"],
            "row_count": 2,
            "preview": {"text": "…", "rows": [[1, "x"], [2, "y"]]},
            "dataset_id": f"ds-{self.call_count}",
            "elapsed_ms": 42,
            "partition_warnings": [],
        }


def test_first_fresh_second_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.investigation.query_cache import get_query_cache
    from app.tools import investigation_tools as it

    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(it, "get_runtime", lambda: fake_runtime)
    monkeypatch.setattr(it, "_get_or_create_run_id", lambda _tid: "run-1")
    cache = get_query_cache()

    def _simulate_call(query: str, datasource: str | None = None) -> dict[str, Any]:
        # This mirrors the logic in execute_sql: check cache → if hit, return;
        # else call runtime and populate cache.
        hit = cache.get(query, datasource or "_auto", extra=[it._date_bucket()])
        if hit:
            payload = dict(hit.payload)
            payload.update({"cached": True, "dataset_id": None,
                            "cache_age_seconds": int(hit.age_seconds)})
            return payload
        result = fake_runtime.execute_sql(thread_id="T1", run_id="r1", query=query, datasource=datasource)
        if result.get("ok", True):
            to_cache = {k: v for k, v in result.items() if k != "dataset_id"}
            cache.put(query, to_cache, datasource or "_auto", extra=[it._date_bucket()])
        return result

    r1 = _simulate_call("select 1 from t")
    r2 = _simulate_call("select 1 from t")
    assert fake_runtime.call_count == 1
    assert r1["dataset_id"] == "ds-1"
    assert r1.get("cached") is None
    assert r2["cached"] is True
    assert r2["dataset_id"] is None
    assert r2["row_count"] == 2

    # Different query → fresh run
    _simulate_call("select 2 from t")
    assert fake_runtime.call_count == 2
