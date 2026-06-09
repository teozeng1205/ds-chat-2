"""Tests for the execute_sql partition hard-gate (§10.2 table-query protocol).

A query on a table with KNOWN partition keys that omits a predicate on any of
them must be rejected BEFORE execution (PartitionFilterRequired), never reaching
the datasource. With the predicate present, the query proceeds normally.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.investigation.executor import PartitionFilterRequired
from app.investigation.runtime import InvestigationRuntime


def test_execute_sql_blocks_missing_partition(monkeypatch) -> None:
    rt = InvestigationRuntime()
    calls = {"n": 0}

    def _fake_exec(datasource: str, query: str) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(rt.registry, "execute_sql", _fake_exec)

    with pytest.raises(PartitionFilterRequired) as excinfo:
        rt.execute_sql(
            thread_id="t-block",
            run_id="r-block",
            query="SELECT * FROM prod.monitoring.combined_audit",
        )
    # Blocked before touching the datasource, and the error names the key.
    assert calls["n"] == 0
    assert "sales_date" in str(excinfo.value)


def test_execute_sql_allows_with_partition(monkeypatch) -> None:
    rt = InvestigationRuntime()
    monkeypatch.setattr(
        rt.registry,
        "execute_sql",
        lambda datasource, query: pd.DataFrame({"providercode": ["QL2"], "n": [3]}),
    )

    run_id = rt.start_run("t-allow")
    result = rt.execute_sql(
        thread_id="t-allow",
        run_id=run_id,
        query="SELECT providercode, SUM(inputrequestid_count) AS n FROM prod.monitoring.combined_audit WHERE sales_date = 20260101 GROUP BY providercode",
    )
    assert result["row_count"] == 1
    assert result["partition_warnings"] == []


def test_execute_sql_allows_unpartitioned_table(monkeypatch) -> None:
    """Tables with no resolvable partition keys are unaffected and run normally."""
    rt = InvestigationRuntime()
    monkeypatch.setattr(
        rt.registry,
        "execute_sql",
        lambda datasource, query: pd.DataFrame({"site_code": ["AV"]}),
    )

    run_id = rt.start_run("t-dim")
    result = rt.execute_sql(
        thread_id="t-dim",
        run_id=run_id,
        query="SELECT site_code FROM priceeye.site",
    )
    assert result["row_count"] == 1
