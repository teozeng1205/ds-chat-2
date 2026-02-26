from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.investigation.runtime import (
    EntityResolver,
    KnowledgeBase,
    LocalCodeCatalog,
    OperatorRuntime,
    SqlGuard,
    WorkspaceManager,
)


class _FakeRegistry:
    def __init__(self, providers: list[str] | None = None) -> None:
        self.providers = providers or []
        self.seen_tokens: list[str] = []

    def mysql_lookup_codes(self, tokens: list[str]):
        self.seen_tokens = list(tokens)
        return {
            "providers": self.providers,
            "sites": [],
            "customers": [],
        }


class _FakeKBRegistry:
    def inspect_table_metadata(self, table_name: str, datasource: str):
        del datasource
        return {
            "table_name": table_name,
            "columns": [
                {"column_name": "sales_date", "data_type": "bigint", "nullable": False, "is_key": False},
                {"column_name": "customer", "data_type": "varchar", "nullable": True, "is_key": False},
            ],
        }

    def execute_sql(self, datasource: str, query: str):
        del datasource
        del query
        return pd.DataFrame([{"sales_date": 20260226, "customer": "B6"}])


def _seed_codes(path: Path) -> None:
    payload = {
        "providers": ["QL2"],
        "sites": ["AV"],
        "customers": ["B6"],
        "customer_sites": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_entity_resolver_uses_local_codes_then_mysql_fallback(tmp_path: Path):
    codes_path = tmp_path / "common_codes.json"
    _seed_codes(codes_path)

    catalog = LocalCodeCatalog(path=codes_path)
    registry = _FakeRegistry(providers=["NEWP"])
    resolver = EntityResolver(catalog=catalog, registry=registry)

    result = resolver.resolve("top site issues for NEWP and site AV and customer B6")

    assert "AV" in result["sites"]
    assert "B6" in result["customers"]
    assert "NEWP" in result["providers"]
    assert "NEWP" in registry.seen_tokens


def test_sql_guard_allows_readonly_without_partition_requirement():
    guard = SqlGuard()
    query = (
        "SELECT observation_date, impact_score "
        "FROM prod.analytics.market_level_anomalies_v3"
    )

    validated = guard.validate(query)

    assert validated.upper().startswith("SELECT")
    assert "LIMIT" in validated.upper()
    assert validated.strip().endswith(";")


def test_sql_guard_blocks_non_readonly():
    guard = SqlGuard()
    with pytest.raises(ValueError, match="Only SELECT/WITH"):
        guard.validate("DELETE FROM prod.analytics.market_level_anomalies_v3")


def test_workspace_cleanup_retains_manifest(tmp_path: Path):
    workspace = WorkspaceManager(root=tmp_path / "sessions")
    thread_id = "thread-1"
    run_id = workspace.start_run(thread_id)

    frame = pd.DataFrame([{"a": 1}, {"a": 2}])
    record = workspace.save_dataset(
        thread_id=thread_id,
        run_id=run_id,
        df=frame,
        source_metadata={"type": "sql", "query_hash": "abc"},
    )
    assert Path(record["local_path"]).exists()

    cleanup = workspace.cleanup_thread(thread_id, mode="ephemeral_manifest")
    assert cleanup["manifest_retained"] == 1
    assert cleanup["deleted_files"] >= 1
    assert not Path(record["local_path"]).exists()


def test_knowledge_base_parses_tables_doc_and_refreshes(tmp_path: Path):
    source_tables = Path(__file__).resolve().parents[3] / "tables.md"
    assert source_tables.exists()

    codes_path = tmp_path / "common_codes.json"
    _seed_codes(codes_path)
    catalog = LocalCodeCatalog(path=codes_path)

    db_path = tmp_path / "knowledge.sqlite"
    kb = KnowledgeBase(db_path=db_path, catalog=catalog, registry=_FakeKBRegistry())

    result = kb.refresh(force=True)
    assert result["ok"] is True
    assert result["refreshed"] is True


def test_operator_runtime_can_save_new_dataset(tmp_path: Path):
    workspace = WorkspaceManager(root=tmp_path / "sessions")
    thread_id = "thread-op"
    run_id = workspace.start_run(thread_id)
    base = workspace.save_dataset(
        thread_id=thread_id,
        run_id=run_id,
        df=pd.DataFrame([{"x": 1}, {"x": 2}, {"x": 3}]),
        source_metadata={"type": "seed"},
        dataset_name="seed",
    )

    operator = OperatorRuntime(workspace)
    code = (
        "df = load_dataset('" + base["dataset_id"] + "')\n"
        "out = df.assign(x2=df['x'] * 2)\n"
        "save_dataframe(out, 'derived', {'type': 'python'})\n"
        "print('rows', len(out))\n"
    )

    result = operator.run_python(thread_id=thread_id, run_id=run_id, code=code)
    assert result["ok"] is True
    assert "rows 3" in result["stdout"]
    assert len(result["created_datasets"]) == 1
