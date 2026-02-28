from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.investigation.catalog import KnowledgeBase, LocalCodeCatalog
from app.investigation.datasources import DatasourceRegistry, datasource_for_table
from app.investigation.entity_resolution import EntityResolver
from app.investigation.executor import OperatorRuntime, PartitionGuard, SqlGuard
from app.investigation.workspace import WorkspaceManager


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


def _seed_codes(path: Path) -> None:
    payload = {
        "providers": ["QL2"],
        "sites": ["AV"],
        "customers": ["B6"],
        "customer_sites": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_codes_v2(path: Path) -> None:
    payload = {
        "providers": [
            {"code": "QL2", "name": "QL2 Software", "aliases": ["QL2"]},
            {"code": "AA", "name": "American Airlines", "aliases": ["American", "AAL"]},
        ],
        "sites": [
            {"code": "AV", "name": "Aviasales", "aliases": []},
        ],
        "customers": [
            {"code": "B6", "name": "JetBlue Airways", "aliases": ["JetBlue", "Jet Blue"]},
        ],
        "customer_sites": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# ── Entity Resolution Tests ──


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


def test_entity_resolution_pipe_separator(tmp_path: Path):
    codes_path = tmp_path / "common_codes.json"
    _seed_codes(codes_path)

    catalog = LocalCodeCatalog(path=codes_path)
    registry = _FakeRegistry()
    resolver = EntityResolver(catalog=catalog, registry=registry)

    result = resolver.resolve("QL2|AV")

    assert "QL2" in result["providers"]
    assert "AV" in result["sites"]


def test_entity_resolution_mysql_cache(tmp_path: Path):
    codes_path = tmp_path / "common_codes.json"
    _seed_codes(codes_path)

    catalog = LocalCodeCatalog(path=codes_path)
    registry = _FakeRegistry(providers=["NEWP"])
    resolver = EntityResolver(catalog=catalog, registry=registry)

    # First call
    resolver.resolve("NEWP test")
    assert len(registry.seen_tokens) > 0

    # Reset seen_tokens and call again - should use cache
    registry.seen_tokens = []
    resolver.resolve("NEWP test")
    assert len(registry.seen_tokens) == 0  # cache hit, no MySQL call


# ── Common Codes V2 Alias Tests ──


def test_common_codes_v2_aliases(tmp_path: Path):
    codes_path = tmp_path / "common_codes.json"
    _seed_codes_v2(codes_path)

    catalog = LocalCodeCatalog(path=codes_path)
    assert catalog.resolve("JetBlue") == ("customer", "B6")
    assert catalog.resolve("Jet Blue") == ("customer", "B6")
    assert catalog.resolve("American") == ("provider", "AA")
    assert catalog.resolve("AAL") == ("provider", "AA")
    assert catalog.resolve("QL2") == ("provider", "QL2")


# ── SQL Guard Tests ──


def test_sql_guard_allows_readonly():
    guard = SqlGuard()
    query = "SELECT observation_date, impact_score FROM prod.analytics.market_level_anomalies_v3"

    validated = guard.validate(query)

    assert validated.upper().startswith("SELECT")
    assert "LIMIT" in validated.upper()
    assert validated.strip().endswith(";")


def test_sql_guard_blocks_non_readonly():
    guard = SqlGuard()
    with pytest.raises(ValueError, match="Only SELECT/WITH"):
        guard.validate("DELETE FROM prod.analytics.market_level_anomalies_v3")


# ── Partition Guard Tests ──


def test_partition_guard_warns_missing_sales_date():
    query = "SELECT * FROM prod.monitoring.provider_combined_audit WHERE providercode = 'QL2'"
    warnings = PartitionGuard.check(query)
    assert len(warnings) >= 1
    assert "sales_date" in warnings[0].lower()


def test_partition_guard_no_warning_when_present():
    query = "SELECT * FROM prod.monitoring.provider_combined_audit WHERE sales_date = 20260226 AND providercode = 'QL2'"
    warnings = PartitionGuard.check(query)
    assert len(warnings) == 0


def test_partition_guard_warns_missing_customer_for_analytics():
    query = "SELECT * FROM analytics.market_level_anomalies_v3 WHERE sales_date = 20260226"
    warnings = PartitionGuard.check(query)
    assert len(warnings) >= 1
    assert "customer" in warnings[0].lower()


def test_partition_guard_no_warning_for_unknown_table():
    query = "SELECT * FROM some.unknown_table"
    warnings = PartitionGuard.check(query)
    assert len(warnings) == 0


# ── Datasource Routing Tests ──


def test_datasource_for_table_routing():
    assert datasource_for_table("prod.monitoring.provider_combined_audit") == "redshift_core"
    assert datasource_for_table("prod.monitoring.combined_audit") == "redshift_core"
    assert datasource_for_table("analytics.market_level_anomalies_v3") == "redshift_analytics"
    assert datasource_for_table("priceeye.customer_defaults") == "mysql_priceeye"
    assert datasource_for_table("collection_optimizer.delta_swia_input_v1") == "redshift_core"


# ── Workspace Tests ──


def test_workspace_save_dataset_includes_column_types(tmp_path: Path):
    workspace = WorkspaceManager(root=tmp_path / "sessions")
    thread_id = "thread-types"
    run_id = workspace.start_run(thread_id)

    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    record = workspace.save_dataset(
        thread_id=thread_id,
        run_id=run_id,
        df=frame,
        source_metadata={"type": "test"},
    )

    assert "column_types" in record
    assert record["column_types"]["a"] == "int64"
    assert record["column_types"]["b"] == "object"


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


# ── Knowledge Base Tests ──


def test_knowledge_index_parses_tables_doc_and_refreshes(tmp_path: Path):
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir(parents=True, exist_ok=True)

    (knowledge_root / "tables.md").write_text(
        "| Table | Notes |\n|---|---|\n| `prod.monitoring.combined_audit` | core monitoring table |\n",
        encoding="utf-8",
    )
    codes_path = knowledge_root / "common_codes.json"
    _seed_codes(codes_path)

    catalog = LocalCodeCatalog(path=codes_path)
    db_path = tmp_path / "knowledge.sqlite"
    kb = KnowledgeBase(root=knowledge_root, db_path=db_path)

    result = kb.refresh(force=True, catalog=catalog)
    assert result["ok"] is True
    assert result["refreshed"] is True
    retrieved = kb.retrieve(question="combined audit", entities={})
    assert "prod.monitoring.combined_audit" in retrieved["candidate_tables"]


def test_knowledge_retrieval_includes_partition_info(tmp_path: Path):
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir(parents=True, exist_ok=True)

    (knowledge_root / "tables.md").write_text(
        "| Table | Notes |\n|---|---|\n| `prod.monitoring.combined_audit` | core monitoring table |\n",
        encoding="utf-8",
    )
    # Write live metadata with partitions
    live_meta = {
        "tables": [
            {
                "table_name": "prod.monitoring.combined_audit",
                "datasource": "redshift_core",
                "status": "ok",
                "partitions": [{"column": "sales_date", "role": "recommended", "inferred_type": "date"}],
                "columns": [{"column_name": "sales_date", "data_type": "bigint"}],
            }
        ]
    }
    (knowledge_root / "common_table_live_metadata.json").write_text(json.dumps(live_meta), encoding="utf-8")
    codes_path = knowledge_root / "common_codes.json"
    _seed_codes(codes_path)

    catalog = LocalCodeCatalog(path=codes_path)
    db_path = tmp_path / "knowledge.sqlite"
    kb = KnowledgeBase(root=knowledge_root, db_path=db_path)
    kb.refresh(force=True, catalog=catalog)

    retrieved = kb.retrieve(question="combined audit", entities={})
    table_hints = retrieved.get("table_hints", [])
    assert len(table_hints) > 0
    first_hint = table_hints[0]
    assert "partitions" in first_hint
    assert len(first_hint["partitions"]) > 0


# ── Operator Runtime Tests ──


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


# ── Datasource Registry Tests ──


def test_datasource_registry_runs_assume_3vdev(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = b"AWS_ACCESS_KEY_ID=abc\x00AWS_SECRET_ACCESS_KEY=def\x00AWS_SESSION_TOKEN=ghi\x00"
        stderr = b""

    def _fake_run(cmd, capture_output, text):
        del capture_output
        del text
        calls.append(cmd)
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    registry = DatasourceRegistry()
    result = registry.ensure_credentials()

    assert result["ok"] is True
    assert calls
    assert calls[0][0] == "zsh"
    assert "assume 3VDEV" in calls[0][-1]
