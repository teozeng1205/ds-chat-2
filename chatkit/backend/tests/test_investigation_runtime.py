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


def test_datasource_for_table_federated_schemas():
    """Federated MySQL-via-Redshift queries must reach a Redshift cluster.

    The cascading investigation skill tells the agent to fall through to
    `federated_priceeye.*` for config-drop diagnostics when a prod table is
    empty for a specific customer. Those queries MUST route to a Redshift
    reader (which knows how to execute federated SQL), not to the MySQL
    reader — the MySQL connector has no federation bridge.
    """
    # priceeye federated tables → redshift_analytics (default fall-through)
    assert datasource_for_table("federated_priceeye.customer_defaults") == "redshift_analytics"
    assert datasource_for_table("federated_priceeye.site_hierarchy") == "redshift_analytics"
    # metadata federated tables → redshift_analytics
    assert datasource_for_table("federated_metadata.airportlocation_extra") == "redshift_analytics"
    # scheduling federated tables → redshift_core (only cluster that has them)
    assert datasource_for_table("federated_scheduling.some_table") == "redshift_core"
    # Full-query strings work too (runtime passes raw SQL into this fn)
    assert datasource_for_table(
        "SELECT * FROM federated_scheduling.foo WHERE a = 1"
    ) == "redshift_analytics"  # startswith on raw SELECT falls through to default
    # Case-insensitive + local.federated_* keeps dev isolated on core
    assert datasource_for_table("local.federated_priceeye.customer_defaults") == "redshift_core"


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


def test_workspace_save_dataset_handles_duplicate_column_names(tmp_path: Path):
    workspace = WorkspaceManager(root=tmp_path / "sessions")
    thread_id = "thread-duplicate-types"
    run_id = workspace.start_run(thread_id)

    frame = pd.DataFrame([[1, "x"], [2, "y"]], columns=["a", "a"])
    record = workspace.save_dataset(
        thread_id=thread_id,
        run_id=run_id,
        df=frame,
        source_metadata={"type": "test"},
    )

    assert record["columns"] == ["a", "a"]
    assert record["column_types"] == {"a": "int64", "a#2": "object"}


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


def test_datasource_registry_runs_export_credentials_then_falls_back_to_assume(
    monkeypatch: pytest.MonkeyPatch,
):
    """The bootstrap has two paths — first `aws configure export-credentials`
    (text=True, str output), and if that loads nothing, `assume 3VDEV; env -0`
    (text=False, bytes output). The mock must honor the `text` kwarg so
    `line.startswith("export ")` gets a str, not bytes.
    """
    # Make sure no ambient AWS creds short-circuit the bootstrap.
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
              "AWS_DEFAULT_PROFILE", "AWS_CREDENTIAL_EXPIRATION"):
        monkeypatch.delenv(k, raising=False)

    calls: list[tuple[list[str], bool]] = []

    class _Proc:
        def __init__(self, *, returncode: int, stdout):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = b"" if isinstance(stdout, bytes) else ""

    def _fake_run(cmd, _capture_output=False, text=False, **_kw):
        calls.append((cmd, bool(text)))
        # First call: export-credentials with text=True → return str, empty
        # stdout so the primary path "loads 0" and we fall through.
        if text and cmd[0] == "zsh" and "export-credentials" in cmd[-1]:
            return _Proc(returncode=0, stdout="")
        # Second call: assume 3VDEV via zsh with text=False → return bytes with
        # NUL-separated KEY=VALUE pairs that the fallback parses.
        if not text and cmd[0] == "zsh" and "assume 3VDEV" in cmd[-1]:
            payload = (
                b"AWS_ACCESS_KEY_ID=abc\x00"
                b"AWS_SECRET_ACCESS_KEY=def\x00"
                b"AWS_SESSION_TOKEN=ghi\x00"
            )
            return _Proc(returncode=0, stdout=payload)
        # Anything else (sts get-caller-identity etc.) — succeed empty.
        return _Proc(returncode=0, stdout="" if text else b"")

    monkeypatch.setattr("subprocess.run", _fake_run)
    registry = DatasourceRegistry()
    result = registry.ensure_credentials()

    assert result["ok"] is True
    # Both the primary and fallback paths ran in order.
    assert any("export-credentials" in c[0][-1] for c in calls)
    assert any("assume 3VDEV" in c[0][-1] for c in calls)
