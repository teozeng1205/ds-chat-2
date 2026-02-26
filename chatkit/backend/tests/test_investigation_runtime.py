from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.investigation.catalog import KnowledgeBase, LocalCodeCatalog
from app.investigation.datasources import DatasourceRegistry
from app.investigation.entity_resolution import EntityResolver
from app.investigation.executor import OperatorRuntime, SqlGuard
from app.investigation.planner import AutonomousInvestigationEngine
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


class _FakeRuntime:
    def __init__(self) -> None:
        self.entities = {"providers": ["QL2"], "sites": ["AV"], "customers": ["B6"]}
        self.datasets: list[dict[str, str]] = []

    def resolve_entities(self, input_text: str, sales_date_hint: str | None = None):
        del input_text
        del sales_date_hint
        return dict(self.entities)

    def retrieve_knowledge(self, *, query: str, entities: dict[str, str], top_k: int = 8):
        del query
        del entities
        del top_k
        return {"candidate_tables": ["prod.monitoring.combined_audit"], "task_cards": []}

    def inspect_table_metadata(self, table_name: str, datasource: str | None = None, capture_example_row: bool = True):
        del datasource
        del capture_example_row
        return {"table_name": table_name, "columns": [{"column_name": "sales_date", "data_type": "bigint", "nullable": False}]}

    def extract_sql_to_dataset(self, *, thread_id: str, query: str, datasource: str, run_id: str, metadata: dict | None, dataset_name: str | None):
        del thread_id
        del datasource
        del run_id
        record = {
            "dataset_id": dataset_name or "dataset",
            "row_count": 2,
            "source_metadata": {"query": query, **(metadata or {})},
        }
        self.datasets.append(record)
        return record

    def extract_s3_to_dataset(self, *, thread_id: str, bucket: str, key_or_prefix: str, run_id: str, metadata: dict | None, dataset_name: str | None):
        del thread_id
        del bucket
        del key_or_prefix
        del run_id
        del metadata
        record = {"dataset_id": dataset_name or "s3", "row_count": 3, "source_metadata": {}}
        self.datasets.append(record)
        return record

    def run_dataframe_analysis(self, *, thread_id: str, run_id: str, dataset_ids: list[str], analysis_spec: dict):
        del thread_id
        del run_id
        return {
            "analysis_id": "analysis_1",
            "summary_stats": {"dataset_count": len(dataset_ids)},
            "report_markdown": "analysis",
            "caveats": [] if analysis_spec.get("mode") != "profile_dataset" else ["sampled"],
        }


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
    query = "SELECT observation_date, impact_score FROM prod.analytics.market_level_anomalies_v3"

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


def test_autonomous_engine_runs_table_eda_flow():
    runtime = _FakeRuntime()
    engine = AutonomousInvestigationEngine(runtime)

    result = engine.run(
        thread_id="thread-a",
        run_id="run-a",
        question="can you do a EDA of the table combined_audit",
        sales_date="20260226",
        constraints={},
    )

    assert result["strategy"] == "autonomous_general"
    assert len(result["datasets"]) >= 1
    assert result["analysis"] is not None


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
    registry = DatasourceRegistry(default_profile="3VDEV")
    result = registry.ensure_credentials()

    assert result["ok"] is True
    assert calls
    assert calls[0][0] == "zsh"
    assert "assume 3VDEV" in calls[0][-1]
