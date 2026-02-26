"""Autonomous shell-first investigation runtime for DS Chat."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import DataAnalyzer
from .catalog import KnowledgeBase, LocalCodeCatalog
from .datasources import DatasourceRegistry
from .entity_resolution import EntityResolver
from .executor import OperatorRuntime, SqlGuard
from .planner import AutonomousInvestigationEngine
from .reporting import build_lineage, summarize_answer
from .workspace import WorkspaceManager

log = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent.parent
WORK_ROOT = BACKEND_ROOT / ".work"
SESSION_ROOT = WORK_ROOT / "sessions"
KB_RUNTIME_ROOT = WORK_ROOT / "knowledge"
KB_DB_PATH = KB_RUNTIME_ROOT / "knowledge.sqlite"

INVESTIGATION_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_ROOT = INVESTIGATION_ROOT / "knowledge"
COMMON_CODES_PATH = KNOWLEDGE_ROOT / "common_codes.json"
SQL_BEST_PRACTICES_PATH = KNOWLEDGE_ROOT / "sql_best_practices.md"
TABLES_DOC_PATH = KNOWLEDGE_ROOT / "tables.md"

DEFAULT_SQL_LIMIT = 1000
MAX_SQL_LIMIT = 120000


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _mask_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) <= 3:
        return "***"
    return f"{text[:2]}***{text[-2:]}"


def _mask_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _mask_value(value) for key, value in row.items()}


def _coerce_sales_date(value: str | dt.date | dt.datetime | None) -> str:
    if value is None:
        return dt.date.today().strftime("%Y%m%d")
    if isinstance(value, dt.datetime):
        return value.date().strftime("%Y%m%d")
    if isinstance(value, dt.date):
        return value.strftime("%Y%m%d")
    raw = str(value).strip().lower()
    if raw in {"today", "now"}:
        return dt.date.today().strftime("%Y%m%d")
    if raw == "yesterday":
        return (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
    if raw == "tomorrow":
        return (dt.date.today() + dt.timedelta(days=1)).strftime("%Y%m%d")
    if len(raw) == 8 and raw.isdigit():
        return raw
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return dt.datetime.strptime(raw, "%Y-%m-%d").strftime("%Y%m%d")
    raise ValueError(f"Unsupported sales_date format: {value!r}")


def _canonical_table_name(table_name: str) -> str:
    return re.sub(r"\blocal\.monitoring\.", "prod.monitoring.", table_name.strip(), flags=re.I)


def _canonicalize_sql(query: str) -> str:
    return re.sub(r"\blocal\.monitoring\.", "prod.monitoring.", query, flags=re.I)


class InvestigationRuntime:
    """Coordinator for autonomous data investigation tasks."""

    def __init__(self) -> None:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_ROOT.mkdir(parents=True, exist_ok=True)
        KB_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

        self.registry = DatasourceRegistry(default_profile="3VDEV")
        self.catalog = LocalCodeCatalog(path=COMMON_CODES_PATH)
        self.kb = KnowledgeBase(root=KNOWLEDGE_ROOT, db_path=KB_DB_PATH)
        self.resolver = EntityResolver(catalog=self.catalog, registry=self.registry)
        self.guard = SqlGuard(default_limit=DEFAULT_SQL_LIMIT, max_limit=MAX_SQL_LIMIT)
        self.workspace = WorkspaceManager(root=SESSION_ROOT)
        self.operator = OperatorRuntime(self.workspace)
        self.analyzer = DataAnalyzer()
        self.max_steps = int(os.getenv("INVESTIGATION_MAX_STEPS", "20"))
        self.engine = AutonomousInvestigationEngine(self, max_steps=self.max_steps)
        self.approval_policy = os.getenv("INVESTIGATION_APPROVAL_POLICY", "never")
        self.sandbox_mode = os.getenv("INVESTIGATION_SANDBOX_MODE", "workspace-write")

    @staticmethod
    def _mirror_log_path(thread_id: str, run_id: str) -> Path:
        return Path("/tmp") / "ds-chat-investigation" / thread_id / run_id / "activity.jsonl"

    def _log_event(self, *, thread_id: str, run_id: str, event: str, payload: dict[str, Any]) -> None:
        self.workspace.append_event(thread_id, run_id, event, payload)

        mirror = self._mirror_log_path(thread_id, run_id)
        mirror.parent.mkdir(parents=True, exist_ok=True)
        with mirror.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": _iso_now(), "event": event, "payload": payload}, ensure_ascii=True, default=_json_default) + "\n")

    def ensure_kb_ready(self) -> dict[str, Any]:
        return self.kb.refresh(force=False, catalog=self.catalog)

    def refresh_knowledge_base(self, force: bool = True) -> dict[str, Any]:
        return self.kb.refresh(force=force, catalog=self.catalog)

    def browse_knowledge_files(self, path_or_glob: str) -> dict[str, Any]:
        result = self.kb.browse_files(path_or_glob)
        return result

    def resolve_entities(self, input_text: str, sales_date_hint: str | None = None) -> dict[str, Any]:
        return self.resolver.resolve(input_text, sales_date_hint=sales_date_hint)

    def retrieve_knowledge(self, *, query: str, entities: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
        return self.kb.retrieve(question=query, entities=entities, top_k=top_k)

    def inspect_table_metadata(
        self,
        table_name: str,
        datasource: str | None = None,
        capture_example_row: bool = True,
    ) -> dict[str, Any]:
        canonical_table_name = _canonical_table_name(table_name)
        source = datasource or self._datasource_for_table(canonical_table_name)
        metadata = self.registry.inspect_table_metadata(canonical_table_name, source)

        sample_row_masked: dict[str, Any] | None = None
        if capture_example_row:
            try:
                sample = self.registry.execute_sql(source, f"SELECT * FROM {canonical_table_name} LIMIT 1")
                if not sample.empty:
                    sample_row_masked = _mask_row(sample.iloc[0].to_dict())
            except Exception:
                sample_row_masked = None

        knowledge = self.kb.retrieve(question=canonical_table_name, entities={})
        tier = "common" if canonical_table_name in knowledge.get("candidate_tables", []) else "discovered"
        self.kb.upsert_table_metadata(
            table_name=canonical_table_name,
            datasource=source,
            columns=metadata.get("columns", []),
            partitions=metadata.get("partitions", []),
            sample_row_masked=sample_row_masked,
            tier=tier,
            notes="Discovered from metadata inspection" if tier == "discovered" else "Common table",
        )

        return {
            **metadata,
            "table_name": canonical_table_name,
            "sample_row_masked": sample_row_masked,
            "tier": tier,
        }

    @staticmethod
    def _datasource_for_table(table_name: str) -> str:
        normalized = _canonical_table_name(table_name)
        if normalized.startswith("priceeye."):
            return "mysql_priceeye"
        if normalized.startswith("prod.monitoring"):
            return "redshift_core"
        return "redshift_analytics"

    def extract_sql_to_dataset(
        self,
        *,
        thread_id: str,
        query: str,
        datasource: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        effective_run_id = run_id or self.workspace.start_run(thread_id)
        canonical_query = _canonicalize_sql(query)
        validated_query = self.guard.validate(canonical_query)

        started = time.time()
        frame = self.registry.execute_sql(datasource, validated_query)
        elapsed_ms = int((time.time() - started) * 1000)

        source_metadata = dict(metadata or {})
        source_metadata.update(
            {
                "type": "sql",
                "datasource": datasource,
                "query": validated_query,
                "query_hash": hex(abs(hash(validated_query)))[2:],
                "elapsed_ms": elapsed_ms,
            }
        )

        record = self.workspace.save_dataset(
            thread_id=thread_id,
            run_id=effective_run_id,
            df=frame,
            source_metadata=source_metadata,
            dataset_name=dataset_name,
        )

        query_entry = {
            "datasource": datasource,
            "query": validated_query,
            "row_count": int(len(frame)),
            "elapsed_ms": elapsed_ms,
            "dataset_id": record["dataset_id"],
        }
        self.workspace.append_query(thread_id, effective_run_id, query_entry)
        self._log_event(thread_id=thread_id, run_id=effective_run_id, event="extract_sql", payload=query_entry)
        return record

    def extract_s3_to_dataset(
        self,
        *,
        thread_id: str,
        bucket: str,
        key_or_prefix: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        effective_run_id = run_id or self.workspace.start_run(thread_id)
        started = time.time()
        frame, keys = self.registry.fetch_s3_csv(bucket, key_or_prefix)
        elapsed_ms = int((time.time() - started) * 1000)

        source_metadata = dict(metadata or {})
        source_metadata.update(
            {
                "type": "s3",
                "bucket": bucket,
                "key_or_prefix": key_or_prefix,
                "keys": keys,
                "elapsed_ms": elapsed_ms,
            }
        )

        record = self.workspace.save_dataset(
            thread_id=thread_id,
            run_id=effective_run_id,
            df=frame,
            source_metadata=source_metadata,
            dataset_name=dataset_name,
        )

        source_entry = {
            "bucket": bucket,
            "key_or_prefix": key_or_prefix,
            "keys": keys,
            "row_count": int(len(frame)),
            "dataset_id": record["dataset_id"],
            "elapsed_ms": elapsed_ms,
        }
        self.workspace.append_source(thread_id, effective_run_id, source_entry)
        self._log_event(thread_id=thread_id, run_id=effective_run_id, event="extract_s3", payload=source_entry)
        return record

    def run_dataframe_analysis(
        self,
        *,
        thread_id: str,
        run_id: str,
        dataset_ids: list[str],
        analysis_spec: dict[str, Any],
    ) -> dict[str, Any]:
        frames = self.workspace.read_datasets(thread_id=thread_id, run_id=run_id, dataset_ids=dataset_ids)
        payload = self.analyzer.analyze(frames=frames, analysis_spec=analysis_spec)
        record = self.workspace.record_analysis(thread_id=thread_id, run_id=run_id, analysis_payload=payload)

        event = {
            "analysis_id": record["analysis_id"],
            "analysis_mode": payload.get("analysis_mode", "profile_dataset"),
            "dataset_ids": dataset_ids,
        }
        self._log_event(thread_id=thread_id, run_id=run_id, event="analysis_complete", payload=event)
        return {
            "analysis_id": record["analysis_id"],
            "local_path": record["local_path"],
            "results": payload.get("results", {}),
            "summary_stats": payload.get("summary_stats", {}),
            "report_markdown": payload.get("report_markdown", ""),
            "caveats": payload.get("caveats", []),
        }

    def operator_run_python(self, *, thread_id: str, code: str, run_id: str | None = None) -> dict[str, Any]:
        effective_run_id = run_id or self.workspace.start_run(thread_id)
        self._log_event(
            thread_id=thread_id,
            run_id=effective_run_id,
            event="operator_run_python_start",
            payload={"approval_policy": self.approval_policy, "sandbox_mode": self.sandbox_mode},
        )
        result = self.operator.run_python(thread_id=thread_id, run_id=effective_run_id, code=code)
        self._log_event(
            thread_id=thread_id,
            run_id=effective_run_id,
            event="operator_run_python_done",
            payload={"created_datasets": len(result.get("created_datasets", []))},
        )
        return result

    def run_table_eda(
        self,
        *,
        thread_id: str,
        table_name: str,
        datasource: str | None = None,
        constraints: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        effective_run_id = run_id or self.workspace.start_run(thread_id)
        question = f"can you do a EDA of the table {table_name}"
        result = self.investigate_issue(
            thread_id=thread_id,
            question=question,
            sales_date=None,
            constraints={**(constraints or {}), "table_name": table_name, "datasource": datasource},
            run_id=effective_run_id,
        )
        return result

    def cleanup_session_workspace(self, thread_id: str, mode: str = "ephemeral_manifest") -> dict[str, Any]:
        return self.workspace.cleanup_thread(thread_id=thread_id, mode=mode)

    @staticmethod
    def _should_retry_on_missing_key(error: Exception) -> bool:
        message = str(error).lower()
        return "nosuchkey" in message or "404" in message

    def investigate_issue(
        self,
        *,
        thread_id: str,
        question: str,
        sales_date: str | None = None,
        constraints: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_kb_ready()
        effective_run_id = run_id or self.workspace.start_run(thread_id)
        effective_sales_date = _coerce_sales_date(sales_date)

        self._log_event(
            thread_id=thread_id,
            run_id=effective_run_id,
            event="plan_start",
            payload={"question": question, "sales_date": effective_sales_date},
        )

        errors: list[dict[str, Any]] = []
        warnings: list[str] = []

        try:
            loop_result = self.engine.run(
                thread_id=thread_id,
                run_id=effective_run_id,
                question=question,
                sales_date=effective_sales_date,
                constraints=constraints or {},
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"error": type(exc).__name__, "message": str(exc)})
            loop_result = {
                "strategy": "autonomous",
                "sales_date": effective_sales_date,
                "entities": {},
                "knowledge": {},
                "datasets": [],
                "analysis": None,
                "warnings": [],
                "clarification": None,
            }

        datasets = list(loop_result.get("datasets", []))
        analysis = loop_result.get("analysis")
        warnings.extend(loop_result.get("warnings", []))
        clarification = loop_result.get("clarification")
        observations = list(loop_result.get("observations", []))

        for row in observations:
            if not isinstance(row, dict):
                continue
            self._log_event(
                thread_id=thread_id,
                run_id=effective_run_id,
                event="action_observation",
                payload=row,
            )

        answer = summarize_answer(
            question=question,
            strategy=str(loop_result.get("strategy", "autonomous")),
            datasets=datasets,
            analysis=analysis if isinstance(analysis, dict) else None,
            warnings=warnings,
            clarification=clarification,
        )

        lineage = build_lineage(
            run_id=effective_run_id,
            datasets=datasets,
            analysis=analysis if isinstance(analysis, dict) else None,
            warnings=warnings,
        )

        result = {
            "thread_id": thread_id,
            "run_id": effective_run_id,
            "strategy": loop_result.get("strategy", "autonomous"),
            "sales_date": loop_result.get("sales_date", effective_sales_date),
            "question": question,
            "entities": loop_result.get("entities", {}),
            "knowledge": loop_result.get("knowledge", {}),
            "datasets": datasets,
            "analysis": analysis,
            "warnings": warnings,
            "errors": errors,
            "answer": answer,
            "lineage": lineage,
            "partial_result": bool(errors),
            "clarification": clarification,
            "observations": observations,
            "autonomy_policy": {
                "approval_policy": self.approval_policy,
                "sandbox_mode": self.sandbox_mode,
            },
        }

        self._log_event(
            thread_id=thread_id,
            run_id=effective_run_id,
            event="investigation_complete",
            payload={
                "strategy": result["strategy"],
                "dataset_count": len(datasets),
                "error_count": len(errors),
            },
        )

        return result


def ensure_knowledge_layout() -> None:
    KNOWLEDGE_ROOT.mkdir(parents=True, exist_ok=True)
    if not TABLES_DOC_PATH.exists():
        source = REPO_ROOT / "tables.md"
        if source.exists():
            TABLES_DOC_PATH.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            TABLES_DOC_PATH.write_text("# Tables\n", encoding="utf-8")
    if not COMMON_CODES_PATH.exists():
        COMMON_CODES_PATH.write_text(
            json.dumps({"providers": [], "sites": [], "customers": [], "customer_sites": []}, indent=2),
            encoding="utf-8",
        )
    if not SQL_BEST_PRACTICES_PATH.exists():
        SQL_BEST_PRACTICES_PATH.write_text("# SQL Best Practices\n", encoding="utf-8")
    (KNOWLEDGE_ROOT / "task_cards").mkdir(parents=True, exist_ok=True)


_RUNTIME: InvestigationRuntime | None = None
_RUNTIME_LOCK = threading.RLock()


def get_runtime() -> InvestigationRuntime:
    global _RUNTIME
    ensure_knowledge_layout()
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = InvestigationRuntime()
        return _RUNTIME


def cleanup_thread_workspace(thread_id: str, mode: str = "ephemeral_manifest") -> dict[str, Any]:
    return get_runtime().cleanup_session_workspace(thread_id=thread_id, mode=mode)


def is_investigation_engine_enabled() -> bool:
    return _bool_env("INVESTIGATION_ENGINE_ENABLED", True)


__all__ = [
    "AutonomousInvestigationEngine",
    "DataAnalyzer",
    "DatasourceRegistry",
    "EntityResolver",
    "InvestigationRuntime",
    "KnowledgeBase",
    "LocalCodeCatalog",
    "OperatorRuntime",
    "SqlGuard",
    "WorkspaceManager",
    "cleanup_thread_workspace",
    "get_runtime",
    "is_investigation_engine_enabled",
]
