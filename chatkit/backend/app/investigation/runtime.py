"""Thin service layer for DS Chat investigation tools."""

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

from .catalog import KnowledgeBase, LocalCodeCatalog
from .datasources import DatasourceRegistry, datasource_for_table
from .entity_resolution import EntityResolver
from .executor import OperatorRuntime, PartitionGuard, SqlGuard
from .workspace import WorkspaceManager

log = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent.parent


def _partition_check(validated_query: str) -> list[str]:
    """Resolve partition warnings via Glue first, static map as fallback.

    Partition keys come live from `aws glue get-table` when the catalog
    is reachable. When Glue can't answer (no creds, table not in Glue,
    any AWS error), fall back to the static hardcoded map so the guard
    never fails the query.
    """
    try:
        from .glue_catalog import get_default_catalog
        guard = PartitionGuard.from_glue(get_default_catalog())
        return guard.check_live(validated_query)
    except Exception as exc:  # noqa: BLE001 — never crash the query over the guard
        log.debug("Glue partition guard fell back to static map: %s", exc)
        return PartitionGuard.check(validated_query)
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


def _format_preview_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Return a compact markdown-style table string from a DataFrame preview.

    Produces output that is easy for the LLM to pattern-match on (structured
    plain text, not raw JSON).
    """
    if df.empty:
        return "(empty result set)"
    preview = df.head(max_rows)
    # Truncate long cell values to keep the table readable
    formatted = preview.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(
            lambda v: str(v) if len(str(v)) <= 60 else str(v)[:57] + "..."
        )
    try:
        table = formatted.to_markdown(index=False)
    except ImportError:
        # Fallback: build a simple pipe-delimited table without tabulate
        cols = list(formatted.columns)
        lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, row in formatted.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        table = "\n".join(lines)
    suffix = f"\n... ({len(df)} rows total)" if len(df) > max_rows else ""
    return (table or "(empty)") + suffix


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


class InvestigationRuntime:
    """Service layer providing tool implementations for the agentic investigation agent."""

    def __init__(self) -> None:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_ROOT.mkdir(parents=True, exist_ok=True)
        KB_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

        self.registry = DatasourceRegistry()
        self.catalog = LocalCodeCatalog(path=COMMON_CODES_PATH)
        self.kb = KnowledgeBase(root=KNOWLEDGE_ROOT, db_path=KB_DB_PATH)
        self.resolver = EntityResolver(catalog=self.catalog, registry=self.registry)
        self.guard = SqlGuard(default_limit=DEFAULT_SQL_LIMIT, max_limit=MAX_SQL_LIMIT)
        self.workspace = WorkspaceManager(root=SESSION_ROOT)
        self.operator = OperatorRuntime(self.workspace)

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

    # ── Tool implementation: execute_sql ──

    def execute_sql(
        self,
        *,
        thread_id: str,
        run_id: str,
        query: str,
        datasource: str | None = None,
    ) -> dict[str, Any]:
        """Execute read-only SQL, validate with guards, save result as dataset."""
        effective_datasource = datasource or datasource_for_table(query)
        validated_query = self.guard.validate(query)

        # Partition guard warnings — Glue catalog first, static map fallback.
        partition_warnings = _partition_check(validated_query)

        started = time.time()
        frame = self.registry.execute_sql(effective_datasource, validated_query)
        elapsed_ms = int((time.time() - started) * 1000)

        source_metadata = {
            "type": "sql",
            "datasource": effective_datasource,
            "query": validated_query,
            "query_hash": hex(abs(hash(validated_query)))[2:],
            "elapsed_ms": elapsed_ms,
        }

        record = self.workspace.save_dataset(
            thread_id=thread_id,
            run_id=run_id,
            df=frame,
            source_metadata=source_metadata,
        )

        query_entry = {
            "datasource": effective_datasource,
            "query": validated_query,
            "row_count": int(len(frame)),
            "elapsed_ms": elapsed_ms,
            "dataset_id": record["dataset_id"],
        }
        self.workspace.append_query(thread_id, run_id, query_entry)
        self._log_event(thread_id=thread_id, run_id=run_id, event="execute_sql", payload=query_entry)

        # Build summary for LLM observation
        columns = list(frame.columns)
        preview_rows = frame.head(20).to_dict(orient="records") if not frame.empty else []
        return {
            "dataset_id": record["dataset_id"],
            "row_count": int(len(frame)),
            "columns": columns,
            "column_types": {col: str(frame[col].dtype) for col in columns},
            "preview": preview_rows,
            "preview_text": _format_preview_table(frame),
            "elapsed_ms": elapsed_ms,
            "partition_warnings": partition_warnings,
        }

    # ── Tool implementation: fetch_s3 ──

    def fetch_s3(
        self,
        *,
        thread_id: str,
        run_id: str,
        bucket: str,
        key_or_prefix: str,
    ) -> dict[str, Any]:
        """Fetch S3 data and save as dataset."""
        started = time.time()
        frame, keys = self.registry.fetch_s3_data(bucket, key_or_prefix)
        elapsed_ms = int((time.time() - started) * 1000)

        source_metadata = {
            "type": "s3",
            "bucket": bucket,
            "key_or_prefix": key_or_prefix,
            "keys": keys,
            "elapsed_ms": elapsed_ms,
        }

        record = self.workspace.save_dataset(
            thread_id=thread_id,
            run_id=run_id,
            df=frame,
            source_metadata=source_metadata,
        )

        source_entry = {
            "bucket": bucket,
            "key_or_prefix": key_or_prefix,
            "keys": keys,
            "row_count": int(len(frame)),
            "dataset_id": record["dataset_id"],
            "elapsed_ms": elapsed_ms,
        }
        self.workspace.append_source(thread_id, run_id, source_entry)
        self._log_event(thread_id=thread_id, run_id=run_id, event="fetch_s3", payload=source_entry)

        columns = list(frame.columns)
        preview_rows = frame.head(20).to_dict(orient="records") if not frame.empty else []
        return {
            "dataset_id": record["dataset_id"],
            "row_count": int(len(frame)),
            "columns": columns,
            "column_types": {col: str(frame[col].dtype) for col in columns},
            "preview": preview_rows,
            "preview_text": _format_preview_table(frame),
            "s3_keys": keys,
            "elapsed_ms": elapsed_ms,
        }

    # ── Tool implementation: run_python ──

    def run_python(
        self,
        *,
        thread_id: str,
        run_id: str,
        code: str,
    ) -> dict[str, Any]:
        """Execute Python/pandas code against workspace datasets."""
        self._log_event(
            thread_id=thread_id,
            run_id=run_id,
            event="run_python_start",
            payload={},
        )
        result = self.operator.run_python(thread_id=thread_id, run_id=run_id, code=code)
        self._log_event(
            thread_id=thread_id,
            run_id=run_id,
            event="run_python_done",
            payload={
                "created_datasets": len(result.get("created_datasets", [])),
                "created_analyses": len(result.get("created_analyses", [])),
            },
        )
        return result

    # ── Tool implementation: inspect_table ──

    def inspect_table(
        self,
        table_name: str,
        datasource: str | None = None,
    ) -> dict[str, Any]:
        """Get schema, partitions, and masked sample row for a table."""
        source = datasource or datasource_for_table(table_name)
        metadata = self.registry.inspect_table_metadata(table_name, source)

        sample_row_masked: dict[str, Any] | None = None
        try:
            sample = self.registry.execute_sql(source, f"SELECT * FROM {table_name} LIMIT 1")
            if not sample.empty:
                sample_row_masked = _mask_row(sample.iloc[0].to_dict())
        except Exception:
            sample_row_masked = None

        knowledge = self.kb.retrieve(question=table_name, entities={})
        tier = "common" if table_name in knowledge.get("candidate_tables", []) else "discovered"
        self.kb.upsert_table_metadata(
            table_name=table_name,
            datasource=source,
            columns=metadata.get("columns", []),
            partitions=metadata.get("partitions", []),
            sample_row_masked=sample_row_masked,
            tier=tier,
            notes="Discovered from metadata inspection" if tier == "discovered" else "Common table",
        )

        return {
            **metadata,
            "table_name": table_name,
            "sample_row_masked": sample_row_masked,
            "tier": tier,
        }

    # ── Tool implementation: search_kb ──

    def search_kb(self, query: str) -> dict[str, Any]:
        """Search local knowledge base for matching tables and docs."""
        self.ensure_kb_ready()
        return self.kb.retrieve(question=query, entities={})

    # ── Tool implementation: resolve_codes ──

    def resolve_codes(self, text: str) -> dict[str, Any]:
        """Resolve provider/site/customer from text."""
        return self.resolver.resolve(text)

    # ── Workspace management ──

    def cleanup(self, thread_id: str, mode: str = "ephemeral_manifest") -> dict[str, Any]:
        return self.workspace.cleanup_thread(thread_id=thread_id, mode=mode)

    def start_run(self, thread_id: str) -> str:
        return self.workspace.start_run(thread_id)


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
    return get_runtime().cleanup(thread_id=thread_id, mode=mode)


__all__ = [
    "DatasourceRegistry",
    "EntityResolver",
    "InvestigationRuntime",
    "KnowledgeBase",
    "LocalCodeCatalog",
    "OperatorRuntime",
    "PartitionGuard",
    "SqlGuard",
    "WorkspaceManager",
    "cleanup_thread_workspace",
    "get_runtime",
]
