"""Execution primitives: SQL guard, partition guard, and Python operator runtime."""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from .workspace import WorkspaceManager


class PartitionFilterRequired(ValueError):
    """Raised when a query targets a partitioned table without a partition predicate.

    The table-query protocol hard-gates partitioned tables: a query that omits a
    predicate on a known partition key is rejected before execution instead of
    silently full-scanning. The message lists the missing key(s) so the agent can
    immediately retry with a valid predicate (or aggregate with GROUP BY).
    """


class SqlGuard:
    """Minimal guardrails for read-only SQL with row clamp."""

    def __init__(self, *, default_limit: int = 1000, max_limit: int = 120000) -> None:
        self.default_limit = default_limit
        self.max_limit = max_limit

    @staticmethod
    def _strip_comments(query: str) -> str:
        cleaned = re.sub(r"--.*?$", "", query, flags=re.M)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.S)
        return cleaned.strip()

    @staticmethod
    def _is_read_only(query: str) -> bool:
        lowered = query.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            return False
        forbidden = [" insert ", " update ", " delete ", " drop ", " alter ", " truncate ", " create ", " grant ", " revoke "]
        return all(token not in f" {lowered} " for token in forbidden)

    @staticmethod
    def _single_statement(query: str) -> bool:
        return query.count(";") <= 1

    def _apply_limit(self, query: str) -> str:
        stripped = query.rstrip().rstrip(";")
        match = re.search(r"\blimit\s+(\d+)\b", stripped, flags=re.I)
        if match:
            value = int(match.group(1))
            clamped = min(value, self.max_limit)
            stripped = re.sub(r"\blimit\s+\d+\b", f"LIMIT {clamped}", stripped, flags=re.I)
        else:
            stripped = f"{stripped} LIMIT {self.default_limit}"
        return stripped + ";"

    @staticmethod
    def _normalize_known_schema_aliases(query: str) -> str:
        # Redshift does not reliably support PostgreSQL's pg_typeof in these
        # external/federated contexts. Treat type-probe attempts as a harmless
        # constant projection so exploratory calls do not fail the agent run.
        query = re.sub(r"\bpg_typeof\s*\([^)]*\)", "'unknown'", query, flags=re.I)
        # Redshift can reject CONCAT(col, '-', col) for mixed varchar/unknown
        # literals. The || operator is the idiomatic equivalent and avoids the
        # type-resolution failure.
        query = re.sub(
            r"\bCONCAT\s*\(\s*([^,()]+?)\s*,\s*'-'\s*,\s*([^,()]+?)\s*\)",
            r"(\1 || '-' || \2)",
            query,
            flags=re.I,
        )
        lowered = query.lower()
        if "prod.monitoring.provider_combined_audit" not in lowered:
            return query
        # Provider combined audit is an aggregated provider/site view. It has
        # `inputrequestid_count`, not raw `inputrequestid`; normalize the most
        # common mistaken aggregate before it reaches Redshift.
        query = re.sub(
            r"count\s*\(\s*distinct\s+inputrequestid\s*\)",
            "SUM(inputrequestid_count)",
            query,
            flags=re.I,
        )
        return re.sub(r"\binputrequestid\b", "inputrequestid_count", query, flags=re.I)

    def validate(self, query: str) -> str:
        cleaned = self._strip_comments(query)
        if not cleaned:
            raise ValueError("SQL query is empty")
        if not self._single_statement(cleaned):
            raise ValueError("Only single-statement SQL is supported")
        if not self._is_read_only(cleaned):
            raise ValueError("Only SELECT/WITH read-only SQL is supported")
        return self._apply_limit(self._normalize_known_schema_aliases(cleaned))


class PartitionGuard:
    """Validates that queries include required partition predicates in WHERE clause.

    Two modes:
      1. Legacy classmethod `check(query, table_name=None)` uses a static
         hardcoded map of known tables → partition columns. Kept
         unchanged for existing call sites.
      2. Instance mode constructed via `PartitionGuard.from_glue(glue_catalog)`:
         `check_live(query, table_name=None)` looks up partition keys
         live from the Glue Data Catalog. Falls back to the static map
         when Glue can't resolve the table.
    """

    # Known table -> required partition columns mapping
    _REQUIRED_PARTITIONS: dict[str, list[str]] = {
        "analytics.market_level_anomalies_v3": ["sales_date", "customer"],
        "analytics.market_level_anomalies_v4": ["sales_date", "customer"],
        "analytics.market_level_analysis_v2": ["sales_date", "customer"],
        "analytics.segment_level_analysis_v2": ["sales_date", "customer"],
        "prod.monitoring.provider_combined_audit": ["sales_date"],
        "prod.monitoring.combined_audit": ["sales_date"],
        "prod.common_output.common_output_format": ["sales_date"],
    }

    def __init__(self, glue_catalog: Any | None = None) -> None:
        self._glue = glue_catalog

    @classmethod
    def from_glue(cls, glue_catalog: Any) -> "PartitionGuard":
        """Build a guard that consults the live Glue catalog on each check."""
        return cls(glue_catalog=glue_catalog)

    @classmethod
    def check(cls, query: str, table_name: str | None = None) -> list[str]:
        """Legacy classmethod check against the static map. Unchanged."""
        warnings: list[str] = []
        if table_name is None:
            tables = cls._extract_table_names(query)
        else:
            tables = [table_name.strip().lower()]

        lowered_query = query.lower()

        for table in tables:
            required = cls._REQUIRED_PARTITIONS.get(table)
            if required is None:
                for known_table, partitions in cls._REQUIRED_PARTITIONS.items():
                    if table.endswith(known_table) or known_table.endswith(table):
                        required = partitions
                        break

            if required is None:
                continue

            for partition_col in required:
                if partition_col not in lowered_query:
                    warnings.append(
                        f"Query on {table} is missing required partition filter: {partition_col}. "
                        f"Add WHERE {partition_col} = <value> to avoid full table scan."
                    )

        return warnings

    def check_live(self, query: str, table_name: str | None = None) -> list[str]:
        """Instance check. Uses Glue when available, falls back to the static map.

        Missing-partition warnings use the same wording as the legacy
        classmethod so downstream UI is consistent.
        """
        warnings: list[str] = []
        tables = (
            [table_name.strip().lower()]
            if table_name is not None
            else self._extract_table_names(query)
        )
        lowered_query = query.lower()

        for table in tables:
            required = self._required_for(table)
            if required is None:
                continue
            for partition_col in required:
                if partition_col.lower() not in lowered_query:
                    warnings.append(
                        f"Query on {table} is missing required partition filter: {partition_col}. "
                        f"Add WHERE {partition_col} = <value> to avoid full table scan."
                    )
        return warnings

    def _required_for(self, table: str) -> list[str] | None:
        """Resolve partition columns for a table, Glue first then static fallback."""
        if self._glue is not None:
            glue_partitions = self._lookup_glue(table)
            if glue_partitions is not None:
                return glue_partitions

        static = self._REQUIRED_PARTITIONS.get(table)
        if static is not None:
            return static
        for known_table, partitions in self._REQUIRED_PARTITIONS.items():
            if table.endswith(known_table) or known_table.endswith(table):
                return partitions
        return None

    def _lookup_glue(self, table: str) -> list[str] | None:
        """Ask Glue for partition keys. Accepts 'db.name', 'schema.db.name', or bare."""
        ref_parts = [p for p in table.split(".") if p]
        db: str | None
        name: str
        if len(ref_parts) >= 2:
            db, name = ref_parts[-2], ref_parts[-1]
        else:
            db, name = None, ref_parts[-1] if ref_parts else table

        try:
            if db is not None:
                found = self._glue.get_table(db, name)
                if found is None:
                    hits = self._glue.discover_table(name)
                    found = hits[0] if hits else None
            else:
                hits = self._glue.discover_table(name)
                found = hits[0] if hits else None
        except Exception:
            return None

        if found is None:
            return None
        keys = list(getattr(found, "partition_key_names", ()) or ())
        return keys or None

    @staticmethod
    def _extract_table_names(query: str) -> list[str]:
        """Extract table names from FROM/JOIN clauses."""
        pattern = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.I)
        return [m.group(1).lower() for m in pattern.finditer(query)]


class OperatorRuntime:
    """Pandas-first Python executor over run-local dataset artifacts."""

    _DANGEROUS_PATTERNS = (
        "os.system",
        "subprocess.",
        "shutil.rmtree",
        "rm -rf",
        "Path('/').",
        "unlink(",
    )

    def __init__(self, workspace: WorkspaceManager) -> None:
        self.workspace = workspace

    @classmethod
    def _ensure_safe(cls, code: str) -> None:
        lowered = code.lower()
        for token in cls._DANGEROUS_PATTERNS:
            if token.lower() in lowered:
                raise ValueError(f"Blocked dangerous python operation: {token}")

    def run_python(self, *, thread_id: str, run_id: str, code: str) -> dict[str, Any]:
        self._ensure_safe(code)
        before_ids = {item["dataset_id"] for item in self.workspace.list_dataset_records(thread_id=thread_id, run_id=run_id)}
        before_analyses = {
            item.get("analysis_id")
            for item in self.workspace.load_manifest(thread_id=thread_id, run_id=run_id).get("analyses", [])
            if isinstance(item, dict) and item.get("analysis_id")
        }
        stdout = io.StringIO()

        def list_datasets() -> list[dict[str, Any]]:
            return self.workspace.list_dataset_records(thread_id=thread_id, run_id=run_id)

        def load_dataset(dataset_id: str) -> pd.DataFrame:
            frames = self.workspace.read_datasets(thread_id=thread_id, run_id=run_id, dataset_ids=[dataset_id])
            if dataset_id not in frames:
                raise KeyError(f"Unknown dataset_id: {dataset_id}")
            return frames[dataset_id]

        def save_dataframe(df: pd.DataFrame, dataset_name: str, source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
            return self.workspace.save_dataset(
                thread_id=thread_id,
                run_id=run_id,
                df=df,
                source_metadata=source_metadata or {"type": "python", "generated": True},
                dataset_name=dataset_name,
            )

        def save_plot(fig: Any, name: str) -> str:
            """Save a matplotlib figure and return the file path."""
            plot_path = f"/tmp/{name}.png"
            fig.tight_layout()
            fig.savefig(plot_path, dpi=120)
            if plt:
                plt.close(fig)
            return plot_path

        def save_analysis(payload: dict[str, Any]) -> dict[str, Any]:
            return self.workspace.record_analysis(thread_id=thread_id, run_id=run_id, analysis_payload=payload)

        scope: dict[str, Any] = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "json": json,
            "Path": Path,
            "list_datasets": list_datasets,
            "load_dataset": load_dataset,
            "save_dataframe": save_dataframe,
            "save_plot": save_plot,
            "save_analysis": save_analysis,
        }

        with contextlib.redirect_stdout(stdout):
            exec(code, scope, scope)

        records = self.workspace.list_dataset_records(thread_id=thread_id, run_id=run_id)
        created = [item for item in records if item["dataset_id"] not in before_ids]
        manifest = self.workspace.load_manifest(thread_id=thread_id, run_id=run_id)
        analyses = [item for item in manifest.get("analyses", []) if isinstance(item, dict)]
        created_analyses = [item for item in analyses if item.get("analysis_id") not in before_analyses]
        return {
            "ok": True,
            "stdout": stdout.getvalue(),
            "created_datasets": created,
            "created_analyses": created_analyses,
            "run_id": run_id,
        }


__all__ = ["OperatorRuntime", "PartitionFilterRequired", "PartitionGuard", "SqlGuard"]
