"""Execution primitives: SQL guard and Python operator runtime."""

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

    def validate(self, query: str) -> str:
        cleaned = self._strip_comments(query)
        if not cleaned:
            raise ValueError("SQL query is empty")
        if not self._single_statement(cleaned):
            raise ValueError("Only single-statement SQL is supported")
        if not self._is_read_only(cleaned):
            raise ValueError("Only SELECT/WITH read-only SQL is supported")
        return self._apply_limit(cleaned)


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
            "save_analysis": save_analysis,
        }

        with contextlib.redirect_stdout(stdout):
            exec(code, scope, scope)

        records = self.workspace.list_dataset_records(thread_id=thread_id, run_id=run_id)
        created = [item for item in records if item["dataset_id"] not in before_ids]
        return {
            "ok": True,
            "stdout": stdout.getvalue(),
            "created_datasets": created,
            "run_id": run_id,
        }


__all__ = ["OperatorRuntime", "SqlGuard"]
