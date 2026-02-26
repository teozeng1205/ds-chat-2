"""Workspace artifact management for autonomous investigation runs."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


class WorkspaceManager:
    """Manage per-thread/per-run datasets, analyses, logs, and manifests."""

    def __init__(self, *, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(value: str) -> str:
        cleaned = "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-"})
        return cleaned[:80] or "default"

    def _run_path(self, thread_id: str, run_id: str) -> Path:
        return self.root / self._safe(thread_id) / self._safe(run_id)

    def _manifest_path(self, thread_id: str, run_id: str) -> Path:
        return self._run_path(thread_id, run_id) / "manifest.json"

    def _activity_log_path(self, thread_id: str, run_id: str) -> Path:
        return self._run_path(thread_id, run_id) / "logs" / "activity.jsonl"

    def start_run(self, thread_id: str, run_id: str | None = None) -> str:
        final_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        run_path = self._run_path(thread_id, final_run_id)
        (run_path / "datasets").mkdir(parents=True, exist_ok=True)
        (run_path / "analysis").mkdir(parents=True, exist_ok=True)
        (run_path / "logs").mkdir(parents=True, exist_ok=True)
        self._write_manifest(
            thread_id,
            final_run_id,
            {
                "thread_id": thread_id,
                "run_id": final_run_id,
                "created_at": time.time(),
                "datasets": [],
                "analyses": [],
                "events": [],
                "queries": [],
                "sources": [],
                "warnings": [],
            },
        )
        return final_run_id

    def _read_manifest(self, thread_id: str, run_id: str) -> dict[str, Any]:
        path = self._manifest_path(thread_id, run_id)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, thread_id: str, run_id: str, payload: dict[str, Any]) -> None:
        path = self._manifest_path(thread_id, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")

    def append_event(self, thread_id: str, run_id: str, event: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": time.time(),
            "event": event,
            "payload": payload,
        }
        log_path = self._activity_log_path(thread_id, run_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")

        manifest = self._read_manifest(thread_id, run_id)
        manifest.setdefault("events", []).append(record)
        self._write_manifest(thread_id, run_id, manifest)

    def add_warning(self, thread_id: str, run_id: str, warning_text: str) -> None:
        manifest = self._read_manifest(thread_id, run_id)
        warnings = manifest.setdefault("warnings", [])
        if warning_text not in warnings:
            warnings.append(warning_text)
            self._write_manifest(thread_id, run_id, manifest)

    def append_query(self, thread_id: str, run_id: str, query_entry: dict[str, Any]) -> None:
        manifest = self._read_manifest(thread_id, run_id)
        manifest.setdefault("queries", []).append(query_entry)
        self._write_manifest(thread_id, run_id, manifest)

    def append_source(self, thread_id: str, run_id: str, source_entry: dict[str, Any]) -> None:
        manifest = self._read_manifest(thread_id, run_id)
        manifest.setdefault("sources", []).append(source_entry)
        self._write_manifest(thread_id, run_id, manifest)

    def save_dataset(
        self,
        *,
        thread_id: str,
        run_id: str,
        df: pd.DataFrame,
        source_metadata: dict[str, Any],
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        dataset_id = self._safe(dataset_name or f"dataset_{uuid.uuid4().hex[:10]}")
        dataset_dir = self._run_path(thread_id, run_id) / "datasets" / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = dataset_dir / "data.parquet"
        csv_path = dataset_dir / "data.csv"
        if df.empty:
            df.to_csv(csv_path, index=False)
            local_path = csv_path
            file_format = "csv"
        else:
            try:
                df.to_parquet(parquet_path, index=False)
                local_path = parquet_path
                file_format = "parquet"
            except Exception:
                df.to_csv(csv_path, index=False)
                local_path = csv_path
                file_format = "csv"

        record = {
            "dataset_id": dataset_id,
            "row_count": int(len(df)),
            "columns": [str(col) for col in df.columns],
            "local_path": str(local_path),
            "format": file_format,
            "source_metadata": source_metadata,
            "created_at": time.time(),
        }

        manifest = self._read_manifest(thread_id, run_id)
        manifest.setdefault("datasets", []).append(record)
        self._write_manifest(thread_id, run_id, manifest)
        return record

    def list_dataset_records(self, *, thread_id: str, run_id: str) -> list[dict[str, Any]]:
        manifest = self._read_manifest(thread_id, run_id)
        return list(manifest.get("datasets", []))

    def read_datasets(
        self,
        *,
        thread_id: str,
        run_id: str,
        dataset_ids: list[str],
    ) -> dict[str, pd.DataFrame]:
        records = {item["dataset_id"]: item for item in self.list_dataset_records(thread_id=thread_id, run_id=run_id)}
        frames: dict[str, pd.DataFrame] = {}
        for dataset_id in dataset_ids:
            row = records.get(dataset_id)
            if row is None:
                continue
            local_path = Path(str(row.get("local_path", "")))
            if not local_path.exists():
                continue
            if local_path.suffix.lower() == ".parquet":
                frames[dataset_id] = pd.read_parquet(local_path)
            else:
                frames[dataset_id] = pd.read_csv(local_path)
        return frames

    def record_analysis(self, *, thread_id: str, run_id: str, analysis_payload: dict[str, Any]) -> dict[str, Any]:
        analysis_id = self._safe(f"analysis_{uuid.uuid4().hex[:10]}")
        out_path = self._run_path(thread_id, run_id) / "analysis" / f"{analysis_id}.json"
        out_path.write_text(json.dumps(analysis_payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
        record = {
            "analysis_id": analysis_id,
            "local_path": str(out_path),
            "created_at": time.time(),
            "analysis_mode": analysis_payload.get("analysis_mode", "profile_dataset"),
            "summary_stats": analysis_payload.get("summary_stats", {}),
        }
        manifest = self._read_manifest(thread_id, run_id)
        manifest.setdefault("analyses", []).append(record)
        self._write_manifest(thread_id, run_id, manifest)
        return record

    def load_manifest(self, *, thread_id: str, run_id: str) -> dict[str, Any]:
        return self._read_manifest(thread_id, run_id)

    def cleanup_thread(self, thread_id: str, mode: str = "ephemeral_manifest") -> dict[str, Any]:
        thread_root = self.root / self._safe(thread_id)
        if not thread_root.exists():
            return {
                "thread_id": thread_id,
                "deleted_files": 0,
                "deleted_bytes": 0,
                "manifest_retained": 0,
                "mode": mode,
            }

        deleted_files = 0
        deleted_bytes = 0
        manifest_retained = 0

        for run_path in sorted([path for path in thread_root.iterdir() if path.is_dir()]):
            manifest_path = run_path / "manifest.json"
            manifest_copy: str | None = None
            if mode == "ephemeral_manifest" and manifest_path.exists():
                manifest_copy = manifest_path.read_text(encoding="utf-8")

            for path in run_path.rglob("*"):
                if not path.is_file():
                    continue
                if mode == "ephemeral_manifest" and path.name == "manifest.json":
                    continue
                deleted_files += 1
                try:
                    deleted_bytes += path.stat().st_size
                except OSError:
                    pass
                path.unlink(missing_ok=True)

            if mode == "ephemeral_manifest" and manifest_copy is not None:
                manifest_path.write_text(manifest_copy, encoding="utf-8")
                manifest_retained += 1

            for subdir in sorted([path for path in run_path.rglob("*") if path.is_dir()], reverse=True):
                if any(subdir.iterdir()):
                    continue
                subdir.rmdir()
            if mode != "ephemeral_manifest" and run_path.exists() and not any(run_path.iterdir()):
                run_path.rmdir()

        if mode != "ephemeral_manifest" and thread_root.exists() and not any(thread_root.iterdir()):
            thread_root.rmdir()

        return {
            "thread_id": thread_id,
            "deleted_files": deleted_files,
            "deleted_bytes": deleted_bytes,
            "manifest_retained": manifest_retained,
            "mode": mode,
            "root": str(thread_root),
        }


__all__ = ["WorkspaceManager"]
