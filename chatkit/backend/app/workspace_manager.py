"""Per-turn workspace and dataset artifact management."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .investigation.types import DatasetHandle, DatasetManifest


DEFAULT_RUNTIME_ROOT = Path(__file__).resolve().parent.parent / ".runtime" / "workspaces"


def _safe_dataset_id(raw: str | None = None) -> str:
    if raw:
        cleaned = "".join(ch for ch in raw.strip() if ch.isalnum() or ch in {"_", "-"})
        if cleaned:
            return cleaned[:64]
    return f"dataset_{uuid.uuid4().hex[:12]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _as_schema(df: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": str(col), "dtype": str(dtype)} for col, dtype in df.dtypes.items()]


class TurnWorkspace:
    """Filesystem contract for a single user turn workspace."""

    def __init__(self, root_path: Path):
        self.root_path = root_path.resolve()
        self.datasets_path = (self.root_path / "datasets").resolve()
        self.datasets_path.mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, dataset_id: str) -> Path:
        return (self.datasets_path / _safe_dataset_id(dataset_id)).resolve()

    def write_dataset(
        self,
        *,
        df: pd.DataFrame,
        dataset_id: str | None,
        source_type: str,
        source_ref: str,
        query: str | None = None,
        s3_keys: list[str] | None = None,
        partitions: dict[str, Any] | None = None,
        lineage: list[str] | None = None,
        source_step_id: str | None = None,
        preview_rows: int = 200,
    ) -> tuple[DatasetHandle, DatasetManifest]:
        final_dataset_id = _safe_dataset_id(dataset_id)
        out_dir = self.dataset_dir(final_dataset_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / "data.parquet"
        preview_path = out_dir / "preview.csv"
        manifest_path = out_dir / "manifest.json"

        frame = df.copy()
        frame.to_parquet(parquet_path, index=False)
        frame.head(max(0, preview_rows)).to_csv(preview_path, index=False)

        manifest = DatasetManifest(
            dataset_id=final_dataset_id,
            source_type=source_type,  # type: ignore[arg-type]
            source_ref=source_ref,
            query=query,
            s3_keys=s3_keys or [],
            partitions=partitions or {},
            row_count=int(len(frame)),
            columns_schema=_as_schema(frame),
            lineage=lineage or [],
            sha256=_sha256_file(parquet_path),
        )
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json", by_alias=True), indent=2),
            encoding="utf-8",
        )

        handle = DatasetHandle(
            dataset_id=final_dataset_id,
            path=str(parquet_path),
            manifest_path=str(manifest_path),
            row_count=int(len(frame)),
            columns=[str(c) for c in frame.columns],
            source_step_id=source_step_id,
        )
        return handle, manifest

    def read_dataset(self, dataset_id: str) -> pd.DataFrame:
        parquet_path = self.dataset_dir(dataset_id) / "data.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Dataset parquet not found for {dataset_id}: {parquet_path}")
        return pd.read_parquet(parquet_path)

    def read_manifest(self, dataset_id: str) -> DatasetManifest:
        manifest_path = self.dataset_dir(dataset_id) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found for {dataset_id}: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return DatasetManifest.model_validate(payload)

    def list_dataset_ids(self) -> list[str]:
        if not self.datasets_path.exists():
            return []
        return sorted([p.name for p in self.datasets_path.iterdir() if p.is_dir()])

    def cleanup(self) -> dict[str, Any]:
        start = time.monotonic()
        bytes_removed = 0
        files_removed = 0

        if self.root_path.exists():
            for path in self.root_path.rglob("*"):
                if path.is_file():
                    try:
                        bytes_removed += path.stat().st_size
                        files_removed += 1
                    except OSError:
                        pass
            shutil.rmtree(self.root_path, ignore_errors=True)

        return {
            "workspace": str(self.root_path),
            "files_removed": files_removed,
            "bytes_removed": bytes_removed,
            "duration_ms": int((time.monotonic() - start) * 1000),
            "deleted": not self.root_path.exists(),
        }


class WorkspaceManager:
    """Factory for per-turn workspaces."""

    def __init__(self, runtime_root: Path | None = None):
        self.runtime_root = (runtime_root or DEFAULT_RUNTIME_ROOT).resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)

    def create_turn_workspace(self, thread_id: str, turn_id: str) -> TurnWorkspace:
        safe_thread = _safe_dataset_id(thread_id)
        safe_turn = _safe_dataset_id(turn_id)
        root_path = (self.runtime_root / safe_thread / safe_turn).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        return TurnWorkspace(root_path)
