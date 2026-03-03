#!/usr/bin/env python3
"""Live-enrich common table metadata and persist it into local KB sources."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.investigation.runtime import TABLES_DOC_PATH, get_runtime


def _canonical_table(table_name: str) -> str:
    return table_name.replace("local.monitoring.", "prod.monitoring.")


def _parse_common_tables(path: Path) -> list[str]:
    tables: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [part.strip() for part in line.strip("|").split("|")]
        if not cells:
            continue
        candidate = cells[0].strip("`")
        if "." not in candidate or " " in candidate:
            continue
        if candidate.lower().startswith("table"):
            continue
        candidate = _canonical_table(candidate)
        if "{" in candidate or "}" in candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        tables.append(candidate)
    return tables


def _bootstrap_aws_credentials(profile: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["zsh", "-lc", f"assume {profile} >/dev/null 2>&1; env -0"],
        capture_output=True,
        text=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"Failed to assume profile {profile}: {stderr.strip() or 'unknown error'}")

    output = proc.stdout.decode("utf-8", errors="replace")
    loaded = 0
    for pair in output.split("\x00"):
        if not pair or "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        if key.startswith("AWS_"):
            os.environ[key] = value
            loaded += 1

    if loaded == 0:
        fallback = subprocess.run(
            ["granted", "credential-process", "--profile", profile, "--auto-login"],
            capture_output=True,
            text=True,
        )
        if fallback.returncode != 0:
            stderr = fallback.stderr or ""
            raise RuntimeError(f"Credential fallback failed for {profile}: {stderr.strip() or 'unknown error'}")
        payload = json.loads(fallback.stdout)
        os.environ["AWS_ACCESS_KEY_ID"] = str(payload.get("AccessKeyId") or "")
        os.environ["AWS_SECRET_ACCESS_KEY"] = str(payload.get("SecretAccessKey") or "")
        os.environ["AWS_SESSION_TOKEN"] = str(payload.get("SessionToken") or "")
        loaded = 3

    os.environ.setdefault("AWS_REGION", "us-east-1")
    return {"profile": profile, "env_keys_loaded": loaded}


def main() -> int:
    parser = argparse.ArgumentParser(description="Live-enrich common table metadata into KB.")
    parser.add_argument("--profile", default="3VDEV", help="Credential profile for assume/granted bootstrap.")
    parser.add_argument(
        "--out",
        default="app/investigation/knowledge/common_table_live_metadata.json",
        help="Output metadata file path.",
    )
    args = parser.parse_args()

    cred = _bootstrap_aws_credentials(args.profile)
    runtime = get_runtime()
    tables = _parse_common_tables(TABLES_DOC_PATH)
    rows: list[dict[str, Any]] = []

    for table_name in tables:
        datasource = runtime._datasource_for_table(table_name)  # noqa: SLF001
        row: dict[str, Any] = {
            "table_name": table_name,
            "datasource": datasource,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            metadata = runtime.inspect_table_metadata(table_name=table_name, datasource=datasource, capture_example_row=True)
            row["status"] = "ok"
            row["columns"] = metadata.get("columns", [])
            row["partitions"] = metadata.get("partitions", [])
            row["sample_row_masked"] = metadata.get("sample_row_masked")
            try:
                preview = runtime.registry.execute_sql(datasource, f"SELECT * FROM {table_name} LIMIT 5")
                row["preview_row_count"] = int(len(preview))
                row["preview_columns"] = [str(col) for col in preview.columns]
            except Exception as exc:  # noqa: BLE001
                row["preview_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["columns"] = []
            row["partitions"] = []
        rows.append(row)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "table_count": len(rows),
        "tables": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    refresh = runtime.refresh_knowledge_index(force=True)

    ok_count = len([row for row in rows if row.get("status") == "ok"])
    err_count = len(rows) - ok_count
    print(
        json.dumps(
            {
                "credential_bootstrap": cred,
                "metadata_file": str(out_path),
                "table_count": len(rows),
                "ok_count": ok_count,
                "error_count": err_count,
                "kb_refresh": refresh,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
