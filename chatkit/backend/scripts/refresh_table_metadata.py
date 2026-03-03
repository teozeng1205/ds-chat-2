#!/usr/bin/env python3
"""Refresh common_table_live_metadata.json from live datasource schemas.

Connects to all 3 datasources (redshift_analytics, redshift_core, mysql_priceeye),
reads real column definitions and partition info, writes verified metadata.

Usage:
    cd chatkit/backend
    eval "$(assume 3VDEV)"
    .venv/bin/python scripts/refresh_table_metadata.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Bootstrap path so `app.*` imports resolve ──
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.investigation.datasources import DatasourceRegistry, datasource_for_table
from app.investigation.runtime import KNOWLEDGE_ROOT, TABLES_DOC_PATH


# ── Federated schema constants ──

FEDERATED_SCHEMAS_ANALYTICS: list[str] = [
    "federated_priceeye",
    "federated_analytics",
    "federated_metadata",
    "federated_replication",
    "federated_sales_poc",
]

FEDERATED_SCHEMAS_CORE: list[str] = [
    "federated_priceeye",
    "federated_metadata",
    "federated_scheduling",
    "federated_sales_poc",
]

# ── Tables discovered from ds-* repos and priceeye-analytics DDL ──

DISCOVERED_TABLES: list[str] = [
    # priceeye-analytics Glue external tables (redshift_analytics)
    "analytics.market_level_anomalies_v4",
    "analytics.market_level_anomalies_v3",
    "analytics.segment_level_anomalies_v2",
    "analytics.daily_itins_prices_v2",
    "analytics.pax_midt",
    "metadata.table_row_counts",
    "analytics.market_level_analysis_v2",
    "analytics.segment_level_analysis_v2",
    # ds-customer-monitoring (redshift_analytics)
    "analytics.customer_collection_anomalies_v2",
    # ds-internal-monitoring (redshift_core)
    "prod.monitoring.provider_combined_audit",
    "prod.monitoring.combined_audit",
    # ds-channel-comparison (redshift_analytics)
    "prod.common_output.common_output_format",
    # collection_optimizer (redshift_core)
    "collection_optimizer.delta_swia_input_v1",
]


def _canonical_table(table_name: str) -> str:
    return table_name.replace("local.monitoring.", "prod.monitoring.")


def _parse_common_tables(path: Path) -> list[str]:
    """Parse table names from tables.md knowledge file."""
    tables: list[str] = []
    seen: set[str] = set()
    if not path.exists():
        return tables
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


def _discover_federated_tables(registry: DatasourceRegistry) -> list[str]:
    """Discover all tables in federated schemas by querying both clusters.

    Falls back to svv_external_tables if information_schema returns nothing.
    Returns a deduplicated list of "schema.table_name" strings.
    """
    discovered: list[str] = []
    seen: set[str] = set()

    def _fetch_tables(datasource: str, schemas: list[str]) -> list[str]:
        schema_list = ", ".join(f"'{s}'" for s in schemas)
        # Try information_schema first
        try:
            query = (
                "SELECT table_schema, table_name FROM information_schema.tables "
                f"WHERE table_schema IN ({schema_list}) "
                "ORDER BY table_schema, table_name"
            )
            df = registry.execute_sql(datasource, query)
            if not df.empty and len(df) > 0:
                return [f"{row['table_schema']}.{row['table_name']}" for _, row in df.iterrows()]
        except Exception:
            pass

        # Fallback: svv_external_tables (Redshift-specific view for federated/external tables)
        try:
            schema_pattern_conditions = " OR ".join(f"schemaname = '{s}'" for s in schemas)
            query = (
                "SELECT schemaname, tablename FROM svv_external_tables "
                f"WHERE ({schema_pattern_conditions}) "
                "ORDER BY schemaname, tablename"
            )
            df = registry.execute_sql(datasource, query)
            if not df.empty:
                return [f"{row['schemaname']}.{row['tablename']}" for _, row in df.iterrows()]
        except Exception:
            pass

        return []

    for table_ref in _fetch_tables("redshift_analytics", FEDERATED_SCHEMAS_ANALYTICS):
        if table_ref not in seen:
            seen.add(table_ref)
            discovered.append(table_ref)

    for table_ref in _fetch_tables("redshift_core", FEDERATED_SCHEMAS_CORE):
        if table_ref not in seen:
            seen.add(table_ref)
            discovered.append(table_ref)

    return discovered


def _mask_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) <= 3:
        return "***"
    return f"{text[:2]}***{text[-2:]}"


def _refresh_table(registry: DatasourceRegistry, table_name: str) -> dict[str, Any]:
    """Inspect a single table and return its metadata row."""
    datasource = datasource_for_table(table_name)
    row: dict[str, Any] = {
        "table_name": table_name,
        "datasource": datasource,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        metadata = registry.inspect_table_metadata(table_name, datasource)
        row["status"] = "ok"
        row["columns"] = metadata.get("columns", [])
        row["partitions"] = metadata.get("partitions", [])

        # Masked sample row
        try:
            sample = registry.execute_sql(datasource, f"SELECT * FROM {table_name} LIMIT 1")
            if not sample.empty:
                row["sample_row_masked"] = {
                    k: _mask_value(v) for k, v in sample.iloc[0].to_dict().items()
                }
        except Exception:
            row["sample_row_masked"] = None

        # Preview stats
        try:
            preview = registry.execute_sql(datasource, f"SELECT * FROM {table_name} LIMIT 5")
            row["preview_row_count"] = int(len(preview))
            row["preview_columns"] = [str(col) for col in preview.columns]
        except Exception as exc:
            row["preview_error"] = f"{type(exc).__name__}: {exc}"

    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["columns"] = []
        row["partitions"] = []

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh common table live metadata from real schemas.")
    parser.add_argument("--profile", default="3VDEV", help="Credential profile for assume/granted bootstrap.")
    parser.add_argument(
        "--out",
        default=str(KNOWLEDGE_ROOT / "common_table_live_metadata.json"),
        help="Output metadata file path.",
    )
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip AWS credential bootstrap (if already done).")
    args = parser.parse_args()

    if not args.skip_bootstrap:
        cred = _bootstrap_aws_credentials(args.profile)
    else:
        cred = {"profile": args.profile, "skipped": True}

    registry = DatasourceRegistry()

    # Auto-discover federated tables from live clusters
    print("Discovering federated schema tables...")
    federated_tables = _discover_federated_tables(registry)
    print(f"  Found {len(federated_tables)} federated tables across analytics + core clusters.")

    # Merge tables from tables.md + discovered tables + federated tables, deduplicated
    from_doc = _parse_common_tables(TABLES_DOC_PATH)
    all_tables_set: set[str] = set()
    all_tables: list[str] = []
    for t in from_doc + DISCOVERED_TABLES + federated_tables:
        canonical = _canonical_table(t)
        if canonical not in all_tables_set:
            all_tables_set.add(canonical)
            all_tables.append(canonical)

    print(f"Refreshing metadata for {len(all_tables)} tables...")

    rows: list[dict[str, Any]] = []
    for idx, table_name in enumerate(all_tables, 1):
        print(f"  [{idx}/{len(all_tables)}] {table_name} ...", end=" ", flush=True)
        row = _refresh_table(registry, table_name)
        status = row.get("status", "unknown")
        col_count = len(row.get("columns", []))
        print(f"{status} ({col_count} cols)")
        rows.append(row)

    # Write output
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "table_count": len(rows),
        "tables": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")

    ok_count = len([r for r in rows if r.get("status") == "ok"])
    err_count = len(rows) - ok_count
    print(
        json.dumps(
            {
                "credential_bootstrap": cred,
                "metadata_file": str(out_path),
                "table_count": len(rows),
                "ok_count": ok_count,
                "error_count": err_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
