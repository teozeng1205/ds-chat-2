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
import threading
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── Bootstrap path so `app.*` imports resolve ──
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.investigation.datasources import DatasourceRegistry, datasource_for_table
from app.investigation.runtime import KNOWLEDGE_ROOT, TABLES_DOC_PATH


# ── Tables discovered from ds-* repos and priceeye-analytics DDL ──

DISCOVERED_TABLES: list[str] = [
    # ds-internal-monitoring (redshift_core)
    "prod.monitoring.provider_combined_audit",
    "prod.monitoring.combined_audit",
    # ds-channel-comparison (redshift_analytics)
    "prod.common_output.common_output_format",
]


def _tier_for_table(table_name: str) -> str:
    """Assign tier based on table naming convention."""
    parts = table_name.split(".")
    first = parts[0].lower() if parts else ""
    if first == "local":
        return "local-only"
    if first == "prod":
        return "prod"
    if first.startswith("federated_"):
        return "prod-federated"
    if first in ("analytics", "metadata", "collection_optimizer", "tax_reg", "yqyr_cache"):
        return "analytics-env"
    if first == "billing_db":
        return "prod"
    if first in ("priceeye", "taxregression", "sales_poc"):
        return "mysql"
    return "common"


# MySQL databases to crawl (2-part names kept as-is)
_ALLOWED_MYSQL_DBS: set[str] = {"priceeye", "analytics", "sales_poc", "taxregression"}


def _discover_all_schemas(registry: DatasourceRegistry, datasource: str) -> list[str]:
    """Discover all tables from known relevant schemas via system catalog views."""
    tables: list[str] = []
    seen: set[str] = set()

    if datasource == "mysql_priceeye":
        for db in _ALLOWED_MYSQL_DBS:
            try:
                df_tables = registry.execute_sql(datasource, f"SHOW TABLES FROM `{db}`")
                for _, trow in df_tables.iterrows():
                    table_ref = f"{db}.{str(trow.iloc[0])}"
                    if table_ref not in seen:
                        seen.add(table_ref)
                        tables.append(table_ref)
            except Exception:
                pass
        return tables

    # Dynamically discover all tables in the 'prod' and 'local' cross-databases.
    # SVV_ALL_TABLES sees every database the cluster has access to; filtering by
    # database_name IN ('prod', 'local') gives us only the fully-qualified tables
    # we care about without hardcoding any schema names.
    try:
        df = registry.execute_sql(
            datasource,
            "SELECT database_name, schema_name, table_name FROM SVV_ALL_TABLES "
            "WHERE database_name IN ('prod', 'local') "
            "ORDER BY database_name, schema_name, table_name",
        )
        for _, row in df.iterrows():
            db = str(row.get("database_name", row.iloc[0])).lower()
            tbl_schema = str(row.get("schema_name", row.iloc[1]))
            tbl = str(row.get("table_name", row.iloc[2]))
            table_ref = f"{db}.{tbl_schema}.{tbl}"
            if table_ref not in seen:
                seen.add(table_ref)
                tables.append(table_ref)
    except Exception as exc:
        print(f"  WARN: SVV_ALL_TABLES query failed for {datasource}: {exc}")

    return tables


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


def _execute_with_timeout(registry: DatasourceRegistry, datasource: str, sql: str, timeout: int = 20) -> Any:
    """Execute SQL with a wall-clock timeout. Returns DataFrame or None on timeout/error.

    Uses a daemon thread so hung DB connections don't block the main process.
    """
    result: list[Any] = [None]
    exc: list[Exception | None] = [None]

    def _run() -> None:
        try:
            result[0] = registry.execute_sql(datasource, sql)
        except Exception as e:  # noqa: BLE001
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None  # timed out — daemon thread continues but won't block exit
    if exc[0] is not None:
        raise exc[0]
    return result[0]


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
        "tier": _tier_for_table(table_name),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        metadata = registry.inspect_table_metadata(table_name, datasource)
        row["status"] = "ok"
        row["columns"] = metadata.get("columns", [])
        row["partitions"] = metadata.get("partitions", [])

        # Masked sample row (timeout=15s — skips large Spectrum tables that scan S3)
        try:
            sample = _execute_with_timeout(
                registry, datasource, f"SELECT * FROM {table_name} LIMIT 1", timeout=15
            )
            if sample is not None and not sample.empty:
                row["sample_row_masked"] = {
                    k: _mask_value(v) for k, v in sample.iloc[0].to_dict().items()
                }
            else:
                row["sample_row_masked"] = None
        except Exception:
            row["sample_row_masked"] = None

        # Freshness check: ORDER BY DESC LIMIT 1 uses zone maps on Redshift (fast for internal
        # tables). For external/Spectrum tables it may still scan, so timeout=20s.
        partitions = row.get("partitions", [])
        part_cols = [str(p.get("column", "")) for p in partitions if isinstance(p, dict)]
        date_col = next((c for c in part_cols if "date" in c.lower()), None)
        if date_col:
            try:
                max_row = _execute_with_timeout(
                    registry,
                    datasource,
                    f"SELECT {date_col} AS md FROM {table_name} ORDER BY {date_col} DESC LIMIT 1",
                    timeout=20,
                )
                if max_row is not None and not max_row.empty and max_row.iloc[0]["md"] is not None:
                    row["max_sales_date"] = int(max_row.iloc[0]["md"])
            except Exception:
                pass  # Skip freshness if query fails or times out

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
    parser.add_argument("--stale-days", type=int, default=30, help="Days after which a table with no recent data is dropped.")
    args = parser.parse_args()

    if not args.skip_bootstrap:
        cred = _bootstrap_aws_credentials(args.profile)
    else:
        cred = {"profile": args.profile, "skipped": True}

    registry = DatasourceRegistry()

    # Full schema discovery across all datasources
    print("Discovering all schema tables from live datasources...")
    all_discovered: list[str] = []
    for ds in ["redshift_analytics", "redshift_core", "mysql_priceeye"]:
        print(f"  Scanning {ds}...", end=" ", flush=True)
        try:
            found = _discover_all_schemas(registry, ds)
            print(f"{len(found)} tables")
            all_discovered.extend(found)
        except Exception as exc:
            print(f"error ({exc})")

    # Merge tables from tables.md + hardcoded + full discovery, deduplicated
    from_doc = _parse_common_tables(TABLES_DOC_PATH)
    all_tables_set: set[str] = set()
    all_tables: list[str] = []
    for t in from_doc + DISCOVERED_TABLES + all_discovered:
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
        max_date = row.get("max_sales_date", "")
        print(f"{status} ({col_count} cols){' last=' + str(max_date) if max_date else ''}")
        rows.append(row)

    # Filter stale and error tables
    stale_threshold = int(
        (date_type.today() - timedelta(days=args.stale_days)).strftime("%Y%m%d")
    )
    kept_rows: list[dict[str, Any]] = []
    dropped: list[tuple[str, str]] = []
    for r in rows:
        status = r.get("status", "")
        if status == "error":
            dropped.append((r["table_name"], "error"))
            continue
        # Redshift tables must start with prod. or local. (MySQL keeps 2-part names)
        ds = r.get("datasource", "")
        tname = r.get("table_name", "")
        if ds.startswith("redshift") and not (tname.startswith("prod.") or tname.startswith("local.")):
            dropped.append((tname, "non-prod/local prefix"))
            continue
        max_date = r.get("max_sales_date")
        row_count = r.get("row_count_sample", 0)
        if max_date is not None and int(max_date) < stale_threshold and row_count == 0:
            dropped.append((r["table_name"], f"stale (max_date={max_date})"))
            continue
        kept_rows.append(r)

    if dropped:
        print(f"\nDropped {len(dropped)} tables (error or stale):")
        for name, reason in dropped[:20]:
            print(f"  - {name}: {reason}")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped) - 20} more")

    rows = kept_rows

    # Write output
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stale_days": args.stale_days,
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
                "dropped_count": len(dropped),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
