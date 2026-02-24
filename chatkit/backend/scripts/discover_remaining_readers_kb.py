#!/usr/bin/env python3
"""Discover metadata for KB-missing tables from remaining reader profiles.

Outputs:
- knowledgebase/tables/priceeye_remaining_readers_discovered_tables.yaml
- knowledgebase/docs/priceeye_data_model/remaining_readers_discovery.md
- .runtime/remaining_readers_kb_add_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from threevictors.dao import mysql_connector, redshift_connector


ROOT = Path(__file__).resolve().parents[1]
KB_TABLES_DIR = ROOT / "knowledgebase" / "tables"
KB_DOCS_DIR = ROOT / "knowledgebase" / "docs" / "priceeye_data_model"
RUNTIME_DIR = ROOT / ".runtime"
INVENTORY_PATH = RUNTIME_DIR / "remaining_readers_inventory.json"
OUTPUT_TABLE_PATH = KB_TABLES_DIR / "priceeye_remaining_readers_discovered_tables.yaml"
OUTPUT_SUMMARY_PATH = RUNTIME_DIR / "remaining_readers_kb_add_summary.json"
OUTPUT_DOC_PATH = KB_DOCS_DIR / "remaining_readers_discovery.md"


@dataclass(frozen=True)
class TargetTable:
    profile: str
    kind: str
    schema: str
    table: str
    physical: str


def bootstrap_creds(profile: str) -> None:
    proc = subprocess.run(
        ["granted", "credential-process", "--profile", profile, "--auto-login"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    os.environ["AWS_ACCESS_KEY_ID"] = payload["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = payload["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = payload["SessionToken"]
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def sanitize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_existing_physical_names() -> set[str]:
    names: set[str] = set()
    for path in sorted(KB_TABLES_DIR.glob("*.yaml")):
        if path.resolve() == OUTPUT_TABLE_PATH.resolve():
            # Rebuild this file from the other KB table sources.
            continue
        try:
            payload = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        for spec in payload.get("tables") or []:
            physical = spec.get("physical_name")
            if physical:
                names.add(str(physical).lower())
    return names


def load_targets() -> tuple[list[TargetTable], dict[str, Any]]:
    payload = json.loads(INVENTORY_PATH.read_text())
    existing = load_existing_physical_names()

    seen: dict[str, TargetTable] = {}
    for row in payload.get("tables") or []:
        physical = str(row.get("physical") or "").strip().lower()
        if not physical or physical in existing:
            continue
        if physical in seen:
            continue
        target = TargetTable(
            profile=str(row.get("profile") or ""),
            kind=str(row.get("kind") or ""),
            schema=str(row.get("schema") or ""),
            table=str(row.get("table") or ""),
            physical=physical,
        )
        if target.profile and target.kind and target.schema and target.table:
            seen[physical] = target

    inventory_by_profile: dict[str, Any] = {
        row.get("profile"): row for row in payload.get("inventory") or [] if row.get("profile")
    }

    return sorted(seen.values(), key=lambda t: t.physical), inventory_by_profile


class MysqlProfileConnector(mysql_connector.MySQLConnector):
    def __init__(self, profile: str):
        self._profile = profile
        super().__init__()

    def get_properties_filename(self):
        return self._profile


class RedshiftProfileConnector(redshift_connector.RedshiftConnector):
    def __init__(self, profile: str):
        self._profile = profile
        super().__init__()

    def get_properties_filename(self):
        return self._profile


def _chunks(values: list[str], size: int = 200) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def fetch_mysql_columns(conn: Any, schema_to_tables: dict[str, set[str]]) -> dict[tuple[str, str], list[str]]:
    results: dict[tuple[str, str], list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        schemas = sorted(schema_to_tables.keys())
        for chunk in _chunks(schemas, size=20):
            placeholders = ",".join(["%s"] * len(chunk))
            query = (
                "SELECT table_schema, table_name, column_name, ordinal_position "
                "FROM information_schema.columns "
                f"WHERE table_schema IN ({placeholders}) "
                "ORDER BY table_schema, table_name, ordinal_position"
            )
            cur.execute(query, tuple(chunk))
            for schema, table, column, _ordinal in cur.fetchall():
                key = (str(schema), str(table))
                if key[0] in schema_to_tables and key[1] in schema_to_tables[key[0]]:
                    results[key].append(str(column))
    return dict(results)


def fetch_redshift_ext_columns_and_partitions(
    conn: Any,
    schema_to_tables: dict[str, set[str]],
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], list[str]]]:
    columns: dict[tuple[str, str], list[str]] = defaultdict(list)
    parts: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)

    with conn.cursor() as cur:
        schemas = sorted(schema_to_tables.keys())
        for chunk in _chunks(schemas, size=20):
            values = ",".join(["%s"] * len(chunk))
            query = (
                "SELECT schemaname, tablename, columnname, part_key "
                "FROM svv_external_columns "
                f"WHERE schemaname IN ({values}) "
                "ORDER BY schemaname, tablename, columnnum"
            )
            cur.execute(query, tuple(chunk))
            for schema, table, column, part_key in cur.fetchall():
                skey = str(schema)
                tkey = str(table)
                if skey not in schema_to_tables or tkey not in schema_to_tables[skey]:
                    continue
                key = (skey, tkey)
                col = str(column)
                columns[key].append(col)
                try:
                    part_num = int(part_key) if part_key is not None else 0
                except Exception:
                    part_num = 0
                if part_num > 0:
                    parts[key].append((part_num, col))

    partition_cols: dict[tuple[str, str], list[str]] = {}
    for key, items in parts.items():
        dedup: dict[int, str] = {}
        for idx, name in items:
            dedup[idx] = name
        partition_cols[key] = [dedup[idx] for idx in sorted(dedup.keys())]
    return dict(columns), partition_cols


def fetch_redshift_pg_columns(conn: Any, schema_to_tables: dict[str, set[str]]) -> dict[tuple[str, str], list[str]]:
    columns: dict[tuple[str, str], list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        schemas = sorted(schema_to_tables.keys())
        for chunk in _chunks(schemas, size=20):
            values = ",".join(["%s"] * len(chunk))
            query = (
                "SELECT table_schema, table_name, column_name, ordinal_position "
                "FROM information_schema.columns "
                f"WHERE table_schema IN ({values}) "
                "ORDER BY table_schema, table_name, ordinal_position"
            )
            cur.execute(query, tuple(chunk))
            for schema, table, column, _ordinal in cur.fetchall():
                skey = str(schema)
                tkey = str(table)
                if skey not in schema_to_tables or tkey not in schema_to_tables[skey]:
                    continue
                columns[(skey, tkey)].append(str(column))
    return dict(columns)


def detect_date_column(columns: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in [
        "sales_date",
        "observation_date",
        "customer_observation_date",
        "created_at",
        "updated_at",
        "create_date",
        "event_date",
        "run_date",
        "date",
        "day",
    ]:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def detect_customer_column(columns: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in ["customer", "customers", "input_customer", "customer_code"]:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def detect_join_keys(columns: list[str]) -> dict[str, str]:
    lower_map = {c.lower(): c for c in columns}
    out: dict[str, str] = {}

    provider_cols = ["provider_code", "provider", "provider_id"]
    site_cols = ["site_code", "site", "customer_site_code"]
    customer_cols = ["customer", "customers", "input_customer", "customer_code"]

    for c in provider_cols:
        if c in lower_map:
            out["provider"] = lower_map[c]
            break
    for c in site_cols:
        if c in lower_map:
            out["site"] = lower_map[c]
            break
    for c in customer_cols:
        if c in lower_map:
            out["customer"] = lower_map[c]
            break
    if "id" in lower_map:
        out["id"] = lower_map["id"]

    return out


def detect_entity_columns(columns: list[str]) -> dict[str, str]:
    keys = detect_join_keys(columns)
    out: dict[str, str] = {}
    for k in ["provider", "site", "customer"]:
        if k in keys:
            out[k] = keys[k]
    return out


def build_table_spec(target: TargetTable, columns: list[str], partition_cols: list[str]) -> dict[str, Any]:
    col_preview = ", ".join(columns[:8]) if columns else "none"
    part_desc = ", ".join(partition_cols) if partition_cols else "none"
    profile_slug = sanitize_token(target.profile.replace(".properties", ""))
    table_id = f"remaining_{profile_slug}_{sanitize_token(target.schema)}_{sanitize_token(target.table)}"

    source_system = "redshift" if target.kind.startswith("redshift") else "mysql"
    default_customer = detect_customer_column(columns)
    default_date = detect_date_column(columns)

    description = (
        f"Remaining-reader discovered table {target.physical} via {target.profile} "
        f"({target.kind}); partition columns: {part_desc}. "
        f"Columns({len(columns)}): {col_preview}."
    )

    primary_keys: list[str] = []
    if "id" in {c.lower() for c in columns}:
        primary_keys.append(next(c for c in columns if c.lower() == "id"))

    return {
        "table_id": table_id,
        "physical_name": target.physical,
        "source_system": source_system,
        "environment": "3VDEV",
        "description": description,
        "primary_keys": primary_keys,
        "join_keys": detect_join_keys(columns),
        "entity_columns": detect_entity_columns(columns),
        "partition_policy": {
            "partition_columns": partition_cols,
            "required_predicates": list(partition_cols),
            "notes": "Partition columns discovered from metadata catalogs." if partition_cols else "No partition metadata (or non-partitioned table).",
        },
        "default_date_column": default_date,
        "default_customer_column": default_customer,
        "default_limit": 50000,
        "tags": [
            source_system,
            "remaining-readers",
            "discovered",
            profile_slug,
            sanitize_token(target.kind),
            "partition-checked",
        ],
    }


def render_doc(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Remaining Readers Discovery")
    lines.append("")
    lines.append("Scope: KB enrichment for reader profiles that were not yet represented in table specs.")
    lines.append(f"Generated at (UTC): `{summary['generated_at']}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Missing unique tables found from inventory: {summary['missing_unique_tables']}")
    lines.append(f"- Table specs added to KB file: {summary['added_specs']}")
    lines.append(f"- Profiles scanned: {summary['profiles_scanned']}")
    lines.append(f"- Profile failures: {summary['profile_failures']}")
    lines.append("")
    lines.append("## Kind Breakdown")
    for kind, count in summary.get("by_kind", {}).items():
        lines.append(f"- `{kind}`: {count}")
    lines.append("")
    lines.append("## Profile Breakdown")
    for profile, details in summary.get("by_profile", {}).items():
        lines.append(
            f"- `{profile}`: {details.get('tables', 0)} tables, "
            f"{details.get('partitioned_tables', 0)} partitioned, status `{details.get('status', 'ok')}`"
        )
    if summary.get("errors"):
        lines.append("")
        lines.append("## Errors")
        for err in summary["errors"]:
            lines.append(f"- `{err['profile']}`: {err['error']}")
    lines.append("")
    lines.append("## Output")
    lines.append("- `tables/priceeye_remaining_readers_discovered_tables.yaml`")
    lines.append("- `.runtime/remaining_readers_kb_add_summary.json`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws-profile", default="3VDEV")
    args = parser.parse_args()

    bootstrap_creds(args.aws_profile)

    targets, inventory_lookup = load_targets()
    if not targets:
        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "missing_unique_tables": 0,
            "added_specs": 0,
            "profiles_scanned": 0,
            "profile_failures": 0,
            "by_kind": {},
            "by_profile": {},
            "errors": [],
        }
        OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        OUTPUT_TABLE_PATH.write_text("tables: []\n")
        OUTPUT_DOC_PATH.write_text(render_doc(summary))
        return 0

    grouped: dict[str, list[TargetTable]] = defaultdict(list)
    for t in targets:
        grouped[t.profile].append(t)

    specs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    by_profile: dict[str, dict[str, Any]] = {}

    for profile, profile_targets in sorted(grouped.items()):
        inv = inventory_lookup.get(profile, {})
        kind = str(inv.get("kind") or profile_targets[0].kind)

        ext_schema_to_tables: dict[str, set[str]] = defaultdict(set)
        pg_schema_to_tables: dict[str, set[str]] = defaultdict(set)
        mysql_schema_to_tables: dict[str, set[str]] = defaultdict(set)

        for t in profile_targets:
            if t.kind == "mysql":
                mysql_schema_to_tables[t.schema].add(t.table)
            elif t.kind == "redshift_ext":
                ext_schema_to_tables[t.schema].add(t.table)
            elif t.kind == "redshift_pg":
                pg_schema_to_tables[t.schema].add(t.table)

        columns_map: dict[tuple[str, str], list[str]] = {}
        partitions_map: dict[tuple[str, str], list[str]] = {}

        try:
            if kind == "mysql":
                reader = MysqlProfileConnector(profile)
                try:
                    columns_map = fetch_mysql_columns(reader.get_connection(), mysql_schema_to_tables)
                finally:
                    reader.close()
            else:
                reader = RedshiftProfileConnector(profile)
                try:
                    if ext_schema_to_tables:
                        ext_cols, ext_parts = fetch_redshift_ext_columns_and_partitions(
                            reader.get_connection(), ext_schema_to_tables
                        )
                        columns_map.update(ext_cols)
                        partitions_map.update(ext_parts)
                    if pg_schema_to_tables:
                        pg_cols = fetch_redshift_pg_columns(reader.get_connection(), pg_schema_to_tables)
                        columns_map.update(pg_cols)
                finally:
                    reader.close()
        except Exception as exc:
            errors.append({"profile": profile, "error": f"{type(exc).__name__}: {exc}"})
            by_profile[profile] = {
                "tables": len(profile_targets),
                "partitioned_tables": 0,
                "status": "error",
            }
            continue

        prof_partitioned = 0
        prof_added = 0
        for t in profile_targets:
            key = (t.schema, t.table)
            columns = columns_map.get(key, [])
            part_cols = partitions_map.get(key, []) if t.kind == "redshift_ext" else []
            if part_cols:
                prof_partitioned += 1
            specs.append(build_table_spec(t, columns, part_cols))
            prof_added += 1

        by_profile[profile] = {
            "tables": prof_added,
            "partitioned_tables": prof_partitioned,
            "status": "ok",
        }

    specs.sort(key=lambda s: (str(s.get("physical_name", "")), str(s.get("table_id", ""))))
    OUTPUT_TABLE_PATH.write_text(yaml.safe_dump({"tables": specs}, sort_keys=False, allow_unicode=False))

    by_kind = Counter(t.kind for t in targets)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "missing_unique_tables": len(targets),
        "added_specs": len(specs),
        "profiles_scanned": len(grouped),
        "profile_failures": len(errors),
        "by_kind": dict(sorted(by_kind.items())),
        "by_profile": by_profile,
        "errors": errors,
        "output_table_file": str(OUTPUT_TABLE_PATH),
        "output_doc_file": str(OUTPUT_DOC_PATH),
    }
    OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    OUTPUT_DOC_PATH.write_text(render_doc(summary))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
