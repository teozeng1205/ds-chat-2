#!/usr/bin/env python3
"""Regenerate aws_infrastructure.md from live AWS CLI discovery.

Usage:
    # 1. Authenticate (interactive browser required for initial SSO login)
    #    aws sso login --sso-session 3V

    # 2. Export static creds (avoids botocore RefreshableCredentials bug)
    #    eval "$(aws configure export-credentials --profile 3VDEV --format env)"
    #    unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_CREDENTIAL_EXPIRATION

    # 3. Run discovery (3VDEV section)
    #    python scripts/refresh_aws_infra.py

    # 4. Optionally refresh 3VPROD section too
    #    eval "$(aws configure export-credentials --profile 3VPROD --format env)"
    #    unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_CREDENTIAL_EXPIRATION
    #    python scripts/refresh_aws_infra.py --account 3VPROD
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "app" / "investigation" / "knowledge" / "docs"
OUTPUT_FILE = KB_DIR / "aws_infrastructure.md"
REGION = "us-east-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def aws(*args: str, silent_errors: bool = False) -> dict | list | str | None:
    """Run an AWS CLI command and return parsed JSON output."""
    cmd = ["aws", "--region", REGION, "--output", "json", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            if not silent_errors:
                print(f"  [WARN] {' '.join(args[:4])}: {result.stderr.strip()[:120]}")
            return None
        return json.loads(result.stdout) if result.stdout.strip() else None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {' '.join(args[:4])}")
        return None
    except Exception as e:  # noqa: BLE001
        if not silent_errors:
            print(f"  [ERR] {e}")
        return None


def jq(data: dict | list | None, *keys: str, default: str = "?") -> str:
    """Safe nested key access returning a string."""
    if data is None:
        return default
    cur = data
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list) and isinstance(k, int):
            cur = cur[k] if k < len(cur) else None
        else:
            return default
        if cur is None:
            return default
    return str(cur)


def md_table(rows: list[list[str]], headers: list[str]) -> str:
    """Format a markdown table."""
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------

def discover_identity() -> dict:
    print("  Checking identity...")
    data = aws("sts", "get-caller-identity")
    return data or {}


def discover_redshift_serverless() -> str:
    print("  Redshift Serverless workgroups...")
    data = aws("redshift-serverless", "list-workgroups")
    if not data:
        return "_No data (credentials may not have redshift-serverless access)_\n"
    rows = []
    for wg in data.get("workgroups", []):
        name = wg.get("workgroupName", "?")
        status = wg.get("status", "?")
        endpoint = wg.get("endpoint", {}).get("address", "?")
        rows.append([name, status, endpoint])
    return md_table(rows, ["Workgroup", "Status", "Endpoint"]) + "\n"


def discover_rds_clusters() -> str:
    print("  RDS clusters...")
    data = aws("rds", "describe-db-clusters")
    if not data:
        return "_No data_\n"
    rows = []
    for cluster in data.get("DBClusters", []):
        rows.append([
            cluster.get("DBClusterIdentifier", "?"),
            cluster.get("Engine", "?"),
            cluster.get("Status", "?"),
            cluster.get("Endpoint", "?"),
        ])
    return md_table(rows, ["Cluster ID", "Engine", "Status", "Endpoint"]) + "\n"


def discover_step_functions() -> str:
    print("  Step Functions...")
    data = aws("stepfunctions", "list-state-machines", "--max-results", "200")
    if not data:
        return "_No data_\n"
    rows = []
    for sfn in data.get("stateMachines", []):
        name = sfn.get("name", "?")
        arn = sfn.get("stateMachineArn", "?")
        rows.append([name, arn])
    return md_table(rows, ["Name", "ARN"]) + "\n"


def discover_lambdas() -> str:
    print("  Lambda functions...")
    data = aws("lambda", "list-functions", "--max-items", "200")
    if not data:
        return "_No data_\n"
    rows = []
    for fn in data.get("Functions", []):
        rows.append([
            fn.get("FunctionName", "?"),
            fn.get("Runtime", "?"),
            str(fn.get("Timeout", "?")),
            fn.get("LastModified", "?")[:10],
        ])
    rows.sort(key=lambda r: r[0])
    return md_table(rows, ["Function Name", "Runtime", "Timeout(s)", "Last Modified"]) + "\n"


def discover_lambda_configs(names: list[str]) -> str:
    print(f"  Lambda configs for {len(names)} key functions...")
    sections = []
    for name in names:
        data = aws("lambda", "get-function-configuration", "--function-name", name,
                   silent_errors=True)
        if not data:
            sections.append(f"**{name}** — _not found_\n")
            continue
        env = data.get("Environment", {}).get("Variables", {})
        env_lines = "\n".join(f"  - `{k}`: `{v}`" for k, v in sorted(env.items()))
        sections.append(
            f"**{name}**\n"
            f"- Runtime: `{data.get('Runtime', '?')}`\n"
            f"- Timeout: `{data.get('Timeout', '?')}s` | Memory: `{data.get('MemorySize', '?')}MB`\n"
            f"- Handler: `{data.get('Handler', '?')}`\n"
            f"- Environment:\n{env_lines or '  _(none)_'}\n"
        )
    return "\n".join(sections)


def discover_glue_databases() -> str:
    print("  Glue databases...")
    data = aws("glue", "get-databases", "--max-results", "200")
    if not data:
        return "_No data_\n"
    rows = []
    for db in data.get("DatabaseList", []):
        name = db.get("Name", "?")
        loc = db.get("LocationUri", "")
        rows.append([name, loc[:60] if loc else ""])
    return md_table(rows, ["Database", "Location"]) + "\n"


def discover_glue_tables(db_name: str, max_results: int = 50) -> str:
    print(f"  Glue tables in {db_name}...")
    data = aws("glue", "get-tables", "--database-name", db_name,
               "--max-results", str(max_results), silent_errors=True)
    if not data:
        return f"_No tables found in {db_name}_\n"
    tables = [t.get("Name", "?") for t in data.get("TableList", [])]
    return ", ".join(f"`{t}`" for t in sorted(tables)) + "\n"


def discover_glue_partitions(db_name: str, table_name: str, n: int = 3) -> str:
    data = aws("glue", "get-partitions", "--database-name", db_name,
               "--table-name", table_name, "--max-results", str(n * 5),
               silent_errors=True)
    if not data:
        return "_no partitions_"
    parts = data.get("Partitions", [])
    if not parts:
        return "_empty_"
    # show last n
    last = parts[-n:]
    return ", ".join(str(p.get("Values", [])) for p in last)


def discover_eventbridge_rules() -> str:
    print("  EventBridge rules...")
    data = aws("events", "list-rules", "--limit", "100")
    if not data:
        return "_No data_\n"
    rows = []
    for rule in data.get("Rules", []):
        rows.append([
            rule.get("Name", "?"),
            rule.get("State", "?"),
            rule.get("ScheduleExpression") or rule.get("EventPattern", "?")[:50],
        ])
    return md_table(rows, ["Rule Name", "State", "Schedule / Pattern"]) + "\n"


def discover_cw_alarms() -> str:
    print("  CloudWatch alarms...")
    data = aws("cloudwatch", "describe-alarms", "--max-records", "100")
    if not data:
        return "_No data_\n"
    rows = []
    for alarm in data.get("MetricAlarms", []):
        rows.append([
            alarm.get("AlarmName", "?"),
            alarm.get("StateValue", "?"),
            alarm.get("StateReason", "")[:60],
        ])
    return md_table(rows, ["Alarm Name", "State", "Reason (truncated)"]) + "\n"


def discover_s3_buckets() -> str:
    print("  S3 buckets...")
    result = subprocess.run(
        ["aws", "s3", "ls", "--region", REGION],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return "_No data_\n"
    lines = result.stdout.strip().splitlines()
    buckets = [line.split()[-1] for line in lines if line.strip()]
    return "\n".join(f"- `{b}`" for b in sorted(buckets)) + "\n"


def discover_secrets() -> str:
    print("  Secrets Manager...")
    data = aws("secretsmanager", "list-secrets", "--max-results", "100")
    if not data:
        return "_No data_\n"
    names = sorted(s.get("Name", "?") for s in data.get("SecretList", []))
    return "\n".join(f"- `{n}`" for n in names) + "\n"


def discover_ssm_params() -> str:
    print("  SSM parameters...")
    data = aws("ssm", "describe-parameters", "--max-results", "50")
    if not data:
        return "_No data_\n"
    names = sorted(p.get("Name", "?") for p in data.get("Parameters", []))
    return "\n".join(f"- `{n}`" for n in names) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

KEY_LAMBDAS = [
    "anomalies_process_customer_v2",
    "alerts",
    "partitioncreator",
    "dropdead-detector",
    "persist-audit-data-redshift",
    "persist-audit-data-mysql",
]

KEY_GLUE_DBS = [
    "analytics_db",
    "monitoring_db",
    "billing_db",
    "collection_optimizer_db",
    "site-metrics-db",
    "tax_reg_db",
    "priceeye_audits_db",
    "common_output_db",
    "data_lakes_db",
    "yqyr_cache_db",
]

FRESHNESS_TABLES = [
    ("analytics_db", "market_level_anomalies_v4"),
    ("monitoring_db", "provider_combined_audit"),
    ("monitoring_db", "customer_combined_audit_v2"),
    ("common_output_db", "common_output_format"),
]


def build_doc(account: str, account_id: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []

    sections.append(f"# PriceEye AWS Infrastructure Reference")
    sections.append(
        f"*Last refreshed: {now}*\n"
        f"*Account: {account} ({account_id})*\n"
        f"*Regenerate: `python scripts/refresh_aws_infra.py`*"
    )

    print(f"\n[{account}] Redshift Serverless")
    sections.append(f"## Redshift Serverless Workgroups\n\n" + discover_redshift_serverless())

    print(f"[{account}] RDS clusters")
    sections.append(f"## Aurora MySQL Clusters\n\n" + discover_rds_clusters())

    print(f"[{account}] Step Functions")
    sections.append(f"## Step Functions\n\n" + discover_step_functions())

    print(f"[{account}] Lambda")
    sections.append(f"## Lambda Functions\n\n" + discover_lambdas())

    print(f"[{account}] Lambda configs")
    sections.append(f"## Key Lambda Configurations\n\n" + discover_lambda_configs(KEY_LAMBDAS))

    print(f"[{account}] Glue databases")
    sections.append(f"## Glue Databases\n\n" + discover_glue_databases())

    glue_tables_section = f"## Glue Tables (Key Databases)\n"
    for db in KEY_GLUE_DBS:
        glue_tables_section += f"\n### {db}\n\n" + discover_glue_tables(db)
    sections.append(glue_tables_section)

    print(f"[{account}] Glue partition freshness")
    freshness_rows = []
    for db, tbl in FRESHNESS_TABLES:
        latest = discover_glue_partitions(db, tbl)
        freshness_rows.append([f"`{db}.{tbl}`", latest])
    sections.append(
        "## Data Freshness (Latest Glue Partitions)\n\n"
        + md_table(freshness_rows, ["Table", "Latest Partition Values"])
    )

    print(f"[{account}] EventBridge")
    sections.append(f"## EventBridge Rules\n\n" + discover_eventbridge_rules())

    print(f"[{account}] CloudWatch alarms")
    sections.append(f"## CloudWatch Alarms\n\n" + discover_cw_alarms())

    print(f"[{account}] S3 buckets")
    sections.append(f"## S3 Buckets\n\n" + discover_s3_buckets())

    print(f"[{account}] Secrets Manager")
    sections.append(f"## Secrets Manager (Names Only)\n\n" + discover_secrets())

    print(f"[{account}] SSM parameters")
    sections.append(f"## SSM Parameter Store\n\n" + discover_ssm_params())

    return "\n\n---\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh AWS infrastructure KB")
    parser.add_argument("--account", default="3VDEV", help="Account label (3VDEV or 3VPROD)")
    args = parser.parse_args()

    print("Checking AWS identity...")
    identity = discover_identity()
    if not identity:
        print("ERROR: Cannot call AWS. Check credentials (see MEMORY.md for SSO steps).")
        sys.exit(1)

    account_id = identity.get("Account", "unknown")
    print(f"Account: {account_id} ({args.account})\n")

    doc = build_doc(args.account, account_id)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(doc)
    print(f"\nWrote {len(doc.splitlines())} lines to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
