"""Datasource adapters backed by ds-threevictors connectors and S3 util."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    from threevictors.dao import mysql_connector, redshift_connector
    from threevictors.s3_util import s3_util
except Exception:  # pragma: no cover
    mysql_connector = None  # type: ignore[assignment]
    redshift_connector = None  # type: ignore[assignment]
    s3_util = None  # type: ignore[assignment]


class DatasourceDependencyError(RuntimeError):
    """Raised when ds-threevictors dependencies are unavailable."""


class CredentialsBootstrapError(RuntimeError):
    """Raised when assume-based credential bootstrap fails."""


class AnalyticsRedshiftReader(redshift_connector.RedshiftConnector if redshift_connector else object):
    def get_properties_filename(self):
        return "database-analytics-redshift-serverless-reader.properties"


class CoreRedshiftReader(redshift_connector.RedshiftConnector if redshift_connector else object):
    def get_properties_filename(self):
        return "database-core-redshift-serverless-reader.properties"


class PriceEyeMySQLReader(mysql_connector.MySQLConnector if mysql_connector else object):
    def get_properties_filename(self):
        return "database-priceeye-reader.properties"


@dataclass(frozen=True)
class TableRef:
    schema: str
    table: str


# Canonical datasource routing table
_TABLE_ROUTING: list[tuple[str, str]] = [
    ("priceeye.", "mysql_priceeye"),
    ("prod.monitoring", "redshift_core"),
    ("local.monitoring", "redshift_core"),
    ("collection_optimizer.", "redshift_core"),
    ("local.site_metrics", "redshift_core"),
    ("billing_db.", "redshift_core"),
    # Federated schemas (prod MySQL via Redshift external schema federation)
    # federated_scheduling only exists in the core cluster
    ("federated_scheduling.", "redshift_core"),
    # All other federated_* schemas fall through to redshift_analytics (default)
]


def datasource_for_table(table_name: str) -> str:
    """Single canonical routing function: table name -> datasource key."""
    normalized = table_name.strip().lower()
    for prefix, ds in _TABLE_ROUTING:
        if normalized.startswith(prefix):
            return ds
    return "redshift_analytics"


class DatasourceRegistry:
    """Unified read access for SQL and S3 with always-on 3VDEV bootstrap."""

    _DANGEROUS_SQL = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|call)\b", re.I)

    def __init__(self) -> None:
        self._cred_lock = threading.Lock()
        self._creds_ready = False
        self._s3 = s3_util.S3Util() if s3_util else None

    @staticmethod
    def parse_table_name(table_name: str) -> tuple[str, str, str]:
        raw = table_name.strip()
        parts = raw.split(".")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return "", parts[0], parts[1]
        raise ValueError(f"Unsupported table identifier: {table_name}")

    @staticmethod
    def _table_ref(table_name: str) -> TableRef:
        _, schema, table = DatasourceRegistry.parse_table_name(table_name)
        return TableRef(schema=schema, table=table)

    def ensure_credentials(self) -> dict[str, Any]:
        """Run `assume 3VDEV` once and load AWS env into current process."""
        with self._cred_lock:
            if self._creds_ready:
                return {"ok": True, "profile": "3VDEV", "cached": True}

            cmd = "assume 3VDEV >/dev/null 2>&1; env -0"
            proc = subprocess.run(
                ["zsh", "-lc", cmd],
                capture_output=True,
                text=False,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
                raise CredentialsBootstrapError(
                    f"Credential bootstrap failed for profile 3VDEV: {stderr.strip() or 'unknown error'}"
                )

            output = proc.stdout.decode("utf-8", errors="replace")
            loaded = 0
            for pair in output.split("\x00"):
                if not pair or "=" not in pair:
                    continue
                key, value = pair.split("=", 1)
                if key.startswith("AWS_"):
                    os.environ[key] = value
                    loaded += 1
            os.environ.setdefault("AWS_REGION", "us-east-1")

            if loaded == 0:
                fallback = subprocess.run(
                    ["granted", "credential-process", "--profile", "3VDEV", "--auto-login"],
                    capture_output=True,
                    text=True,
                )
                if fallback.returncode != 0:
                    stderr = fallback.stderr or ""
                    raise CredentialsBootstrapError(
                        f"assume produced no AWS env and credential-process fallback failed: {stderr.strip() or 'unknown error'}"
                    )
                payload = json.loads(fallback.stdout)
                os.environ["AWS_ACCESS_KEY_ID"] = str(payload.get("AccessKeyId") or "")
                os.environ["AWS_SECRET_ACCESS_KEY"] = str(payload.get("SecretAccessKey") or "")
                os.environ["AWS_SESSION_TOKEN"] = str(payload.get("SessionToken") or "")
                loaded = 3
            self._creds_ready = True
            return {"ok": True, "profile": "3VDEV", "cached": False, "env_keys_loaded": loaded}

    @staticmethod
    def _query_df(connector: Any, query: str) -> pd.DataFrame:
        with connector.get_connection().cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall() or []
        return pd.DataFrame(rows, columns=columns)

    def _connector(self, datasource: str) -> Any:
        self.ensure_credentials()
        if redshift_connector is None or mysql_connector is None:
            raise DatasourceDependencyError("ds-threevictors is not installed in this environment")
        if datasource == "redshift_core":
            return CoreRedshiftReader()
        if datasource == "redshift_analytics":
            return AnalyticsRedshiftReader()
        if datasource == "mysql_priceeye":
            return PriceEyeMySQLReader()
        raise ValueError(f"Unsupported datasource: {datasource}")

    def execute_sql(self, datasource: str, query: str) -> pd.DataFrame:
        if self._DANGEROUS_SQL.search(query):
            raise ValueError("Only read-only SQL is supported")
        connector = self._connector(datasource)
        return self._query_df(connector, query)

    def inspect_table_metadata(self, table_name: str, datasource: str) -> dict[str, Any]:
        table_ref = self._table_ref(table_name)
        columns: list[dict[str, Any]] = []
        partitions: list[dict[str, Any]] = []

        if datasource.startswith("redshift"):
            query = (
                "SELECT column_name, data_type, is_nullable "
                "FROM svv_columns "
                f"WHERE table_schema = '{table_ref.schema}' AND table_name = '{table_ref.table}' "
                "ORDER BY ordinal_position"
            )
            frame = self.execute_sql(datasource, query)
            for _, row in frame.iterrows():
                col = str(row.get("column_name", ""))
                dtype = str(row.get("data_type", ""))
                nullable = str(row.get("is_nullable", "YES")).upper() == "YES"
                columns.append(
                    {
                        "column_name": col,
                        "data_type": dtype,
                        "nullable": nullable,
                        "is_key": False,
                    }
                )
        elif datasource == "mysql_priceeye":
            query = f"DESCRIBE {table_ref.schema}.{table_ref.table}"
            frame = self.execute_sql(datasource, query)
            for _, row in frame.iterrows():
                col = str(row.get("Field", ""))
                dtype = str(row.get("Type", ""))
                nullable = str(row.get("Null", "YES")).upper() == "YES"
                columns.append(
                    {
                        "column_name": col,
                        "data_type": dtype,
                        "nullable": nullable,
                        "is_key": str(row.get("Key", "")) in {"PRI", "MUL"},
                    }
                )
        else:
            raise ValueError(f"Unsupported datasource for metadata inspection: {datasource}")

        for item in columns:
            key = item["column_name"].lower()
            if key in {"sales_date", "customer", "providercode", "sitecode"}:
                partitions.append(
                    {
                        "column": key,
                        "role": "recommended",
                        "inferred_type": "date" if key.endswith("date") else "categorical",
                    }
                )
            elif key.endswith("_date") or key == "date":
                partitions.append({"column": key, "role": "recommended", "inferred_type": "date"})

        return {
            "table_name": table_name,
            "datasource": datasource,
            "columns": columns,
            "partitions": partitions,
        }

    def mysql_lookup_codes(self, tokens: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"providers": [], "sites": [], "customers": []}
        cleaned = sorted({t.strip().upper() for t in tokens if t and t.strip()})
        if not cleaned:
            return out

        def _lookup(table: str, candidates: list[str]) -> list[str]:
            cols = self.inspect_table_metadata(table, "mysql_priceeye").get("columns", [])
            names = {str(col.get("column_name", "")).lower() for col in cols}
            match_col = next((c for c in candidates if c.lower() in names), None)
            if not match_col:
                return []
            in_clause = ", ".join(f"'{token.replace("'", "''")}'" for token in cleaned)
            query = (
                f"SELECT DISTINCT UPPER({match_col}) AS code FROM {table} "
                f"WHERE UPPER({match_col}) IN ({in_clause})"
            )
            frame = self.execute_sql("mysql_priceeye", query)
            if "code" not in frame.columns:
                return []
            return sorted({str(v).strip().upper() for v in frame["code"].tolist() if str(v).strip()})

        out["providers"] = _lookup("priceeye.provider", ["providercode", "provider_code", "code", "provider"])
        out["sites"] = _lookup("priceeye.site", ["sitecode", "site_code", "code", "site"])
        out["customers"] = _lookup("priceeye.customer", ["customer", "customercode", "customer_code", "code", "name"])
        return out

    def fetch_s3_data(
        self, bucket: str, key_or_prefix: str, *, max_files: int = 30
    ) -> tuple[pd.DataFrame, list[str]]:
        """Fetch CSV, Parquet, or JSONL data from S3."""
        self.ensure_credentials()
        if self._s3 is None:
            raise DatasourceDependencyError("ds-threevictors s3_util is not installed")

        s3_client = getattr(self._s3, "s3_client", None)
        if s3_client is None and hasattr(self._s3, "get_s3_client"):
            s3_client = self._s3.get_s3_client()
        if s3_client is None:
            raise DatasourceDependencyError("Unable to initialize S3 client from ds-threevictors S3Util")

        key = key_or_prefix.strip()
        supported_extensions = {".csv", ".parquet", ".jsonl", ".json"}
        keys: list[str] = []

        # Check if key is a direct file reference
        lower_key = key.lower()
        if any(lower_key.endswith(ext) for ext in supported_extensions):
            keys = [key]
        else:
            continuation: str | None = None
            while len(keys) < max_files:
                kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": key, "MaxKeys": 1000}
                if continuation:
                    kwargs["ContinuationToken"] = continuation
                response = s3_client.list_objects_v2(**kwargs)
                for item in response.get("Contents", []) or []:
                    k = str(item.get("Key", ""))
                    if any(k.lower().endswith(ext) for ext in supported_extensions):
                        keys.append(k)
                        if len(keys) >= max_files:
                            break
                if not response.get("IsTruncated"):
                    break
                continuation = response.get("NextContinuationToken")
                if not continuation:
                    break

        if not keys:
            raise FileNotFoundError(f"No supported data files found at s3://{bucket}/{key_or_prefix}")

        frames: list[pd.DataFrame] = []
        for object_key in keys:
            obj = s3_client.get_object(Bucket=bucket, Key=object_key)
            body = obj["Body"].read()
            if not body:
                continue

            lower_obj_key = object_key.lower()
            if lower_obj_key.endswith(".parquet"):
                frame = pd.read_parquet(io.BytesIO(body))
            elif lower_obj_key.endswith(".jsonl"):
                text = body.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                frame = pd.read_json(io.StringIO(text), lines=True)
            elif lower_obj_key.endswith(".json"):
                text = body.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                frame = pd.read_json(io.StringIO(text))
            else:
                # CSV with delimiter sniffing
                text = body.decode("utf-8", errors="replace")
                if not text.strip():
                    continue
                dialect = csv.Sniffer().sniff(text[: min(len(text), 4096)], delimiters=",|\t;")
                frame = pd.read_csv(io.StringIO(text), sep=dialect.delimiter)

            frame["_s3_key"] = object_key
            frames.append(frame)

        if not frames:
            raise FileNotFoundError(f"Data files were found but empty at s3://{bucket}/{key_or_prefix}")

        merged = pd.concat(frames, ignore_index=True)
        return merged, keys


__all__ = [
    "CredentialsBootstrapError",
    "DatasourceDependencyError",
    "DatasourceRegistry",
    "datasource_for_table",
]
