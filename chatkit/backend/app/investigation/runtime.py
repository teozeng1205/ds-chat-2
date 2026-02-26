"""Shell-first, KB-guided investigation runtime for DS Chat next generation."""

from __future__ import annotations

import contextlib
import datetime as dt
from decimal import Decimal
import glob
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

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

from threevictors.dao import mysql_connector, redshift_connector
from threevictors.s3_util import s3_util

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)
log.propagate = False

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent.parent
WORK_ROOT = BACKEND_ROOT / ".work"
SESSION_ROOT = WORK_ROOT / "sessions"
KB_ROOT = WORK_ROOT / "knowledge"
KB_DB_PATH = KB_ROOT / "knowledge_base.sqlite"

INVESTIGATION_ROOT = Path(__file__).resolve().parent
COMMON_CODES_PATH = INVESTIGATION_ROOT / "common_codes.json"
TASK_RECIPES_PATH = INVESTIGATION_ROOT / "task_recipes.json"
SQL_BEST_PRACTICES_PATH = INVESTIGATION_ROOT / "sql_best_practices.md"
TABLES_DOC_PATH = REPO_ROOT / "tables.md"

KB_VERSION = "2026.02.26-next-gen-v3"
DEFAULT_SQL_LIMIT = 1000
MAX_SQL_LIMIT = 6000
MAX_PREVIEW_ROWS = 200
DEFAULT_BUCKET_COUNT = 12
DEFAULT_KB_MAX_EXTERNAL_FILES = 120

TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "temporar",
    "connection",
    "reset",
    "throttl",
)

ANOMALY_BUCKET = "s3-atp-3victors-3vdev-use1-collection-anomalies"
CUSTOMER_PREFIX = "collection-customer/v1"
PROVIDER_PREFIX = "collection-provider/v1"
LATEREQUEST_PREFIX = "collection-laterequests/v1"

COMMON_TABLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "prod.monitoring.provider_combined_audit": {
        "datasource": "redshift_core",
        "tier": "common",
        "partitions": [
            {"column": "sales_date", "role": "required"},
            {"column": "customer", "role": "recommended"},
        ],
        "join_hints": ["join with prod.monitoring.combined_audit on shared dimensions"],
        "semantic_tags": ["site_issues", "monitoring", "provider", "impact"],
    },
    "prod.monitoring.combined_audit": {
        "datasource": "redshift_core",
        "tier": "common",
        "partitions": [
            {"column": "sales_date", "role": "required"},
            {"column": "customer", "role": "recommended"},
        ],
        "join_hints": ["join with provider_combined_audit when provider/site dimensions needed"],
        "semantic_tags": ["monitoring", "requests"],
    },
    "prod.analytics.market_level_anomalies_v3": {
        "datasource": "redshift_analytics",
        "tier": "common",
        "partitions": [
            {"column": "sales_date", "role": "recommended"},
            {"column": "customer", "role": "recommended"},
        ],
        "join_hints": ["join with priceeye.customer for enrichment"],
        "semantic_tags": ["market", "anomalies", "impact_score"],
    },
    "prod.analytics.market_level_anomalies_v4": {
        "datasource": "redshift_analytics",
        "tier": "common",
        "partitions": [
            {"column": "sales_date", "role": "recommended"},
            {"column": "customer", "role": "recommended"},
        ],
        "join_hints": ["newer variant of market anomalies"],
        "semantic_tags": ["market", "anomalies"],
    },
    "priceeye.provider": {
        "datasource": "mysql_priceeye",
        "tier": "common",
        "partitions": [],
        "join_hints": ["lookup provider code metadata"],
        "semantic_tags": ["provider", "reference"],
    },
    "priceeye.site": {
        "datasource": "mysql_priceeye",
        "tier": "common",
        "partitions": [],
        "join_hints": ["lookup site code metadata"],
        "semantic_tags": ["site", "reference"],
    },
    "priceeye.customer": {
        "datasource": "mysql_priceeye",
        "tier": "common",
        "partitions": [],
        "join_hints": ["lookup customer code metadata"],
        "semantic_tags": ["customer", "reference"],
    },
}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_code(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, default=_json_default)


def _extract_keywords(text: str) -> list[str]:
    if not text:
        return []
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9_]{2,}", text)]


def _mask_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) <= 3:
        return "***"
    return f"{text[:2]}***{text[-2:]}"


def _mask_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _mask_value(value) for key, value in row.items()}


def _coerce_sales_date(value: str | dt.date | dt.datetime | None) -> str:
    if value is None:
        return dt.date.today().strftime("%Y%m%d")
    if isinstance(value, dt.datetime):
        return value.date().strftime("%Y%m%d")
    if isinstance(value, dt.date):
        return value.strftime("%Y%m%d")
    raw = str(value).strip().lower()
    if raw in {"today", "now"}:
        return dt.date.today().strftime("%Y%m%d")
    if raw == "yesterday":
        return (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
    if raw == "tomorrow":
        return (dt.date.today() + dt.timedelta(days=1)).strftime("%Y%m%d")
    if len(raw) == 8 and raw.isdigit():
        return raw
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return dt.datetime.strptime(raw, "%Y-%m-%d").strftime("%Y%m%d")
    raise ValueError(f"Unsupported sales_date format: {value!r}")


def _extract_sales_date_from_text(text: str) -> str | None:
    lowered = text.lower()
    if "yesterday" in lowered:
        return (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
    if "tomorrow" in lowered:
        return (dt.date.today() + dt.timedelta(days=1)).strftime("%Y%m%d")
    if "today" in lowered:
        return dt.date.today().strftime("%Y%m%d")
    match_compact = re.search(r"\b(20\d{6})\b", text)
    if match_compact:
        return match_compact.group(1)
    match_dash = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if match_dash:
        return dt.datetime.strptime(match_dash.group(1), "%Y-%m-%d").strftime("%Y%m%d")
    return None


def _log_structured(event: str, **fields: Any) -> None:
    log.info(_json_dumps({"ts": _iso_now(), "event": event, **fields}))


class AnalyticsRedshiftReader(redshift_connector.RedshiftConnector):
    def get_properties_filename(self):
        return "database-analytics-redshift-serverless-reader.properties"


class CoreRedshiftReader(redshift_connector.RedshiftConnector):
    def get_properties_filename(self):
        return "database-core-redshift-serverless-reader.properties"


class PriceEyeMySQLReader(mysql_connector.MySQLConnector):
    def get_properties_filename(self):
        return "database-priceeye-reader.properties"


class DatasourceRegistry:
    """Datasource execution layer backed by ds-threevictors connectors."""

    def __init__(self) -> None:
        self._s3 = s3_util.S3Util()

    @staticmethod
    def _query_df(connector: Any, query: str) -> pd.DataFrame:
        with connector.get_connection().cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=colnames)

    def execute_sql(self, datasource: str, query: str) -> pd.DataFrame:
        connector: Any | None = None
        started = time.perf_counter()
        try:
            if datasource == "redshift_analytics":
                connector = AnalyticsRedshiftReader()
            elif datasource == "redshift_core":
                connector = CoreRedshiftReader()
            elif datasource == "mysql_priceeye":
                connector = PriceEyeMySQLReader()
            else:
                raise ValueError(f"Unsupported datasource: {datasource}")

            df = self._query_df(connector, query)
            _log_structured(
                "sql_query_complete",
                datasource=datasource,
                row_count=int(len(df)),
                latency_ms=int((time.perf_counter() - started) * 1000),
                query_hash=_stable_hash(query)[:16],
            )
            return df
        finally:
            if connector is not None:
                connector.close()

    def fetch_s3_csv(self, bucket: str, key: str) -> pd.DataFrame:
        content = self._s3.get_object(bucket, key)
        for sep in (None, ",", "\t"):
            try:
                if sep is None:
                    frame = pd.read_csv(io.StringIO(content), sep=None, engine="python")
                else:
                    frame = pd.read_csv(io.StringIO(content), sep=sep)
                if not frame.empty or len(frame.columns) > 1:
                    frame.columns = [str(col).strip() for col in frame.columns]
                    return frame
            except Exception:
                continue
        return pd.DataFrame()

    def list_s3_csv_keys(self, bucket: str, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self._s3.s3_client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                key = item.get("Key")
                if isinstance(key, str) and key.endswith(".csv"):
                    keys.append(key)
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        return sorted(keys)

    def fetch_s3_prefix_csv(self, bucket: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
        keys = self.list_s3_csv_keys(bucket, prefix)
        frames: list[pd.DataFrame] = []
        for key in keys:
            df = self.fetch_s3_csv(bucket, key)
            if df.empty:
                continue
            tagged = df.copy()
            tagged["source_key"] = key
            frames.append(tagged)
        if not frames:
            return pd.DataFrame(), keys
        return pd.concat(frames, ignore_index=True), keys

    def mysql_lookup_codes(self, tokens: Sequence[str]) -> dict[str, list[str]]:
        normalized = sorted({_normalize_code(token) for token in tokens if _normalize_code(token)})
        if not normalized:
            return {"providers": [], "sites": [], "customers": []}

        connector: PriceEyeMySQLReader | None = None
        try:
            connector = PriceEyeMySQLReader()
            provider_col = self._discover_mysql_code_column(connector, "priceeye.provider")
            site_col = self._discover_mysql_code_column(connector, "priceeye.site")
            customer_col = self._discover_mysql_code_column(connector, "priceeye.customer")
            return {
                "providers": self._lookup_codes_in_table(connector, "priceeye.provider", provider_col, normalized),
                "sites": self._lookup_codes_in_table(connector, "priceeye.site", site_col, normalized),
                "customers": self._lookup_codes_in_table(connector, "priceeye.customer", customer_col, normalized),
            }
        finally:
            if connector is not None:
                connector.close()

    def inspect_table_metadata(self, table_name: str, datasource: str) -> dict[str, Any]:
        if datasource == "mysql_priceeye":
            return self._inspect_mysql_table(table_name)
        return self._inspect_redshift_table(table_name, datasource)

    def _inspect_mysql_table(self, table_name: str) -> dict[str, Any]:
        connector = PriceEyeMySQLReader()
        try:
            query = f"DESCRIBE {table_name};"
            df = self._query_df(connector, query)
            columns: list[dict[str, Any]] = []
            for row in df.to_dict(orient="records"):
                columns.append(
                    {
                        "column_name": str(row.get("Field") or row.get("field")),
                        "data_type": str(row.get("Type") or row.get("type") or "unknown"),
                        "nullable": str(row.get("Null") or "").upper() == "YES",
                        "is_key": str(row.get("Key") or "") in {"PRI", "MUL"},
                    }
                )
            return {
                "table_name": table_name,
                "datasource": "mysql_priceeye",
                "columns": columns,
            }
        finally:
            connector.close()

    def _inspect_redshift_table(self, table_name: str, datasource: str) -> dict[str, Any]:
        connector: Any
        if datasource == "redshift_core":
            connector = CoreRedshiftReader()
        else:
            connector = AnalyticsRedshiftReader()

        try:
            schema, table = self._split_schema_table(table_name)
            query = (
                "SELECT column_name, data_type, is_nullable "
                "FROM SVV_COLUMNS "
                f"WHERE table_schema = '{schema}' AND table_name = '{table}' "
                "ORDER BY ordinal_position"
            )
            df = self._query_df(connector, query)
            columns = [
                {
                    "column_name": str(row.get("column_name")),
                    "data_type": str(row.get("data_type") or "unknown"),
                    "nullable": str(row.get("is_nullable") or "").upper() == "YES",
                    "is_key": False,
                }
                for row in df.to_dict(orient="records")
            ]
            return {
                "table_name": table_name,
                "datasource": datasource,
                "columns": columns,
            }
        finally:
            connector.close()

    @staticmethod
    def _split_schema_table(table_name: str) -> tuple[str, str]:
        parts = table_name.split(".")
        if len(parts) >= 3:
            return parts[-2], parts[-1]
        if len(parts) == 2:
            return parts[0], parts[1]
        return "public", parts[0]

    def _discover_mysql_code_column(self, connector: PriceEyeMySQLReader, table: str) -> str | None:
        query = f"DESCRIBE {table};"
        with connector.get_connection().cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        columns = [str(row[0]).strip().lower() for row in rows if row]
        for candidate in ("code", "providercode", "sitecode", "customer", "customercode", "name"):
            if candidate in columns:
                return candidate
        return columns[0] if columns else None

    def _lookup_codes_in_table(
        self,
        connector: PriceEyeMySQLReader,
        table: str,
        column: str | None,
        normalized_tokens: Sequence[str],
    ) -> list[str]:
        if not column:
            return []
        escaped = "', '".join(token.replace("'", "''") for token in normalized_tokens)
        query = (
            f"SELECT DISTINCT UPPER(TRIM({column})) AS code "
            f"FROM {table} WHERE UPPER(TRIM({column})) IN ('{escaped}')"
        )
        with connector.get_connection().cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        return sorted({_normalize_code(row[0]) for row in rows if row and row[0]})


class LocalCodeCatalog:
    def __init__(self, path: Path = COMMON_CODES_PATH) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, set[str]]:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                _json_pretty(
                    {
                        "providers": ["QL2", "AA", "DL", "SK", "LH"],
                        "sites": ["AV", "QF", "DY"],
                        "customers": ["AA", "B6", "DL", "SK", "LH"],
                        "customer_sites": ["AA|AV", "B6|QF"],
                    }
                ),
                encoding="utf-8",
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "providers": {_normalize_code(item) for item in payload.get("providers", []) if item},
            "sites": {_normalize_code(item) for item in payload.get("sites", []) if item},
            "customers": {_normalize_code(item) for item in payload.get("customers", []) if item},
            "customer_sites": {
                str(item).strip().upper() for item in payload.get("customer_sites", []) if str(item).strip()
            },
        }

    def _persist(self) -> None:
        payload = {
            "providers": sorted(self._data["providers"]),
            "sites": sorted(self._data["sites"]),
            "customers": sorted(self._data["customers"]),
            "customer_sites": sorted(self._data["customer_sites"]),
        }
        self.path.write_text(_json_pretty(payload), encoding="utf-8")

    def resolve_tokens(self, tokens: Iterable[str]) -> dict[str, list[str]]:
        normalized = {_normalize_code(token) for token in tokens if _normalize_code(token)}
        return {
            "providers": sorted(token for token in normalized if token in self._data["providers"]),
            "sites": sorted(token for token in normalized if token in self._data["sites"]),
            "customers": sorted(token for token in normalized if token in self._data["customers"]),
        }

    def add_codes(self, *, providers: Iterable[str] = (), sites: Iterable[str] = (), customers: Iterable[str] = ()) -> None:
        with self._lock:
            self._data["providers"].update({_normalize_code(item) for item in providers if _normalize_code(item)})
            self._data["sites"].update({_normalize_code(item) for item in sites if _normalize_code(item)})
            self._data["customers"].update({_normalize_code(item) for item in customers if _normalize_code(item)})
            self._persist()

    def kb_rows(self) -> list[tuple[str, str, str, str]]:
        now = _iso_now()
        rows: list[tuple[str, str, str, str]] = []
        for item in sorted(self._data["providers"]):
            rows.append((item, "provider", "common_codes", now))
        for item in sorted(self._data["sites"]):
            rows.append((item, "site", "common_codes", now))
        for item in sorted(self._data["customers"]):
            rows.append((item, "customer", "common_codes", now))
        return rows


class EntityResolver:
    def __init__(self, catalog: LocalCodeCatalog, registry: DatasourceRegistry) -> None:
        self.catalog = catalog
        self.registry = registry

    def resolve(self, input_text: str, sales_date_hint: str | None = None) -> dict[str, Any]:
        extracted = self._extract_tokens(input_text)
        local = self.catalog.resolve_tokens(extracted["tokens"])

        providers = set(local["providers"])
        sites = set(local["sites"])
        customers = set(local["customers"])

        for provider, site in extracted["provider_site_pairs"]:
            providers.add(provider)
            sites.add(site)

        providers.update(extracted["provider_hints"])
        sites.update(extracted["site_hints"])
        customers.update(extracted["customer_hints"])

        unresolved = sorted(
            token
            for token in extracted["tokens"]
            if token not in providers and token not in sites and token not in customers
        )

        mysql_hits = {"providers": [], "sites": [], "customers": []}
        if unresolved:
            try:
                mysql_hits = self.registry.mysql_lookup_codes(unresolved)
                providers.update(mysql_hits["providers"])
                sites.update(mysql_hits["sites"])
                customers.update(mysql_hits["customers"])
                self.catalog.add_codes(
                    providers=mysql_hits["providers"],
                    sites=mysql_hits["sites"],
                    customers=mysql_hits["customers"],
                )
            except Exception as exc:  # noqa: BLE001
                _log_structured("entity_mysql_lookup_failed", error=str(exc), unresolved_count=len(unresolved))

        unresolved_after = sorted(
            token
            for token in unresolved
            if token not in providers and token not in sites and token not in customers
        )

        sales_date = _coerce_sales_date(sales_date_hint or _extract_sales_date_from_text(input_text))

        return {
            "providers": sorted(providers),
            "sites": sorted(sites),
            "customers": sorted(customers),
            "unresolved_tokens": unresolved_after,
            "mysql_hits": mysql_hits,
            "sales_date_hint": sales_date,
        }

    def _extract_tokens(self, text: str) -> dict[str, Any]:
        raw_tokens = {_normalize_code(tok) for tok in re.findall(r"[A-Za-z0-9|]{2,}", text)}
        stopwords = {
            "WHAT",
            "ARE",
            "THE",
            "TOP",
            "SITE",
            "SITES",
            "ISSUE",
            "ISSUES",
            "FOR",
            "AND",
            "WITH",
            "TODAY",
            "YESTERDAY",
            "TOMORROW",
            "IMPACT",
            "SCORE",
            "DISTRIBUTION",
            "CUSTOMER",
            "PROVIDER",
            "ANOMALIES",
            "ANOMALY",
            "MARKET",
            "SHOW",
            "TABLE",
            "QUERY",
        }
        tokens = sorted(
            token
            for token in raw_tokens
            if token
            and token not in stopwords
            and len(token) <= 10
            and (any(ch.isdigit() for ch in token) or len(token) <= 4)
        )

        pairs: list[tuple[str, str]] = []
        for left, right in re.findall(r"\b([A-Za-z0-9]{2,8})\|([A-Za-z0-9]{2,8})\b", text):
            pairs.append((_normalize_code(left), _normalize_code(right)))

        provider_hints = {
            _normalize_code(match)
            for match in re.findall(r"\bprovider\s+([A-Za-z0-9]{2,10})\b", text, flags=re.IGNORECASE)
        }
        site_hints = {
            _normalize_code(match)
            for match in re.findall(r"\bsite\s+([A-Za-z0-9]{2,10})\b", text, flags=re.IGNORECASE)
        }
        customer_hints = {
            _normalize_code(match)
            for match in re.findall(r"\bcustomer\s+([A-Za-z0-9]{2,10})\b", text, flags=re.IGNORECASE)
        }

        return {
            "tokens": tokens,
            "provider_site_pairs": pairs,
            "provider_hints": {tok for tok in provider_hints if tok},
            "site_hints": {tok for tok in site_hints if tok},
            "customer_hints": {tok for tok in customer_hints if tok},
        }


class SqlGuard:
    """Minimal guardrails: read-only + single statement + bounded limit."""

    def ensure_read_only(self, query: str) -> None:
        normalized = " ".join(query.strip().split())
        upper = normalized.upper()
        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            raise ValueError("Only SELECT/WITH queries are allowed")
        forbidden = (" INSERT ", " UPDATE ", " DELETE ", " DROP ", " ALTER ", " TRUNCATE ", " CREATE ", " GRANT ", " REVOKE ")
        padded = f" {upper} "
        for token in forbidden:
            if token in padded:
                raise ValueError(f"Forbidden SQL token: {token.strip()}")

    def ensure_single_statement(self, query: str) -> None:
        stripped = query.strip()
        if ";" in stripped.rstrip(";"):
            raise ValueError("Only single-statement SQL is allowed")

    def ensure_limit(self, query: str, *, default_limit: int = DEFAULT_SQL_LIMIT, max_limit: int = MAX_SQL_LIMIT) -> str:
        cleaned = query.strip().rstrip(";")
        match = re.search(r"\bLIMIT\s+(\d+)\b", cleaned, flags=re.IGNORECASE)
        if match:
            limit = int(match.group(1))
            if limit > max_limit:
                cleaned = re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {max_limit}", cleaned, flags=re.IGNORECASE)
        else:
            cleaned = f"{cleaned} LIMIT {default_limit}"
        return cleaned + ";"

    def validate(self, query: str) -> str:
        self.ensure_read_only(query)
        self.ensure_single_statement(query)
        return self.ensure_limit(query)


class KnowledgeBase:
    def __init__(self, db_path: Path, catalog: LocalCodeCatalog, registry: DatasourceRegistry) -> None:
        self.db_path = db_path
        self.catalog = catalog
        self.registry = registry
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kb_tables (
                    table_name TEXT PRIMARY KEY,
                    datasource TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    notes TEXT,
                    join_hints_json TEXT NOT NULL,
                    semantic_tags_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kb_partitions (
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    inferred_type TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(table_name, column_name)
                );

                CREATE TABLE IF NOT EXISTS kb_columns (
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    data_type TEXT,
                    nullable INTEGER,
                    semantic_tags_json TEXT NOT NULL,
                    is_key INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(table_name, column_name)
                );

                CREATE TABLE IF NOT EXISTS kb_example_rows (
                    table_name TEXT PRIMARY KEY,
                    example_json_masked TEXT NOT NULL,
                    sample_query_used TEXT,
                    mask_policy TEXT NOT NULL,
                    captured_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kb_relationships (
                    id TEXT PRIMARY KEY,
                    left_table TEXT NOT NULL,
                    right_table TEXT NOT NULL,
                    join_keys TEXT,
                    relationship_type TEXT,
                    confidence REAL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kb_query_patterns (
                    id TEXT PRIMARY KEY,
                    intent TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kb_codes (
                    code TEXT NOT NULL,
                    code_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(code, code_type)
                );

                CREATE TABLE IF NOT EXISTS kb_documents (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    checksum TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES kb_documents(id)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content
                );

                CREATE TABLE IF NOT EXISTS kb_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def ensure_ready(self) -> dict[str, Any]:
        self.ensure_schema()
        current_hashes = self._source_hashes()
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kb_meta WHERE key='source_hashes'").fetchone()
            old_hashes = json.loads(row["value"]) if row else {}
        if old_hashes != current_hashes:
            return self.refresh(force=True)
        return {"ok": True, "refreshed": False}

    def refresh(self, force: bool = False) -> dict[str, Any]:
        del force
        with self._lock:
            self.ensure_schema()
            now = _iso_now()
            doc_count = 0
            chunk_count = 0
            with self._connect() as conn:
                conn.execute("DELETE FROM kb_tables")
                conn.execute("DELETE FROM kb_partitions")
                conn.execute("DELETE FROM kb_columns")
                conn.execute("DELETE FROM kb_relationships")
                conn.execute("DELETE FROM kb_query_patterns")
                conn.execute("DELETE FROM kb_codes")
                conn.execute("DELETE FROM kb_documents")
                conn.execute("DELETE FROM kb_chunks")
                conn.execute("DELETE FROM kb_chunks_fts")

                table_rows = self._collect_table_rows()
                for row in table_rows:
                    conn.execute(
                        """
                        INSERT INTO kb_tables (table_name, datasource, tier, notes, join_hints_json, semantic_tags_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["table_name"],
                            row["datasource"],
                            row["tier"],
                            row.get("notes", ""),
                            _json_dumps(row.get("join_hints", [])),
                            _json_dumps(row.get("semantic_tags", [])),
                            now,
                        ),
                    )
                    for part in row.get("partitions", []):
                        conn.execute(
                            """
                            INSERT INTO kb_partitions (table_name, column_name, role, inferred_type, updated_at)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                row["table_name"],
                                part.get("column"),
                                part.get("role", "recommended"),
                                part.get("inferred_type", "unknown"),
                                now,
                            ),
                        )

                self._seed_relationships(conn, now)
                self._seed_query_patterns(conn, now)
                conn.executemany(
                    "INSERT INTO kb_codes (code, code_type, source, updated_at) VALUES (?, ?, ?, ?)",
                    self.catalog.kb_rows(),
                )

                common_tables = [name for name, meta in COMMON_TABLE_DEFAULTS.items() if meta.get("tier") == "common"]
                for table_name in common_tables:
                    try:
                        self._upsert_table_metadata_from_inspection(conn, table_name)
                    except Exception as exc:  # noqa: BLE001
                        _log_structured("kb_common_table_inspection_failed", table_name=table_name, error=str(exc))

                for source_path in self._collect_source_files():
                    text = self._read_file(source_path)
                    if not text.strip():
                        continue
                    doc_id = _stable_hash(str(source_path))[:24]
                    conn.execute(
                        "INSERT INTO kb_documents (id, source_path, checksum, updated_at) VALUES (?, ?, ?, ?)",
                        (doc_id, str(source_path), _stable_hash(text), now),
                    )
                    chunks = self._chunk_text(text)
                    for idx, chunk in enumerate(chunks):
                        chunk_id = _stable_hash(f"{doc_id}:{idx}:{chunk[:80]}")[:40]
                        conn.execute(
                            "INSERT INTO kb_chunks (id, document_id, chunk_index, content, created_at) VALUES (?, ?, ?, ?, ?)",
                            (chunk_id, doc_id, idx, chunk, now),
                        )
                        conn.execute("INSERT INTO kb_chunks_fts (chunk_id, content) VALUES (?, ?)", (chunk_id, chunk))
                    doc_count += 1
                    chunk_count += len(chunks)

                conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('kb_version', ?)", (KB_VERSION,))
                conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('last_refresh', ?)", (now,))
                conn.execute(
                    "INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('source_hashes', ?)",
                    (_json_dumps(self._source_hashes()),),
                )

            return {"ok": True, "refreshed": True, "documents": doc_count, "chunks": chunk_count}

    def retrieve(self, intent: str, question: str, entities: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
        self.ensure_schema()
        keywords = _extract_keywords(question)
        with self._connect() as conn:
            lexical_hits = self._lexical_hits(conn, keywords, top_k=top_k)
            candidate_tables = self._candidate_tables(conn, intent, question)
            partition_requirements = self._partition_requirements(conn, candidate_tables)
            patterns = [
                dict(row)
                for row in conn.execute(
                    "SELECT intent, name, description, payload_json FROM kb_query_patterns WHERE intent=? OR intent='generic'",
                    (intent,),
                ).fetchall()
            ]
            for item in patterns:
                item["payload"] = json.loads(item.pop("payload_json"))

        return {
            "intent": intent,
            "entities": entities,
            "candidate_tables": candidate_tables,
            "partition_requirements": partition_requirements,
            "query_patterns": patterns,
            "evidence_chunks": lexical_hits,
        }

    def upsert_discovered_table(
        self,
        *,
        table_name: str,
        datasource: str,
        columns: list[dict[str, Any]],
        partitions: list[dict[str, Any]],
        example_row: dict[str, Any] | None,
        sample_query: str | None,
    ) -> None:
        now = _iso_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO kb_tables (table_name, datasource, tier, notes, join_hints_json, semantic_tags_json, updated_at)
                VALUES (?, ?, 'discovered', ?, ?, ?, ?)
                """,
                (
                    table_name,
                    datasource,
                    "Auto-discovered via inspect_table_metadata",
                    _json_dumps([]),
                    _json_dumps(["discovered"]),
                    now,
                ),
            )
            conn.execute("DELETE FROM kb_partitions WHERE table_name=?", (table_name,))
            for part in partitions:
                conn.execute(
                    "INSERT INTO kb_partitions (table_name, column_name, role, inferred_type, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (table_name, part.get("column"), part.get("role", "recommended"), part.get("inferred_type", "unknown"), now),
                )

            conn.execute("DELETE FROM kb_columns WHERE table_name=?", (table_name,))
            for column in columns:
                conn.execute(
                    """
                    INSERT INTO kb_columns (table_name, column_name, data_type, nullable, semantic_tags_json, is_key, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        table_name,
                        column.get("column_name"),
                        column.get("data_type", "unknown"),
                        int(bool(column.get("nullable", False))),
                        _json_dumps(column.get("semantic_tags", [])),
                        int(bool(column.get("is_key", False))),
                        now,
                    ),
                )

            if example_row is not None:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO kb_example_rows (table_name, example_json_masked, sample_query_used, mask_policy, captured_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (table_name, _json_dumps(example_row), sample_query or "", "masked", now),
                )

    def _upsert_table_metadata_from_inspection(self, conn: sqlite3.Connection, table_name: str) -> None:
        datasource = self._datasource_for_table(table_name)
        inspection = self.registry.inspect_table_metadata(table_name, datasource)
        columns = inspection.get("columns", [])
        now = _iso_now()
        conn.execute("DELETE FROM kb_columns WHERE table_name=?", (table_name,))
        for column in columns:
            conn.execute(
                """
                INSERT INTO kb_columns (table_name, column_name, data_type, nullable, semantic_tags_json, is_key, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table_name,
                    column.get("column_name"),
                    column.get("data_type", "unknown"),
                    int(bool(column.get("nullable", False))),
                    _json_dumps([]),
                    int(bool(column.get("is_key", False))),
                    now,
                ),
            )

        example_query = f"SELECT * FROM {table_name} LIMIT 1"
        try:
            example_df = self.registry.execute_sql(datasource, example_query)
            if not example_df.empty:
                masked = _mask_row(example_df.iloc[0].to_dict())
                conn.execute(
                    "INSERT OR REPLACE INTO kb_example_rows (table_name, example_json_masked, sample_query_used, mask_policy, captured_at) VALUES (?, ?, ?, ?, ?)",
                    (table_name, _json_dumps(masked), example_query, "masked", now),
                )
        except Exception:
            pass

    def _collect_table_rows(self) -> list[dict[str, Any]]:
        rows = self._parse_tables_doc(TABLES_DOC_PATH)
        merged: dict[str, dict[str, Any]] = {row["table_name"]: row for row in rows}
        for table_name, meta in COMMON_TABLE_DEFAULTS.items():
            if table_name not in merged:
                merged[table_name] = {
                    "table_name": table_name,
                    "datasource": meta["datasource"],
                    "tier": meta.get("tier", "common"),
                    "notes": "seeded common table",
                    "partitions": meta.get("partitions", []),
                    "join_hints": meta.get("join_hints", []),
                    "semantic_tags": meta.get("semantic_tags", []),
                }
            else:
                current = merged[table_name]
                if not current.get("partitions"):
                    current["partitions"] = meta.get("partitions", [])
                if not current.get("join_hints"):
                    current["join_hints"] = meta.get("join_hints", [])
                if not current.get("semantic_tags"):
                    current["semantic_tags"] = meta.get("semantic_tags", [])
                current["tier"] = "common"

        return sorted(merged.values(), key=lambda item: item["table_name"])

    def _parse_tables_doc(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        current_datasource = "redshift_analytics"
        in_table = False
        rows: list[dict[str, Any]] = []

        for raw in lines:
            line = raw.strip()
            lower = line.lower()
            if line.startswith("### ") and "database-" in lower:
                in_table = False
                if "database-core-redshift" in lower:
                    current_datasource = "redshift_core"
                elif "database-priceeye-reader" in lower:
                    current_datasource = "mysql_priceeye"
                else:
                    current_datasource = "redshift_analytics"
                continue

            if line.startswith("#### Tables queried"):
                in_table = True
                continue

            if in_table and line.startswith("### "):
                in_table = False

            if not in_table or not line.startswith("|"):
                continue
            if "---" in line or ("Table" in line and "Notes" in line):
                continue

            parts = [segment.strip() for segment in line.strip("|").split("|")]
            if len(parts) < 1:
                continue
            table_name = parts[0]
            if not table_name or table_name.startswith("{"):
                continue

            rows.append(
                {
                    "table_name": table_name,
                    "datasource": self._datasource_for_table(table_name) or current_datasource,
                    "tier": "common" if table_name in COMMON_TABLE_DEFAULTS else "reference",
                    "notes": parts[1] if len(parts) > 1 else "",
                    "partitions": COMMON_TABLE_DEFAULTS.get(table_name, {}).get("partitions", []),
                    "join_hints": COMMON_TABLE_DEFAULTS.get(table_name, {}).get("join_hints", []),
                    "semantic_tags": COMMON_TABLE_DEFAULTS.get(table_name, {}).get("semantic_tags", []),
                }
            )

        dedup: dict[str, dict[str, Any]] = {}
        for row in rows:
            existing = dedup.get(row["table_name"])
            if existing is None:
                dedup[row["table_name"]] = row
                continue
            if existing["tier"] != "common" and row["tier"] == "common":
                dedup[row["table_name"]] = row
        return list(dedup.values())

    def _collect_source_files(self) -> list[Path]:
        files = [TABLES_DOC_PATH, COMMON_CODES_PATH, TASK_RECIPES_PATH, SQL_BEST_PRACTICES_PATH]
        external_max = int(os.getenv("INVESTIGATION_KB_MAX_EXTERNAL_FILES", str(DEFAULT_KB_MAX_EXTERNAL_FILES)))
        external_patterns = [
            str(Path("~/git/ds-*").expanduser() / "**" / "*.md"),
            str(Path("~/git/ds-*").expanduser() / "**" / "*.sql"),
        ]
        external_files: list[str] = []
        for pattern in external_patterns:
            external_files.extend(glob.glob(pattern, recursive=True))
        for file_name in sorted(set(external_files))[:external_max]:
            path = Path(file_name)
            if path.is_file():
                files.append(path)
        return [path for path in files if path.exists()]

    def _source_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for source in self._collect_source_files():
            hashes[str(source)] = _stable_hash(self._read_file(source))
        return hashes

    @staticmethod
    def _read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _chunk_text(text: str, *, size: int = 1200, overlap: int = 200) -> list[str]:
        clean = text.strip()
        if not clean:
            return []
        chunks: list[str] = []
        cursor = 0
        while cursor < len(clean):
            end = min(len(clean), cursor + size)
            chunks.append(clean[cursor:end])
            if end >= len(clean):
                break
            cursor = max(end - overlap, cursor + 1)
        return chunks

    @staticmethod
    def _seed_relationships(conn: sqlite3.Connection, now: str) -> None:
        relationships = [
            (
                _stable_hash("rel:provider_combined_vs_combined")[:24],
                "prod.monitoring.provider_combined_audit",
                "prod.monitoring.combined_audit",
                "sales_date,customer,provider/site dimensions",
                "same_domain",
                0.9,
                now,
            ),
            (
                _stable_hash("rel:market_anomaly_customer")[:24],
                "prod.analytics.market_level_anomalies_v3",
                "priceeye.customer",
                "customer",
                "dimension_lookup",
                0.85,
                now,
            ),
        ]
        conn.executemany(
            "INSERT INTO kb_relationships (id, left_table, right_table, join_keys, relationship_type, confidence, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            relationships,
        )

    @staticmethod
    def _seed_query_patterns(conn: sqlite3.Connection, now: str) -> None:
        payloads = [
            {
                "intent": "top_site_issues",
                "name": "top_site_issues",
                "description": "Top site issues by provider/site",
                "payload": {
                    "datasource": "redshift_core",
                    "template": (
                        "SELECT issue_sources, issue_reasons, providercode, sitecode, COUNT(*) AS issue_count "
                        "FROM prod.monitoring.provider_combined_audit "
                        "WHERE sales_date = {{sales_date}} {{provider_filter}} {{site_filter}} "
                        "AND issue_sources <> 'request' AND issue_sources <> '' AND issue_reasons <> '' "
                        "GROUP BY issue_sources, issue_reasons, providercode, sitecode ORDER BY issue_count DESC"
                    ),
                },
            },
            {
                "intent": "market_anomalies_distribution",
                "name": "market_anomalies",
                "description": "Market anomalies for customer",
                "payload": {
                    "datasource": "redshift_analytics",
                    "template": (
                        "SELECT observation_date, mkt, seg, top_offenders, cp, dow, impact_score, customer, sales_date "
                        "FROM prod.analytics.market_level_anomalies_v3 "
                        "WHERE sales_date = {{sales_date}} {{customer_filter}} AND any_anomaly = 1 "
                        "ORDER BY impact_score DESC"
                    ),
                },
            },
            {
                "intent": "customer_collection_anomalies",
                "name": "customer_collection_s3",
                "description": "Customer collection anomalies from S3",
                "payload": {
                    "bucket": ANOMALY_BUCKET,
                    "key_template": f"{CUSTOMER_PREFIX}/{{{{yyyy}}}}/{{{{mm}}}}/{{{{dd}}}}/collect_anomaly_{{{{sales_date}}}}.csv",
                },
            },
            {
                "intent": "generic",
                "name": "table_preview",
                "description": "Preview any explicit table",
                "payload": {
                    "template": "SELECT * FROM {{table_name}} LIMIT 200",
                },
            },
        ]
        for item in payloads:
            conn.execute(
                "INSERT INTO kb_query_patterns (id, intent, name, description, payload_json, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _stable_hash(f"pattern:{item['intent']}:{item['name']}")[:24],
                    item["intent"],
                    item["name"],
                    item["description"],
                    _json_dumps(item["payload"]),
                    now,
                ),
            )

    @staticmethod
    def _datasource_for_table(table_name: str) -> str:
        if table_name.startswith("priceeye."):
            return "mysql_priceeye"
        if table_name.startswith("prod.monitoring") or table_name.startswith("local.monitoring"):
            return "redshift_core"
        return "redshift_analytics"

    def _lexical_hits(self, conn: sqlite3.Connection, keywords: Sequence[str], top_k: int) -> list[dict[str, Any]]:
        if not keywords:
            return []
        match_query = " OR ".join(dict.fromkeys(keywords[:8]))
        rows = conn.execute(
            """
            SELECT f.chunk_id, bm25(kb_chunks_fts) AS rank, c.content, d.source_path
            FROM kb_chunks_fts f
            JOIN kb_chunks c ON c.id = f.chunk_id
            JOIN kb_documents d ON d.id = c.document_id
            WHERE kb_chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_query, top_k),
        ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "score": float(-row["rank"]),
                "source_path": row["source_path"],
                "snippet": row["content"][:240],
            }
            for row in rows
        ]

    def _candidate_tables(self, conn: sqlite3.Connection, intent: str, question: str) -> list[str]:
        base_by_intent = {
            "top_site_issues": ["prod.monitoring.provider_combined_audit", "prod.monitoring.combined_audit"],
            "customer_collection_anomalies": ["prod.monitoring.provider_combined_audit"],
            "market_anomalies_distribution": ["prod.analytics.market_level_anomalies_v3"],
        }
        candidates = set(base_by_intent.get(intent, []))
        for table in re.findall(r"\b[A-Za-z_]+\.[A-Za-z_]+(?:\.[A-Za-z0-9_]+)?\b", question):
            candidates.add(table)
        rows = conn.execute("SELECT table_name, semantic_tags_json FROM kb_tables").fetchall()
        question_tokens = set(_extract_keywords(question))
        for row in rows:
            tags = {str(tag).lower() for tag in json.loads(row["semantic_tags_json"])}
            if tags.intersection(question_tokens):
                candidates.add(row["table_name"])
        return sorted(candidates)

    def _partition_requirements(self, conn: sqlite3.Connection, table_names: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        if not table_names:
            return {}
        rows = conn.execute(
            "SELECT table_name, column_name, role, inferred_type FROM kb_partitions WHERE table_name IN ({})".format(
                ",".join("?" for _ in table_names)
            ),
            tuple(table_names),
        ).fetchall()
        out: dict[str, list[dict[str, Any]]] = {table: [] for table in table_names}
        for row in rows:
            out[row["table_name"]].append(
                {
                    "column": row["column_name"],
                    "role": row["role"],
                    "inferred_type": row["inferred_type"],
                }
            )
        return out


class WorkspaceManager:
    def __init__(self, root: Path = SESSION_ROOT) -> None:
        self.root = root
        self._lock = threading.RLock()

    def start_run(self, thread_id: str) -> str:
        with self._lock:
            run_id = f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
            run_dir = self._run_dir(thread_id, run_id)
            (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
            (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
            manifest = {
                "thread_id": thread_id,
                "run_id": run_id,
                "created_at": _iso_now(),
                "datasets": [],
                "analyses": [],
                "audit": [],
            }
            self._write_manifest(thread_id, run_id, manifest)
            return run_id

    def save_dataset(
        self,
        *,
        thread_id: str,
        run_id: str,
        df: pd.DataFrame,
        source_metadata: dict[str, Any],
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            dataset_id = uuid.uuid4().hex[:16]
            run_dir = self._run_dir(thread_id, run_id)
            datasets_dir = run_dir / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{dataset_name}_{dataset_id}" if dataset_name else dataset_id
            path = datasets_dir / f"{stem}.parquet"
            output_format = "parquet"
            try:
                df.to_parquet(path, index=False)
            except Exception:
                path = datasets_dir / f"{stem}.csv"
                df.to_csv(path, index=False)
                output_format = "csv_fallback"

            record = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "local_path": str(path),
                "row_count": int(len(df)),
                "schema": [{"name": str(column), "dtype": str(dtype)} for column, dtype in df.dtypes.items()],
                "timestamp": _iso_now(),
                "format": output_format,
                "source_metadata": source_metadata,
            }

            manifest = self._read_manifest(thread_id, run_id)
            manifest.setdefault("datasets", []).append(record)
            self._write_manifest(thread_id, run_id, manifest)
            return record

    def read_datasets(self, *, thread_id: str, run_id: str, dataset_ids: Sequence[str]) -> dict[str, pd.DataFrame]:
        manifest = self._read_manifest(thread_id, run_id)
        lookup = {
            item["dataset_id"]: item
            for item in manifest.get("datasets", [])
            if isinstance(item, dict) and "dataset_id" in item
        }
        frames: dict[str, pd.DataFrame] = {}
        for dataset_id in dataset_ids:
            item = lookup.get(dataset_id)
            if not item:
                continue
            path = Path(item["local_path"])
            if not path.exists():
                continue
            if path.suffix.lower() == ".parquet":
                frames[dataset_id] = pd.read_parquet(path)
            else:
                frames[dataset_id] = pd.read_csv(path)
        return frames

    def list_dataset_records(self, *, thread_id: str, run_id: str) -> list[dict[str, Any]]:
        manifest = self._read_manifest(thread_id, run_id)
        return [item for item in manifest.get("datasets", []) if isinstance(item, dict)]

    def record_analysis(self, *, thread_id: str, run_id: str, analysis_payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            analysis_id = uuid.uuid4().hex[:16]
            run_dir = self._run_dir(thread_id, run_id)
            analysis_dir = run_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            path = analysis_dir / f"{analysis_id}.json"
            path.write_text(_json_pretty(analysis_payload), encoding="utf-8")

            record = {
                "analysis_id": analysis_id,
                "local_path": str(path),
                "timestamp": _iso_now(),
                "summary_stats": analysis_payload.get("summary_stats", {}),
            }
            manifest = self._read_manifest(thread_id, run_id)
            manifest.setdefault("analyses", []).append(record)
            self._write_manifest(thread_id, run_id, manifest)
            return record

    def append_audit(self, *, thread_id: str, run_id: str, audit_entry: dict[str, Any]) -> None:
        with self._lock:
            manifest = self._read_manifest(thread_id, run_id)
            manifest.setdefault("audit", []).append(audit_entry)
            self._write_manifest(thread_id, run_id, manifest)

    def cleanup_thread(self, thread_id: str, mode: str = "ephemeral_manifest") -> dict[str, Any]:
        with self._lock:
            thread_dir = self.root / thread_id
            if not thread_dir.exists():
                return {
                    "thread_id": thread_id,
                    "mode": mode,
                    "deleted_files": 0,
                    "bytes_reclaimed": 0,
                    "manifest_retained": 0,
                }

            if mode == "strict_full_purge":
                deleted, reclaimed = self._remove_tree(thread_dir)
                return {
                    "thread_id": thread_id,
                    "mode": mode,
                    "deleted_files": deleted,
                    "bytes_reclaimed": reclaimed,
                    "manifest_retained": 0,
                }

            deleted_files = 0
            reclaimed = 0
            manifest_retained = 0

            for run_dir in sorted(path for path in thread_dir.iterdir() if path.is_dir()):
                manifest_path = run_dir / "manifest.json"
                if manifest_path.exists():
                    manifest_retained += 1

                for directory in (run_dir / "datasets", run_dir / "analysis"):
                    if not directory.exists():
                        continue
                    for child in directory.glob("**/*"):
                        if child.is_file():
                            reclaimed += child.stat().st_size
                            child.unlink(missing_ok=True)
                            deleted_files += 1
                    for nested in sorted(directory.glob("**/*"), reverse=True):
                        if nested.is_dir():
                            nested.rmdir()
                    directory.rmdir()

                if manifest_path.exists():
                    try:
                        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}
                    payload.setdefault("cleanup", []).append(
                        {
                            "mode": mode,
                            "timestamp": _iso_now(),
                            "deleted_files": deleted_files,
                            "bytes_reclaimed": reclaimed,
                        }
                    )
                    manifest_path.write_text(_json_pretty(payload), encoding="utf-8")

            return {
                "thread_id": thread_id,
                "mode": mode,
                "deleted_files": deleted_files,
                "bytes_reclaimed": reclaimed,
                "manifest_retained": manifest_retained,
            }

    def _run_dir(self, thread_id: str, run_id: str) -> Path:
        safe_thread = re.sub(r"[^A-Za-z0-9._-]", "_", thread_id)
        safe_run = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
        return self.root / safe_thread / safe_run

    def _manifest_path(self, thread_id: str, run_id: str) -> Path:
        return self._run_dir(thread_id, run_id) / "manifest.json"

    def _read_manifest(self, thread_id: str, run_id: str) -> dict[str, Any]:
        path = self._manifest_path(thread_id, run_id)
        if not path.exists():
            raise FileNotFoundError(f"Manifest does not exist for run {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, thread_id: str, run_id: str, manifest: dict[str, Any]) -> None:
        path = self._manifest_path(thread_id, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json_pretty(manifest), encoding="utf-8")

    @staticmethod
    def _remove_tree(path: Path) -> tuple[int, int]:
        deleted = 0
        reclaimed = 0
        for child in sorted(path.glob("**/*"), reverse=True):
            if child.is_file():
                reclaimed += child.stat().st_size
                child.unlink(missing_ok=True)
                deleted += 1
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            path.rmdir()
        except OSError:
            pass
        return deleted, reclaimed


class OperatorRuntime:
    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def run_python(self, *, thread_id: str, run_id: str, code: str) -> dict[str, Any]:
        before_records = self.workspace.list_dataset_records(thread_id=thread_id, run_id=run_id)
        before_ids = {record["dataset_id"] for record in before_records}
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
            "WORK_ROOT": str(self.workspace.root),
        }

        with contextlib.redirect_stdout(stdout):
            exec(code, scope, scope)

        after_records = self.workspace.list_dataset_records(thread_id=thread_id, run_id=run_id)
        created = [record for record in after_records if record["dataset_id"] not in before_ids]

        return {
            "ok": True,
            "stdout": stdout.getvalue(),
            "created_datasets": created,
            "run_id": run_id,
        }


class InvestigationPlanner:
    def classify_intent(self, question: str) -> str:
        lowered = question.lower()
        if "top site" in lowered or "site issue" in lowered:
            return "top_site_issues"
        if "customer collection" in lowered and "anomal" in lowered:
            return "customer_collection_anomalies"
        if "market anomal" in lowered or "impact score" in lowered:
            return "market_anomalies_distribution"
        return "generic_investigation"

    def compile_plan(
        self,
        *,
        question: str,
        sales_date: str,
        entities: dict[str, Any],
        knowledge: dict[str, Any],
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or {}
        intent = knowledge.get("intent") or self.classify_intent(question)
        warnings: list[str] = []

        provider = entities.get("providers", [None])[0] if entities.get("providers") else None
        site = entities.get("sites", [None])[0] if entities.get("sites") else None
        customer = entities.get("customers", [None])[0] if entities.get("customers") else None

        provider_filter = f"AND providercode = '{provider}'" if provider else ""
        site_filter = f"AND sitecode = '{site}'" if site else ""
        customer_filter = f"AND customer = '{customer}'" if customer else ""

        extract_steps: list[dict[str, Any]] = []
        analysis_spec: dict[str, Any] = {"type": "summary"}

        if intent == "top_site_issues":
            extract_steps = [
                {
                    "step_id": "top_site_issues",
                    "type": "sql",
                    "datasource": "redshift_core",
                    "query": (
                        "SELECT issue_sources, issue_reasons, providercode, sitecode, COUNT(*) AS issue_count "
                        "FROM prod.monitoring.provider_combined_audit "
                        f"WHERE sales_date = {sales_date} {provider_filter} {site_filter} "
                        "AND issue_sources <> 'request' AND issue_sources <> '' AND issue_reasons <> '' "
                        "GROUP BY issue_sources, issue_reasons, providercode, sitecode "
                        "ORDER BY issue_count DESC"
                    ),
                    "source_metadata": {"intent": intent, "table": "prod.monitoring.provider_combined_audit", "sales_date": sales_date},
                },
                {
                    "step_id": "issue_impact",
                    "type": "sql",
                    "datasource": "redshift_core",
                    "query": (
                        "SELECT providercode, sitecode, COUNT(*) AS total_requests, "
                        "SUM(CASE WHEN (issue_sources <> '' OR filterreason <> '') THEN 1 ELSE 0 END) AS issue_requests, "
                        "ROUND(100.0 * SUM(CASE WHEN (issue_sources <> '' OR filterreason <> '') THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS issue_rate_pct "
                        "FROM prod.monitoring.provider_combined_audit "
                        f"WHERE sales_date = {sales_date} {provider_filter} {site_filter} "
                        "GROUP BY providercode, sitecode ORDER BY issue_rate_pct DESC"
                    ),
                    "source_metadata": {"intent": intent, "table": "prod.monitoring.provider_combined_audit", "sales_date": sales_date},
                },
            ]
            analysis_spec = {"type": "issue_impact", "top_n": int(constraints.get("top_n", 10))}

        elif intent == "customer_collection_anomalies":
            yyyy = sales_date[0:4]
            mm = sales_date[4:6]
            dd = sales_date[6:8]
            key = f"{CUSTOMER_PREFIX}/{yyyy}/{mm}/{dd}/collect_anomaly_{sales_date}.csv"
            extract_steps = [
                {
                    "step_id": "customer_collection_anomalies",
                    "type": "s3_csv",
                    "bucket": ANOMALY_BUCKET,
                    "key": key,
                    "source_metadata": {"intent": intent, "dataset": "customer", "sales_date": sales_date},
                }
            ]
            analysis_spec = {"type": "anomaly_summary", "top_n": int(constraints.get("top_n", 15)), "confirmed_only": True}

        elif intent == "market_anomalies_distribution":
            if not customer:
                warnings.append("No customer partition was detected; running broad market anomaly query.")
            extract_steps = [
                {
                    "step_id": "market_anomalies",
                    "type": "sql",
                    "datasource": "redshift_analytics",
                    "query": (
                        "SELECT observation_date, mkt, seg, top_offenders, cp, dow, impact_score, customer, sales_date "
                        "FROM prod.analytics.market_level_anomalies_v3 "
                        f"WHERE sales_date = {sales_date} {customer_filter} AND any_anomaly = 1 "
                        "ORDER BY impact_score DESC"
                    ),
                    "source_metadata": {"intent": intent, "table": "prod.analytics.market_level_anomalies_v3", "sales_date": sales_date, "customer": customer},
                }
            ]
            analysis_spec = {"type": "distribution", "column": "impact_score", "bucket_count": int(constraints.get("bucket_count", DEFAULT_BUCKET_COUNT))}
        else:
            explicit_tables = re.findall(r"\b[A-Za-z_]+\.[A-Za-z_]+(?:\.[A-Za-z0-9_]+)?\b", question)
            if explicit_tables:
                table_name = explicit_tables[0]
                extract_steps = [
                    {
                        "step_id": "generic_table_preview",
                        "type": "sql",
                        "datasource": _datasource_for_table(table_name),
                        "query": f"SELECT * FROM {table_name} LIMIT {MAX_PREVIEW_ROWS}",
                        "source_metadata": {"intent": "generic_table_preview", "table": table_name},
                    }
                ]
                analysis_spec = {"type": "summary"}
            else:
                warnings.append("No deterministic extraction recipe matched the request.")

        return {
            "intent": intent,
            "sales_date": sales_date,
            "extract_steps": extract_steps,
            "analysis_spec": analysis_spec,
            "warnings": warnings,
            "needs_clarification": len(extract_steps) == 0,
            "entities": {"provider": provider, "site": site, "customer": customer},
        }


def _datasource_for_table(table_name: str) -> str:
    if table_name.startswith("priceeye."):
        return "mysql_priceeye"
    if table_name.startswith("prod.monitoring") or table_name.startswith("local.monitoring"):
        return "redshift_core"
    return "redshift_analytics"


class InvestigationRuntime:
    def __init__(self) -> None:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        KB_ROOT.mkdir(parents=True, exist_ok=True)

        self.registry = DatasourceRegistry()
        self.catalog = LocalCodeCatalog()
        self.resolver = EntityResolver(self.catalog, self.registry)
        self.kb = KnowledgeBase(KB_DB_PATH, self.catalog, self.registry)
        self.guard = SqlGuard()
        self.workspace = WorkspaceManager()
        self.operator = OperatorRuntime(self.workspace)
        self.planner = InvestigationPlanner()

    def ensure_kb_ready(self) -> dict[str, Any]:
        return self.kb.ensure_ready()

    def refresh_knowledge_base(self, force: bool = True) -> dict[str, Any]:
        if force:
            return self.kb.refresh(force=True)
        return self.ensure_kb_ready()

    def resolve_entities(self, input_text: str, sales_date_hint: str | None = None) -> dict[str, Any]:
        return self.resolver.resolve(input_text, sales_date_hint=sales_date_hint)

    def retrieve_knowledge(self, *, intent: str, entities: dict[str, Any], question: str) -> dict[str, Any]:
        self.ensure_kb_ready()
        return self.kb.retrieve(intent=intent, entities=entities, question=question)

    def browse_knowledge_files(self, path_or_glob: str) -> dict[str, Any]:
        patterns: list[str] = []
        raw = path_or_glob.strip()
        if "*" in raw or "?" in raw:
            patterns.append(raw)
        else:
            patterns.append(raw)
            patterns.append(f"{raw}/*")

        allowed_roots = [REPO_ROOT, INVESTIGATION_ROOT]
        files: list[Path] = []
        for pattern in patterns:
            expanded = glob.glob(pattern)
            if not expanded and not Path(pattern).is_absolute():
                expanded = glob.glob(str(REPO_ROOT / pattern))
            for match in expanded:
                path = Path(match).resolve()
                if not path.exists() or not path.is_file():
                    continue
                if any(path == root or root in path.parents for root in allowed_roots):
                    files.append(path)

        out = []
        for path in sorted(set(files))[:200]:
            text = path.read_text(encoding="utf-8", errors="replace")
            out.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "preview": text[:2000],
                }
            )
        return {"matches": out, "count": len(out)}

    def inspect_table_metadata(
        self,
        *,
        table_name: str,
        datasource: str | None = None,
        capture_example_row: bool = True,
    ) -> dict[str, Any]:
        source = datasource or _datasource_for_table(table_name)
        inspected = self.registry.inspect_table_metadata(table_name, source)
        columns = inspected.get("columns", [])
        if not columns:
            raise ValueError(f"No metadata found for table {table_name}")

        partitions = self._infer_partitions(columns)
        sample_query = None
        sample_row: dict[str, Any] | None = None

        if capture_example_row:
            sample_query = f"SELECT * FROM {table_name} LIMIT 1"
            try:
                preview_df = self.registry.execute_sql(source, sample_query)
                if not preview_df.empty:
                    sample_row = _mask_row(preview_df.iloc[0].to_dict())
            except Exception as exc:  # noqa: BLE001
                _log_structured("inspect_table_preview_failed", table_name=table_name, error=str(exc))

        self.kb.upsert_discovered_table(
            table_name=table_name,
            datasource=source,
            columns=columns,
            partitions=partitions,
            example_row=sample_row,
            sample_query=sample_query,
        )

        return {
            "table_name": table_name,
            "datasource": source,
            "columns": columns,
            "partition_candidates": partitions,
            "example_row_masked": sample_row,
            "sample_query": sample_query,
        }

    def extract_sql_to_dataset(
        self,
        *,
        thread_id: str,
        query: str,
        datasource: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        effective_run_id = run_id or self.workspace.start_run(thread_id)
        validated = self.guard.validate(query)

        retries = int((metadata or {}).get("retries", 2))
        error: Exception | None = None
        frame = pd.DataFrame()
        started = time.perf_counter()

        for attempt in range(retries + 1):
            try:
                frame = self.registry.execute_sql(datasource, validated)
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = exc
                if not self._is_transient(exc) or attempt >= retries:
                    break
                time.sleep(0.3 * (attempt + 1))

        if error is not None:
            raise error

        source_metadata = {
            "type": "sql",
            "datasource": datasource,
            "query": validated,
            "query_hash": _stable_hash(validated)[:16],
            **(metadata or {}),
        }
        record = self.workspace.save_dataset(
            thread_id=thread_id,
            run_id=effective_run_id,
            df=frame,
            source_metadata=source_metadata,
            dataset_name=dataset_name,
        )

        self.workspace.append_audit(
            thread_id=thread_id,
            run_id=effective_run_id,
            audit_entry={
                "type": "sql_extract",
                "timestamp": _iso_now(),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "row_count": record["row_count"],
                "datasource": datasource,
                "query_hash": source_metadata["query_hash"],
            },
        )

        return {"run_id": effective_run_id, **record}

    def extract_s3_to_dataset(
        self,
        *,
        thread_id: str,
        bucket: str,
        key_or_prefix: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        effective_run_id = run_id or self.workspace.start_run(thread_id)
        started = time.perf_counter()

        keys_used: list[str] = []
        if key_or_prefix.endswith(".csv"):
            frame = self.registry.fetch_s3_csv(bucket, key_or_prefix)
            keys_used = [key_or_prefix]
        else:
            frame, keys_used = self.registry.fetch_s3_prefix_csv(bucket, key_or_prefix)

        source_metadata = {
            "type": "s3_csv",
            "bucket": bucket,
            "key_or_prefix": key_or_prefix,
            "keys_used": keys_used,
            **(metadata or {}),
        }
        record = self.workspace.save_dataset(
            thread_id=thread_id,
            run_id=effective_run_id,
            df=frame,
            source_metadata=source_metadata,
            dataset_name=dataset_name,
        )

        self.workspace.append_audit(
            thread_id=thread_id,
            run_id=effective_run_id,
            audit_entry={
                "type": "s3_extract",
                "timestamp": _iso_now(),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "row_count": record["row_count"],
                "bucket": bucket,
                "keys_used": keys_used,
            },
        )

        return {"run_id": effective_run_id, **record}

    def operator_run_python(self, *, thread_id: str, code: str, run_id: str | None = None) -> dict[str, Any]:
        effective_run_id = run_id or self.workspace.start_run(thread_id)
        started = time.perf_counter()
        result = self.operator.run_python(thread_id=thread_id, run_id=effective_run_id, code=code)
        self.workspace.append_audit(
            thread_id=thread_id,
            run_id=effective_run_id,
            audit_entry={
                "type": "python_operator",
                "timestamp": _iso_now(),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "created_dataset_count": len(result.get("created_datasets", [])),
            },
        )
        return result

    def run_dataframe_analysis(
        self,
        *,
        thread_id: str,
        run_id: str,
        dataset_ids: Sequence[str],
        analysis_spec: dict[str, Any],
    ) -> dict[str, Any]:
        frames = self.workspace.read_datasets(thread_id=thread_id, run_id=run_id, dataset_ids=dataset_ids)
        analysis_type = analysis_spec.get("type", "summary")

        if analysis_type == "issue_impact":
            payload = self._analyze_issue_impact(frames, analysis_spec)
        elif analysis_type == "anomaly_summary":
            payload = self._analyze_anomaly_summary(frames, analysis_spec)
        elif analysis_type == "distribution":
            payload = self._analyze_distribution(frames, analysis_spec)
        else:
            payload = self._analyze_summary(frames)

        record = self.workspace.record_analysis(thread_id=thread_id, run_id=run_id, analysis_payload=payload)
        self.workspace.append_audit(
            thread_id=thread_id,
            run_id=run_id,
            audit_entry={
                "type": "analysis",
                "analysis_id": record["analysis_id"],
                "analysis_type": analysis_type,
                "timestamp": _iso_now(),
                "dataset_ids": list(dataset_ids),
            },
        )

        return {
            "analysis_id": record["analysis_id"],
            "local_path": record["local_path"],
            "results": payload.get("results", {}),
            "summary_stats": payload.get("summary_stats", {}),
        }

    def cleanup_session_workspace(self, thread_id: str, mode: str = "ephemeral_manifest") -> dict[str, Any]:
        result = self.workspace.cleanup_thread(thread_id, mode)
        _log_structured("workspace_cleanup", **result)
        return {
            **result,
            "manifest_retained": int(result.get("manifest_retained", 0)) > 0,
        }

    def investigate_issue(
        self,
        *,
        thread_id: str,
        question: str,
        sales_date: str | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ensure_kb_ready()

        effective_sales_date = _coerce_sales_date(sales_date or _extract_sales_date_from_text(question))
        entities = self.resolve_entities(question, sales_date_hint=effective_sales_date)
        intent = self.planner.classify_intent(question)
        knowledge = self.retrieve_knowledge(intent=intent, entities=entities, question=question)

        explicit_tables = re.findall(r"\b[A-Za-z_]+\.[A-Za-z_]+(?:\.[A-Za-z0-9_]+)?\b", question)
        discovered_tables: list[dict[str, Any]] = []
        for table_name in explicit_tables:
            if table_name not in knowledge.get("candidate_tables", []):
                knowledge.setdefault("candidate_tables", []).append(table_name)
            try:
                discovered_tables.append(self.inspect_table_metadata(table_name=table_name))
            except Exception as exc:  # noqa: BLE001
                _log_structured("table_discovery_failed", table_name=table_name, error=str(exc))

        plan = self.planner.compile_plan(
            question=question,
            sales_date=effective_sales_date,
            entities=entities,
            knowledge={"intent": intent, **knowledge},
            constraints=constraints,
        )

        run_id = self.workspace.start_run(thread_id)
        dataset_records: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for step in plan.get("extract_steps", []):
            try:
                if step.get("type") == "sql":
                    record = self.extract_sql_to_dataset(
                        thread_id=thread_id,
                        query=str(step.get("query", "")),
                        datasource=str(step.get("datasource", "redshift_analytics")),
                        run_id=run_id,
                        metadata={**step.get("source_metadata", {}), "step_id": step.get("step_id")},
                        dataset_name=str(step.get("step_id", "sql_step")),
                    )
                elif step.get("type") == "s3_csv":
                    key_or_prefix = str(step.get("key") or step.get("prefix") or "")
                    if not key_or_prefix:
                        raise ValueError("Missing key_or_prefix for s3_csv step")
                    record = self.extract_s3_to_dataset(
                        thread_id=thread_id,
                        bucket=str(step.get("bucket", ANOMALY_BUCKET)),
                        key_or_prefix=key_or_prefix,
                        run_id=run_id,
                        metadata={**step.get("source_metadata", {}), "step_id": step.get("step_id")},
                        dataset_name=str(step.get("step_id", "s3_step")),
                    )
                else:
                    raise ValueError(f"Unsupported step type: {step.get('type')}")
                dataset_records.append(record)
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "step_id": step.get("step_id"),
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
                errors.append(payload)
                self.workspace.append_audit(
                    thread_id=thread_id,
                    run_id=run_id,
                    audit_entry={"type": "error", "timestamp": _iso_now(), **payload},
                )

        analysis_result: dict[str, Any] | None = None
        if dataset_records:
            dataset_ids = [item["dataset_id"] for item in dataset_records]
            analysis_result = self.run_dataframe_analysis(
                thread_id=thread_id,
                run_id=run_id,
                dataset_ids=dataset_ids,
                analysis_spec=plan.get("analysis_spec", {}),
            )

        answer = self._synthesize_answer(
            intent=intent,
            sales_date=effective_sales_date,
            entities=entities,
            analysis_result=analysis_result,
            errors=errors,
            warnings=plan.get("warnings", []),
        )

        result = {
            "thread_id": thread_id,
            "run_id": run_id,
            "intent": intent,
            "sales_date": effective_sales_date,
            "entities": entities,
            "knowledge": knowledge,
            "plan": plan,
            "datasets": dataset_records,
            "analysis": analysis_result,
            "discovered_tables": discovered_tables,
            "errors": errors,
            "answer": answer,
            "warnings": plan.get("warnings", []),
            "partial_result": bool(errors),
        }

        _log_structured(
            "investigation_complete",
            thread_id=thread_id,
            run_id=run_id,
            intent=intent,
            dataset_count=len(dataset_records),
            error_count=len(errors),
        )
        return result

    @staticmethod
    def _infer_partitions(columns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        partition_candidates: list[dict[str, Any]] = []
        for column in columns:
            name = str(column.get("column_name", "")).lower()
            if name in {"sales_date", "customer", "providercode", "sitecode"}:
                partition_candidates.append(
                    {
                        "column": name,
                        "role": "recommended",
                        "inferred_type": "categorical" if name != "sales_date" else "date",
                    }
                )
            elif name.endswith("_date"):
                partition_candidates.append(
                    {
                        "column": name,
                        "role": "recommended",
                        "inferred_type": "date",
                    }
                )
        dedup: dict[str, dict[str, Any]] = {}
        for item in partition_candidates:
            dedup[item["column"]] = item
        return list(dedup.values())

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        text = f"{type(exc).__name__}:{exc}".lower()
        return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)

    def _analyze_issue_impact(self, frames: dict[str, pd.DataFrame], analysis_spec: dict[str, Any]) -> dict[str, Any]:
        values = list(frames.values())
        top_n = int(analysis_spec.get("top_n", 10))
        issues_df = values[0] if values else pd.DataFrame()
        impact_df = values[1] if len(values) > 1 else pd.DataFrame()

        issue_rate = None
        if not impact_df.empty and "issue_rate_pct" in impact_df.columns:
            issue_rate = float(pd.to_numeric(impact_df["issue_rate_pct"], errors="coerce").fillna(0).max())

        return {
            "results": {
                "top_issues": issues_df.head(top_n).to_dict(orient="records") if not issues_df.empty else [],
                "impact_rows": impact_df.head(top_n).to_dict(orient="records") if not impact_df.empty else [],
            },
            "summary_stats": {
                "issue_groups": int(len(issues_df)),
                "impact_rows": int(len(impact_df)),
                "max_issue_rate_pct": issue_rate,
            },
        }

    def _analyze_anomaly_summary(self, frames: dict[str, pd.DataFrame], analysis_spec: dict[str, Any]) -> dict[str, Any]:
        top_n = int(analysis_spec.get("top_n", 15))
        if not frames:
            return {"results": {"anomalies": []}, "summary_stats": {"confirmed_anomalies": 0}}

        df = list(frames.values())[0].copy()
        if analysis_spec.get("confirmed_only", True) and {"anomaly_t1", "anomaly_t2"}.issubset(df.columns):
            t1 = pd.to_numeric(df["anomaly_t1"], errors="coerce").fillna(0).astype(int)
            t2 = pd.to_numeric(df["anomaly_t2"], errors="coerce").fillna(0).astype(int)
            df = df[(t1 == 1) & (t2 == 1)].copy()

        def top_counts(column: str) -> list[dict[str, Any]]:
            if column not in df.columns or df.empty:
                return []
            out = (
                df[column]
                .astype(str)
                .value_counts(dropna=False)
                .head(top_n)
                .rename_axis(column)
                .reset_index(name="count")
            )
            return out.to_dict(orient="records")

        return {
            "results": {
                "anomalies": df.head(top_n).to_dict(orient="records"),
                "top_providers": top_counts("providercode"),
                "top_customers": top_counts("customer"),
                "top_sites": top_counts("sitecode"),
            },
            "summary_stats": {
                "confirmed_anomalies": int(len(df)),
            },
        }

    def _analyze_distribution(self, frames: dict[str, pd.DataFrame], analysis_spec: dict[str, Any]) -> dict[str, Any]:
        column = str(analysis_spec.get("column", "impact_score"))
        bucket_count = max(2, int(analysis_spec.get("bucket_count", DEFAULT_BUCKET_COUNT)))

        if not frames:
            return {
                "results": {"distribution": []},
                "summary_stats": {"count": 0, "mean": None, "p50": None, "p90": None, "max": None},
            }

        df = list(frames.values())[0]
        if column not in df.columns:
            return {
                "results": {"distribution": []},
                "summary_stats": {"count": 0, "mean": None, "p50": None, "p90": None, "max": None},
            }

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            return {
                "results": {"distribution": []},
                "summary_stats": {"count": 0, "mean": None, "p50": None, "p90": None, "max": None},
            }

        counts, bins = np.histogram(values, bins=bucket_count)
        distribution = [
            {
                "bucket_start": float(bins[idx]),
                "bucket_end": float(bins[idx + 1]),
                "count": int(counts[idx]),
            }
            for idx in range(len(counts))
        ]

        return {
            "results": {"distribution": distribution},
            "summary_stats": {
                "count": int(values.count()),
                "mean": float(values.mean()),
                "p50": float(values.quantile(0.5)),
                "p90": float(values.quantile(0.9)),
                "max": float(values.max()),
            },
        }

    @staticmethod
    def _analyze_summary(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        if not frames:
            return {"results": {"datasets": []}, "summary_stats": {"dataset_count": 0}}
        results = [
            {"dataset_id": dataset_id, "rows": int(len(df)), "columns": list(df.columns)}
            for dataset_id, df in frames.items()
        ]
        return {"results": {"datasets": results}, "summary_stats": {"dataset_count": len(frames)}}

    @staticmethod
    def _synthesize_answer(
        *,
        intent: str,
        sales_date: str,
        entities: dict[str, Any],
        analysis_result: dict[str, Any] | None,
        errors: Sequence[dict[str, Any]],
        warnings: Sequence[str],
    ) -> str:
        if analysis_result is None:
            base = f"No executable data plan for intent '{intent}' on {sales_date}."
            if warnings:
                base += " Warnings: " + " ".join(warnings)
            return base

        summary = analysis_result.get("summary_stats", {})
        if intent == "top_site_issues":
            provider = entities.get("providers", [])
            site = entities.get("sites", [])
            provider_label = provider[0] if provider else "all providers"
            site_label = site[0] if site else "all sites"
            return (
                f"Top site issue analysis completed for {provider_label}/{site_label} on {sales_date}. "
                f"Issue groups: {summary.get('issue_groups', 0)}; "
                f"max issue rate: {summary.get('max_issue_rate_pct')}%."
            )

        if intent == "customer_collection_anomalies":
            return (
                f"Customer collection anomaly summary for {sales_date}: "
                f"{summary.get('confirmed_anomalies', 0)} confirmed anomalies."
            )

        if intent == "market_anomalies_distribution":
            return (
                f"Market anomaly impact distribution for {sales_date}: "
                f"count={summary.get('count')}, mean={summary.get('mean')}, "
                f"p50={summary.get('p50')}, p90={summary.get('p90')}, max={summary.get('max')}."
            )

        message = f"Investigation finished for {intent} on {sales_date}."
        if errors:
            message += f" Encountered {len(errors)} step error(s)."
        if warnings:
            message += " Warnings: " + " ".join(warnings)
        return message


_RUNTIME: InvestigationRuntime | None = None
_RUNTIME_LOCK = threading.RLock()


def get_runtime() -> InvestigationRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = InvestigationRuntime()
        return _RUNTIME


def cleanup_thread_workspace(thread_id: str, mode: str = "ephemeral_manifest") -> dict[str, Any]:
    runtime = get_runtime()
    return runtime.cleanup_session_workspace(thread_id=thread_id, mode=mode)


def is_next_gen_enabled() -> bool:
    return _bool_env("NEXT_GEN_INVESTIGATION", True)
