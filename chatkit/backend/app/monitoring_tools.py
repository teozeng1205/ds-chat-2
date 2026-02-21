"""Custom function tools for the internal monitoring agent."""

from __future__ import annotations

import datetime
import logging
import re
from io import StringIO
from typing import Any

import pandas as pd
from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent
from threevictors.dao import redshift_connector
from threevictors.s3_util import s3_util

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)
log.propagate = False

ANOMALY_BUCKET = "s3-atp-3victors-3vdev-use1-collection-anomalies"
CUSTOMER_PREFIX = "collection-customer/v1"
PROVIDER_PREFIX = "collection-provider/v1"
LATEREQUEST_PREFIX = "collection-laterequests/v1"
DEFAULT_ANOMALY_LIMIT = 200
MAX_MONITORING_CACHE_ENTRIES = 6

_DATASET_ALIAS_MAP = {
    "CUSTOMER": "customer",
    "CUSTOMERCOLLECTION": "customer",
    "CUSTOMERCOLLECTIONS": "customer",
    "COLLECTIONCUSTOMER": "customer",
    "PROVIDER": "provider",
    "PROVIDERCOLLECTION": "provider",
    "PROVIDERCOLLECTIONS": "provider",
    "COLLECTIONPROVIDER": "provider",
    "LATEREQUEST": "laterequests",
    "LATEREQUESTS": "laterequests",
    "LATE_REQUEST": "laterequests",
    "LATE_REQUESTS": "laterequests",
    "DELIVERY": "laterequests",
    "DELIVERYANOMALY": "laterequests",
    "DELIVERYANOMALIES": "laterequests",
    "COLLECTIONLATEREQUEST": "laterequests",
    "COLLECTIONLATEREQUESTS": "laterequests",
    "ALL": "all",
}

_COLLECTION_SCOPE_TOKENS = {
    "CUSTOMERCOLLECTION",
    "CUSTOMERCOLLECTIONS",
    "COLLECTIONCUSTOMER",
    "PROVIDERCOLLECTION",
    "PROVIDERCOLLECTIONS",
    "COLLECTIONPROVIDER",
    "LATEREQUEST",
    "LATEREQUESTS",
    "COLLECTIONLATEREQUEST",
    "COLLECTIONLATEREQUESTS",
    "DELIVERYANOMALY",
    "DELIVERYANOMALIES",
}

_MONITORING_CACHE: dict[str, dict[str, Any]] = {}
_MONITORING_CACHE_ORDER: list[str] = []


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _parse_sales_date_value(value: str | datetime.date | datetime.datetime | None) -> datetime.date:
    if value is None:
        return datetime.date.today()
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value

    raw = str(value).strip()
    lowered = raw.lower()
    if lowered in {"today", "now"}:
        return datetime.date.today()
    if lowered == "yesterday":
        return datetime.date.today() - datetime.timedelta(days=1)
    if lowered == "tomorrow":
        return datetime.date.today() + datetime.timedelta(days=1)
    if len(raw) == 8 and raw.isdigit():
        return datetime.datetime.strptime(raw, "%Y%m%d").date()
    return datetime.datetime.strptime(raw, "%Y-%m-%d").date()


def _parse_filter_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


def _parse_filter_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_dataset_scope(dataset: str | None) -> tuple[str, list[str]]:
    normalized = _normalize_token(dataset or "customer")
    canonical = _DATASET_ALIAS_MAP.get(normalized)
    if not canonical:
        valid = sorted({"customer", "provider", "laterequests", "all"})
        raise ValueError(f"Unsupported dataset '{dataset}'. Use one of: {', '.join(valid)}.")
    if canonical == "all":
        return canonical, ["customer", "provider", "laterequests"]
    return canonical, [canonical]


def _object_exists(client: s3_util.S3Util, bucket: str, key: str) -> bool:
    try:
        client.s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _list_csv_keys(client: s3_util.S3Util, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.s3_client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = item.get("Key")
            if key and key.endswith(".csv"):
                keys.append(key)
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    return sorted(keys)


def _read_csv(client: s3_util.S3Util, bucket: str, key: str) -> pd.DataFrame:
    raw = client.get_object(bucket, key)
    for sep in (None, ",", "\t"):
        try:
            if sep is None:
                df = pd.read_csv(StringIO(raw), sep=None, engine="python")
            else:
                df = pd.read_csv(StringIO(raw), sep=sep)
            if not df.empty or len(df.columns) > 1:
                break
        except Exception:
            continue
    else:
        df = pd.DataFrame()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _load_many_csvs(client: s3_util.S3Util, keys: list[str]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for key in keys:
        df = _read_csv(client, ANOMALY_BUCKET, key)
        if df.empty:
            continue
        tagged = df.copy()
        tagged["source_key"] = key
        frames.append(tagged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _cache_monitoring_date(date_key: str, payload: dict[str, Any]) -> None:
    _MONITORING_CACHE[date_key] = payload
    if date_key in _MONITORING_CACHE_ORDER:
        _MONITORING_CACHE_ORDER.remove(date_key)
    _MONITORING_CACHE_ORDER.append(date_key)
    while len(_MONITORING_CACHE_ORDER) > MAX_MONITORING_CACHE_ENTRIES:
        stale = _MONITORING_CACHE_ORDER.pop(0)
        _MONITORING_CACHE.pop(stale, None)


def _load_monitoring_partition_data(
    sales_date: str | datetime.date | datetime.datetime | None = None,
) -> dict[str, Any]:
    target_date = _parse_sales_date_value(sales_date)
    y = target_date.strftime("%Y")
    m = target_date.strftime("%m")
    d = target_date.strftime("%d")
    yyyymmdd = target_date.strftime("%Y%m%d")

    cached = _MONITORING_CACHE.get(yyyymmdd)
    if cached is not None:
        return cached

    client = s3_util.S3Util()
    customer_key = f"{CUSTOMER_PREFIX}/{y}/{m}/{d}/collect_anomaly_{yyyymmdd}.csv"
    provider_key = f"{PROVIDER_PREFIX}/{y}/{m}/{d}/provider_anomaly_{yyyymmdd}.csv"
    laterequest_prefix = f"{LATEREQUEST_PREFIX}/{y}/{m}/{d}/"

    customer_keys = [customer_key] if _object_exists(client, ANOMALY_BUCKET, customer_key) else []
    provider_keys = [provider_key] if _object_exists(client, ANOMALY_BUCKET, provider_key) else []
    laterequest_keys = _list_csv_keys(client, ANOMALY_BUCKET, laterequest_prefix)

    payload = {
        "sales_date": yyyymmdd,
        "available_partitions": {
            "customer": bool(customer_keys),
            "provider": bool(provider_keys),
            "laterequests": bool(laterequest_keys),
        },
        "source_keys": {
            "customer": customer_keys,
            "provider": provider_keys,
            "laterequests": laterequest_keys,
        },
        "frames": {
            "customer": _load_many_csvs(client, customer_keys),
            "provider": _load_many_csvs(client, provider_keys),
            "laterequests": _load_many_csvs(client, laterequest_keys),
        },
    }
    _cache_monitoring_date(yyyymmdd, payload)
    return payload


def _add_entity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "prov_site" in out.columns:
        parts = out["prov_site"].astype(str).str.split("|", n=1, expand=True)
        if "providercode" not in out.columns:
            out["providercode"] = parts[0].str.strip()
        if parts.shape[1] > 1 and "sitecode" not in out.columns:
            out["sitecode"] = parts[1].str.strip()
    if "cust_site" in out.columns:
        parts = out["cust_site"].astype(str).str.split("|", n=1, expand=True)
        if "customer" not in out.columns:
            out["customer"] = parts[0].str.strip()
        if parts.shape[1] > 1 and "sitecode" not in out.columns:
            out["sitecode"] = parts[1].str.strip()
    return out


def _confirmed_only_df(df: pd.DataFrame) -> pd.DataFrame:
    if "anomaly_t1" not in df.columns or "anomaly_t2" not in df.columns:
        return df.iloc[0:0].copy()
    t1 = pd.to_numeric(df["anomaly_t1"], errors="coerce").fillna(0).astype(int)
    t2 = pd.to_numeric(df["anomaly_t2"], errors="coerce").fillna(0).astype(int)
    return df[(t1 == 1) & (t2 == 1)].copy()


def _apply_named_filters(
    df: pd.DataFrame,
    *,
    providercode: str | None = None,
    sitecode: str | None = None,
    customer: str | None = None,
    metric_name: str | None = None,
    model_type: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    notes: dict[str, Any] = {}

    provider_set = _parse_filter_set(providercode)
    site_set = _parse_filter_set(sitecode)
    customer_set = _parse_filter_set(customer)
    model_set = _parse_filter_set(model_type)
    metric_tokens = _parse_filter_list(metric_name)
    ignored_scope_tokens = [
        token for token in metric_tokens if _normalize_token(token) in _COLLECTION_SCOPE_TOKENS
    ]
    metric_norm_set = {
        _normalize_token(token)
        for token in metric_tokens
        if _normalize_token(token) and _normalize_token(token) not in _COLLECTION_SCOPE_TOKENS
    }

    if provider_set and "providercode" in out.columns:
        out = out[out["providercode"].astype(str).str.upper().isin(provider_set)]
    if site_set and "sitecode" in out.columns:
        out = out[out["sitecode"].astype(str).str.upper().isin(site_set)]
    if customer_set and "customer" in out.columns:
        out = out[out["customer"].astype(str).str.upper().isin(customer_set)]
    metric_col = (
        "metric_name"
        if "metric_name" in out.columns
        else ("metric" if "metric" in out.columns else None)
    )
    if metric_norm_set and metric_col:
        metric_series = out[metric_col].astype(str).map(_normalize_token)
        out = out[metric_series.isin(metric_norm_set)]
    if model_set and "model_type" in out.columns:
        out = out[out["model_type"].astype(str).str.upper().isin(model_set)]

    if ignored_scope_tokens:
        notes["ignored_metric_scope_tokens"] = ignored_scope_tokens
    return out, notes


def _parse_condition_tokens(filters: str | None) -> tuple[list[tuple[str, str, str]], list[str]]:
    if not filters:
        return [], []
    parsed: list[tuple[str, str, str]] = []
    invalid: list[str] = []
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|=|>|<|~)\s*(.+?)\s*$")
    for raw_condition in [segment.strip() for segment in filters.split(",") if segment.strip()]:
        match = pattern.match(raw_condition)
        if not match:
            invalid.append(raw_condition)
            continue
        col, op, value = match.groups()
        cleaned = value.strip().strip("\"'")
        parsed.append((col, op, cleaned))
    return parsed, invalid


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _apply_expression_filters(
    df: pd.DataFrame,
    filters: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    parsed, invalid = _parse_condition_tokens(filters)
    ignored_columns: list[str] = []
    numeric_parse_errors: list[str] = []
    applied: list[str] = []

    for column, op, raw_value in parsed:
        if column not in out.columns:
            ignored_columns.append(column)
            continue
        series = out[column]

        if op in {">", ">=", "<", "<="}:
            target = _to_float(raw_value)
            if target is None:
                numeric_parse_errors.append(f"{column}{op}{raw_value}")
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            if op == ">":
                mask = numeric > target
            elif op == ">=":
                mask = numeric >= target
            elif op == "<":
                mask = numeric < target
            else:
                mask = numeric <= target
            out = out[mask.fillna(False)]
            applied.append(f"{column}{op}{raw_value}")
            continue

        if op == "~":
            mask = series.astype(str).str.contains(raw_value, case=False, na=False)
            out = out[mask]
            applied.append(f"{column}{op}{raw_value}")
            continue

        target_float = _to_float(raw_value)
        if target_float is not None and pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if op == "=":
                mask = numeric == target_float
            else:
                mask = numeric != target_float
        else:
            normalized_series = series.astype(str).str.upper()
            normalized_target = raw_value.upper()
            if op == "=":
                mask = normalized_series == normalized_target
            else:
                mask = normalized_series != normalized_target
        out = out[mask.fillna(False)]
        applied.append(f"{column}{op}{raw_value}")

    notes = {
        "applied": applied,
        "invalid_syntax": invalid,
        "ignored_columns": ignored_columns,
        "numeric_parse_errors": numeric_parse_errors,
    }
    return out, notes


class AnalyticsReader(redshift_connector.RedshiftConnector):
    """
    Analytics database reader using Redshift connector.

    Provides connection management and query execution for analytics.* tables.
    """

    def __init__(self):
        log.info("Initializing AnalyticsReader")
        super().__init__()
        log.info("AnalyticsReader initialized successfully")

    def get_properties_filename(self):
        """Properties file for Redshift connection configuration."""
        return "database-analytics-redshift-serverless-reader.properties"

    def read_table_head(self, table_name: str, limit: int = 50) -> pd.DataFrame:
        """
        Get data preview (first N rows) from a table.

        Args:
            table_name: Full table name (e.g., 'prod.monitoring.provider_combined_audit')
            limit: Number of rows to return (default: 50)

        Returns:
            DataFrame with first N rows
        """
        query = f"""
        SELECT *
        FROM {table_name}
        LIMIT {limit};
        """

        with self.get_connection().cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=colnames)
            return df

    def query_table(self, query: str, limit: int = 1000) -> pd.DataFrame:
        """
        Execute a SELECT/WITH query on the database.

        Args:
            query: SQL SELECT/WITH statement
            limit: Maximum rows to return (default: 1000, safety limit)

        Returns:
            DataFrame with query results
        """
        normalized = query.strip().upper()
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            raise ValueError("Only SELECT or WITH queries are allowed")

        if "LIMIT" not in normalized:
            query = query.rstrip(";") + f" LIMIT {limit};"

        log.info("Executing query: %s...", query[:100])

        with self.get_connection().cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=colnames)

            log.info("Query returned %s rows", len(df))
            return df

    def get_top_site_issues(
        self,
        target_date: str | None = None,
        providercode: str | None = None,
        sitecode: str | None = None,
    ) -> pd.DataFrame:
        """
        Get top site issues for a specific date.

        Args:
            target_date: Date in YYYYMMDD format (default: today)
            providercode: Provider code(s) - single code (e.g., 'QL2') or comma-separated (e.g., 'QL2,Atlas')
            sitecode: Site code(s) - single code (e.g., 'QF') or comma-separated (e.g., 'QF,DY')

        Returns:
            DataFrame with issue_sources, issue_reasons, providercode, sitecode, and counts
        """
        if target_date is None:
            target_date = datetime.date.today().strftime("%Y%m%d")

        where_clauses = [
            f"sales_date = {target_date}",
            "issue_sources != 'request'",
            "issue_sources != ''",
            "issue_reasons != ''",
        ]

        if providercode:
            providers = [p.strip() for p in providercode.split(",")]
            if len(providers) == 1:
                where_clauses.append(f"providercode = '{providers[0]}'")
            else:
                provider_list = "', '".join(providers)
                where_clauses.append(f"providercode IN ('{provider_list}')")

        if sitecode:
            sites = [s.strip() for s in sitecode.split(",")]
            if len(sites) == 1:
                where_clauses.append(f"sitecode = '{sites[0]}'")
            else:
                site_list = "', '".join(sites)
                where_clauses.append(f"sitecode IN ('{site_list}')")

        where_clause = " AND ".join(where_clauses)

        query = f"""
        SELECT
            issue_sources,
            issue_reasons,
            providercode,
            sitecode,
            COUNT(*) as today_count
        FROM prod.monitoring.provider_combined_audit
        WHERE {where_clause}
        GROUP BY issue_sources, issue_reasons, sitecode, providercode
        ORDER BY today_count DESC;
        """

        log.info("Getting top site issues for date: %s", target_date)
        with self.get_connection().cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=colnames)
            log.info("Found %s issue combinations", len(df))
            return df

    def analyze_issue_scope(
        self,
        providercode: str | None = None,
        sitecode: str | None = None,
        target_date: str | None = None,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """
        Analyze the scope of issues for providers and/or sites.

        Args:
            providercode: Provider code(s) - single code (e.g., 'QL2') or comma-separated (e.g., 'QL2,Atlas')
            sitecode: Site code(s) - single code (e.g., 'QF') or comma-separated (e.g., 'QF,DY')
            target_date: Date in YYYYMMDD format (default: today)
            lookback_days: Number of days to analyze (default: 7)

        Returns:
            DataFrame with issue breakdown by multiple dimensions
        """
        if target_date is None:
            target_date = datetime.date.today().strftime("%Y%m%d")

        target = datetime.datetime.strptime(str(target_date), "%Y%m%d").date()
        start_date = (target - datetime.timedelta(days=lookback_days)).strftime("%Y%m%d")

        where_clauses = []

        if providercode:
            providers = [p.strip() for p in providercode.split(",")]
            if len(providers) == 1:
                where_clauses.append(f"providercode = '{providers[0]}'")
            else:
                provider_list = "', '".join(providers)
                where_clauses.append(f"providercode IN ('{provider_list}')")

        if sitecode:
            sites = [s.strip() for s in sitecode.split(",")]
            if len(sites) == 1:
                where_clauses.append(f"sitecode = '{sites[0]}'")
            else:
                site_list = "', '".join(sites)
                where_clauses.append(f"sitecode IN ('{site_list}')")

        where_clauses.append(f"sales_date BETWEEN {start_date} AND {target_date}")
        where_clauses.append("(issue_sources != '' OR filterreason != '')")
        where_clauses.append("(issue_sources != 'request')")

        where_clause = " AND ".join(where_clauses)

        query = f"""
        SELECT
            providercode,
            sitecode,
            pos,
            triptype,
            los,
            cabin,
            originairportcode,
            destinationairportcode,
            origincitycode,
            destinationcitycode,
            origincountrycode,
            destinationcountrycode,
            departdate,
            EXTRACT(DOW FROM TO_DATE(CAST(departdate AS VARCHAR), 'YYYYMMDD')) as depart_dow,
            DATE_PART('hour', observationtimestamp) as observation_hour,
            issue_sources,
            issue_reasons,
            response_statuses,
            filterreason,
            COUNT(*) as issue_count,
            COUNT(DISTINCT sales_date) as days_with_issues,
            MIN(sales_date) as first_seen_date,
            MAX(sales_date) as last_seen_date
        FROM prod.monitoring.provider_combined_audit
        WHERE {where_clause}
        GROUP BY
            providercode, sitecode, pos, triptype, los, cabin,
            originairportcode, destinationairportcode,
            origincitycode, destinationcitycode,
            origincountrycode, destinationcountrycode,
            departdate, depart_dow, observation_hour,
            issue_sources, issue_reasons, response_statuses, filterreason
        ORDER BY issue_count DESC
        LIMIT 100;
        """

        log.info(
            "Analyzing issue scope for provider=%s, site=%s, date=%s",
            providercode,
            sitecode,
            target_date,
        )
        with self.get_connection().cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=colnames)
            log.info("Found %s dimensional breakdowns", len(df))
            return df


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.to_dict(orient="records")


def _preview_sql(query: str, limit: int = 120) -> str:
    compact = " ".join(query.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


async def _stream_progress(
    ctx: RunContextWrapper[AgentContext],
    icon: str,
    text: str,
) -> None:
    await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))


@function_tool
async def read_table_head(
    ctx: RunContextWrapper[AgentContext],
    table_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return a preview of table rows."""
    await _stream_progress(ctx, "search", f"Previewing `{table_name}` (limit {limit}).")
    reader: AnalyticsReader | None = None
    try:
        reader = AnalyticsReader()
        df = reader.read_table_head(table_name, limit=limit)
        records = _df_records(df)
        await _stream_progress(
            ctx,
            "check-circle",
            f"Table preview complete for `{table_name}`: {len(records)} rows.",
        )
        return records
    except Exception as exc:
        await _stream_progress(ctx, "bug", f"Table preview failed: {type(exc).__name__}.")
        raise
    finally:
        if reader is not None:
            reader.close()


@function_tool
async def query_table(
    ctx: RunContextWrapper[AgentContext],
    query: str,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Run a SELECT/WITH query and return rows."""
    await _stream_progress(
        ctx,
        "search",
        f"Running monitoring SQL (limit {limit}): {_preview_sql(query)}",
    )
    reader: AnalyticsReader | None = None
    try:
        reader = AnalyticsReader()
        await _stream_progress(ctx, "clock", "Executing monitoring SQL query.")
        df = reader.query_table(query, limit=limit)
        records = _df_records(df)
        await _stream_progress(ctx, "check-circle", f"Monitoring SQL complete: {len(records)} rows.")
        return records
    except Exception as exc:
        await _stream_progress(ctx, "bug", f"Monitoring SQL failed: {type(exc).__name__}.")
        raise
    finally:
        if reader is not None:
            reader.close()


@function_tool
async def get_top_site_issues(
    ctx: RunContextWrapper[AgentContext],
    target_date: str | None = None,
    providercode: str | None = None,
    sitecode: str | None = None,
) -> list[dict[str, Any]]:
    """Return top site issues and trend deltas."""
    effective_date = target_date or datetime.date.today().strftime("%Y%m%d")
    provider_label = providercode or "all providers"
    site_label = sitecode or "all sites"
    await _stream_progress(
        ctx,
        "search",
        f"Finding top site issues for {effective_date} ({provider_label}, {site_label}).",
    )
    reader: AnalyticsReader | None = None
    try:
        reader = AnalyticsReader()
        await _stream_progress(ctx, "clock", "Computing issue-source rankings.")
        df = reader.get_top_site_issues(
            target_date=target_date,
            providercode=providercode,
            sitecode=sitecode,
        )
        records = _df_records(df)
        await _stream_progress(
            ctx,
            "check-circle",
            f"Top site issue analysis complete: {len(records)} grouped rows.",
        )
        return records
    except Exception as exc:
        await _stream_progress(ctx, "bug", f"Top site issue analysis failed: {type(exc).__name__}.")
        raise
    finally:
        if reader is not None:
            reader.close()


@function_tool
async def analyze_issue_scope(
    ctx: RunContextWrapper[AgentContext],
    providercode: str | None = None,
    sitecode: str | None = None,
    target_date: str | None = None,
    lookback_days: int = 7,
) -> list[dict[str, Any]]:
    """Return issue scope breakdowns by multiple dimensions."""
    effective_date = target_date or datetime.date.today().strftime("%Y%m%d")
    provider_label = providercode or "all providers"
    site_label = sitecode or "all sites"
    await _stream_progress(
        ctx,
        "search",
        f"Analyzing issue scope for {provider_label}, {site_label} through {effective_date}.",
    )
    reader: AnalyticsReader | None = None
    try:
        reader = AnalyticsReader()
        await _stream_progress(ctx, "clock", f"Running {lookback_days}-day dimensional scope analysis.")
        df = reader.analyze_issue_scope(
            providercode=providercode,
            sitecode=sitecode,
            target_date=target_date,
            lookback_days=lookback_days,
        )
        records = _df_records(df)
        await _stream_progress(
            ctx,
            "check-circle",
            f"Issue scope analysis complete: {len(records)} dimensional rows.",
        )
        return records
    except Exception as exc:
        await _stream_progress(ctx, "bug", f"Issue scope analysis failed: {type(exc).__name__}.")
        raise
    finally:
        if reader is not None:
            reader.close()


@function_tool
async def explore_monitoring_anomaly_data(
    ctx: RunContextWrapper[AgentContext],
    sales_date: str | None = None,
    dataset: str = "customer",
    confirmed_only: bool = True,
    filters: str | None = None,
    columns: str | None = None,
    sort_by: str | None = None,
    descending: bool = True,
    limit: int = 100,
    include_head: bool = True,
    head_rows: int = 10,
    group_by: str | None = None,
    group_limit: int = 25,
) -> dict[str, Any]:
    """
    Load S3 anomaly CSVs into a dataframe-style view, inspect head/schema, and apply interactive filters.

    Filter expression syntax:
    - Comma-separated conditions, e.g. "customer=TS,sitecode=F8,response_itin_ratio<0.9,metric_name~ratio"
    - Supported operators: =, !=, >, >=, <, <=, ~ (contains, case-insensitive)
    """
    target_date = _parse_sales_date_value(sales_date)
    canonical_dataset, datasets = _resolve_dataset_scope(dataset)
    clamped_limit = max(1, min(limit, 2000))
    clamped_head_rows = max(0, min(head_rows, 100))
    clamped_group_limit = max(1, min(group_limit, 200))

    await _stream_progress(
        ctx,
        "search",
        f"Loading anomaly CSV dataframe for {target_date.strftime('%Y%m%d')} ({canonical_dataset}).",
    )
    loaded = _load_monitoring_partition_data(target_date)

    pieces: list[pd.DataFrame] = []
    selected_source_keys: list[str] = []
    for name in datasets:
        frame = loaded["frames"][name]
        if frame.empty:
            continue
        tagged = frame.copy()
        tagged["dataset"] = name
        pieces.append(tagged)
        selected_source_keys.extend(loaded["source_keys"][name])

    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    enriched = _add_entity_columns(combined)
    rows_loaded = int(len(enriched))

    working = enriched.copy()
    if confirmed_only:
        working = _confirmed_only_df(working)
    rows_after_confirmed = int(len(working))

    filtered, filter_notes = _apply_expression_filters(working, filters)
    rows_after_filters = int(len(filtered))

    selected_columns = [col.strip() for col in (columns or "").split(",") if col.strip()]
    missing_columns: list[str] = []
    if selected_columns:
        keep_columns = [col for col in selected_columns if col in filtered.columns]
        missing_columns = [col for col in selected_columns if col not in filtered.columns]
        if keep_columns:
            filtered = filtered[keep_columns]
            working = working[keep_columns]
            enriched = enriched[keep_columns]

    if sort_by:
        if sort_by in filtered.columns:
            filtered = filtered.sort_values(by=sort_by, ascending=not descending, kind="stable")
        else:
            missing_columns.append(sort_by)

    output_rows = filtered.head(clamped_limit)
    truncated = rows_after_filters > clamped_limit

    group_summary: list[dict[str, Any]] = []
    group_columns = [col.strip() for col in (group_by or "").split(",") if col.strip()]
    if group_columns:
        existing = [col for col in group_columns if col in filtered.columns]
        if existing:
            grouped = (
                filtered.groupby(existing, dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False, kind="stable")
                .head(clamped_group_limit)
            )
            group_summary = _df_records(grouped)
        else:
            missing_columns.extend(group_columns)

    def _value_counts(df: pd.DataFrame, column: str, size: int = 15) -> list[dict[str, Any]]:
        if column not in df.columns or df.empty:
            return []
        summary = (
            df[column]
            .astype(str)
            .value_counts(dropna=False)
            .head(size)
            .rename_axis(column)
            .reset_index(name="count")
        )
        return _df_records(summary)

    top_metrics: list[dict[str, Any]] = []
    metric_col = (
        "metric_name"
        if "metric_name" in filtered.columns
        else ("metric" if "metric" in filtered.columns else None)
    )
    if metric_col:
        top_metrics = _value_counts(filtered, metric_col)

    await _stream_progress(
        ctx,
        "check-circle",
        f"Anomaly dataframe exploration complete: {rows_after_filters} rows after filters.",
    )

    return {
        "sales_date": loaded["sales_date"],
        "dataset": canonical_dataset,
        "datasets_included": datasets,
        "available_partitions": loaded["available_partitions"],
        "missing_partitions": [
            name for name, exists in loaded["available_partitions"].items() if not exists
        ],
        "source_keys": selected_source_keys,
        "confirmed_only": confirmed_only,
        "schema_columns": list(enriched.columns),
        "rows_loaded": rows_loaded,
        "rows_after_confirmed": rows_after_confirmed,
        "rows_after_filters": rows_after_filters,
        "truncated": truncated,
        "head": _df_records(working.head(clamped_head_rows)) if include_head and clamped_head_rows else [],
        "rows": _df_records(output_rows),
        "filters": filters,
        "filter_notes": filter_notes,
        "missing_columns": sorted(set(missing_columns)),
        "group_by": group_columns,
        "group_summary": group_summary,
        "top_providers": _value_counts(filtered, "providercode"),
        "top_customers": _value_counts(filtered, "customer"),
        "top_sites": _value_counts(filtered, "sitecode"),
        "top_metrics": top_metrics,
    }


def _query_df(reader: AnalyticsReader, query: str) -> pd.DataFrame:
    with reader.get_connection().cursor() as cursor:
        cursor.execute(query)
        colnames = [desc[0] for desc in cursor.description]
        records = cursor.fetchall()
        return pd.DataFrame(records, columns=colnames)


def _get_internal_monitoring_anomalies_impl(
    sales_date: str | datetime.date | datetime.datetime | None = None,
    providercode: str | None = None,
    sitecode: str | None = None,
    customer: str | None = None,
    metric_name: str | None = None,
    model_type: str | None = None,
    limit: int = DEFAULT_ANOMALY_LIMIT,
) -> dict[str, Any]:
    """Read anomaly CSVs from S3 for a date and return filtered confirmed anomalies."""
    loaded = _load_monitoring_partition_data(sales_date)
    clamped_limit = max(1, min(limit, 2000))

    def _value_counts(df: pd.DataFrame, column: str, size: int = 15) -> list[dict[str, Any]]:
        if column not in df.columns or df.empty:
            return []
        summary = (
            df[column]
            .astype(str)
            .value_counts(dropna=False)
            .head(size)
            .rename_axis(column)
            .reset_index(name="count")
        )
        return _df_records(summary)

    def _dataset_payload(name: str) -> dict[str, Any]:
        raw_df = loaded["frames"][name]
        source_keys = loaded["source_keys"][name]
        enriched = _add_entity_columns(raw_df)
        filtered, filter_notes = _apply_named_filters(
            enriched,
            providercode=providercode,
            sitecode=sitecode,
            customer=customer,
            metric_name=metric_name,
            model_type=model_type,
        )
        confirmed = _confirmed_only_df(filtered)
        metric_col = (
            "metric_name"
            if "metric_name" in confirmed.columns
            else ("metric" if "metric" in confirmed.columns else None)
        )

        payload: dict[str, Any] = {
            "dataset": name,
            "available": bool(source_keys),
            "source_keys": source_keys,
            "rows_total": int(len(raw_df)),
            "rows_after_filters": int(len(filtered)),
            "confirmed_anomalies": int(len(confirmed)),
            "columns": list(raw_df.columns),
            "truncated": int(len(confirmed)) > clamped_limit,
            "anomalies": _df_records(confirmed.head(clamped_limit)),
            "top_providers": _value_counts(confirmed, "providercode"),
            "top_customers": _value_counts(confirmed, "customer"),
            "top_sites": _value_counts(confirmed, "sitecode"),
            "filter_notes": filter_notes,
        }
        payload["top_metrics"] = _value_counts(confirmed, metric_col) if metric_col else []
        return payload

    requested_filters = {
        "providercode": providercode,
        "sitecode": sitecode,
        "customer": customer,
        "metric_name": metric_name,
        "model_type": model_type,
    }

    payload: dict[str, Any] = {
        "sales_date": loaded["sales_date"],
        "bucket": ANOMALY_BUCKET,
        "requested_filters": requested_filters,
        "limit": clamped_limit,
        "available_partitions": loaded["available_partitions"],
    }
    payload["results"] = {
        "customer": _dataset_payload("customer"),
        "provider": _dataset_payload("provider"),
        "laterequests": _dataset_payload("laterequests"),
    }
    payload["missing_partitions"] = [
        name for name, exists in payload["available_partitions"].items() if not exists
    ]
    payload["has_any_partition"] = any(payload["available_partitions"].values())
    return payload
