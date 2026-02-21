"""Custom function tools for the internal monitoring agent."""

from __future__ import annotations

import datetime
import logging
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

    def _parse_sales_date(value: str | datetime.date | datetime.datetime | None) -> datetime.date:
        if value is None:
            return datetime.date.today()
        if isinstance(value, datetime.datetime):
            return value.date()
        if isinstance(value, datetime.date):
            return value
        raw = value.strip()
        if len(raw) == 8 and raw.isdigit():
            return datetime.datetime.strptime(raw, "%Y%m%d").date()
        return datetime.datetime.strptime(raw, "%Y-%m-%d").date()

    def _parse_filter_set(raw: str | None) -> set[str]:
        if not raw:
            return set()
        return {item.strip().upper() for item in raw.split(",") if item.strip()}

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

    def _add_entity_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
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

    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        provider_set = _parse_filter_set(providercode)
        site_set = _parse_filter_set(sitecode)
        customer_set = _parse_filter_set(customer)
        metric_set = _parse_filter_set(metric_name)
        model_set = _parse_filter_set(model_type)

        if provider_set and "providercode" in out.columns:
            out = out[out["providercode"].astype(str).str.upper().isin(provider_set)]
        if site_set and "sitecode" in out.columns:
            out = out[out["sitecode"].astype(str).str.upper().isin(site_set)]
        if customer_set and "customer" in out.columns:
            out = out[out["customer"].astype(str).str.upper().isin(customer_set)]

        metric_col = "metric_name" if "metric_name" in out.columns else ("metric" if "metric" in out.columns else None)
        if metric_set and metric_col:
            out = out[out[metric_col].astype(str).str.upper().isin(metric_set)]
        if model_set and "model_type" in out.columns:
            out = out[out["model_type"].astype(str).str.upper().isin(model_set)]
        return out

    def _confirmed_only(df: pd.DataFrame) -> pd.DataFrame:
        if "anomaly_t1" not in df.columns or "anomaly_t2" not in df.columns:
            return df.iloc[0:0].copy()
        t1 = pd.to_numeric(df["anomaly_t1"], errors="coerce").fillna(0).astype(int)
        t2 = pd.to_numeric(df["anomaly_t2"], errors="coerce").fillna(0).astype(int)
        return df[(t1 == 1) & (t2 == 1)].copy()

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

    def _dataset_payload(name: str, source_keys: list[str], raw_df: pd.DataFrame) -> dict[str, Any]:
        filtered = _apply_filters(_add_entity_columns(raw_df))
        confirmed = _confirmed_only(filtered)
        metric_col = "metric_name" if "metric_name" in confirmed.columns else ("metric" if "metric" in confirmed.columns else None)

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
        }
        payload["top_metrics"] = _value_counts(confirmed, metric_col) if metric_col else []
        return payload

    target_date = _parse_sales_date(sales_date)
    y = target_date.strftime("%Y")
    m = target_date.strftime("%m")
    d = target_date.strftime("%d")
    yyyymmdd = target_date.strftime("%Y%m%d")
    clamped_limit = max(1, min(limit, 2000))

    s3_client = s3_util.S3Util()
    customer_key = f"{CUSTOMER_PREFIX}/{y}/{m}/{d}/collect_anomaly_{yyyymmdd}.csv"
    provider_key = f"{PROVIDER_PREFIX}/{y}/{m}/{d}/provider_anomaly_{yyyymmdd}.csv"
    laterequest_prefix = f"{LATEREQUEST_PREFIX}/{y}/{m}/{d}/"

    customer_keys = [customer_key] if _object_exists(s3_client, ANOMALY_BUCKET, customer_key) else []
    provider_keys = [provider_key] if _object_exists(s3_client, ANOMALY_BUCKET, provider_key) else []
    laterequest_keys = _list_csv_keys(s3_client, ANOMALY_BUCKET, laterequest_prefix)

    def _load_many(keys: list[str]) -> pd.DataFrame:
        if not keys:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for key in keys:
            df = _read_csv(s3_client, ANOMALY_BUCKET, key)
            if df.empty:
                continue
            df = df.copy()
            df["source_key"] = key
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    requested_filters = {
        "providercode": providercode,
        "sitecode": sitecode,
        "customer": customer,
        "metric_name": metric_name,
        "model_type": model_type,
    }

    payload: dict[str, Any] = {
        "sales_date": yyyymmdd,
        "bucket": ANOMALY_BUCKET,
        "requested_filters": requested_filters,
        "limit": clamped_limit,
        "available_partitions": {
            "customer": bool(customer_keys),
            "provider": bool(provider_keys),
            "laterequests": bool(laterequest_keys),
        },
    }

    payload["results"] = {
        "customer": _dataset_payload("customer", customer_keys, _load_many(customer_keys)),
        "provider": _dataset_payload("provider", provider_keys, _load_many(provider_keys)),
        "laterequests": _dataset_payload("laterequests", laterequest_keys, _load_many(laterequest_keys)),
    }
    payload["missing_partitions"] = [
        name for name, exists in payload["available_partitions"].items() if not exists
    ]
    payload["has_any_partition"] = any(payload["available_partitions"].values())
    return payload
