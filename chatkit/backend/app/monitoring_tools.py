"""Custom function tools for the internal monitoring agent."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

import pandas as pd
from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent
from threevictors.dao import redshift_connector

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)
log.propagate = False


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

    def describe_table(self, table_name: str) -> dict:
        """
        Get metadata and key information about a table.

        Args:
            table_name: Full table name (e.g., 'price_anomalies.anomaly_table').
                       For cross-database queries (e.g., 'prod.monitoring.table'),
                       note that information_schema only shows tables in the current database.
                       Use read_table_head() or query_table() for cross-database access.

        Returns:
            dict with table metadata
        """
        parts = table_name.split(".")
        if len(parts) == 3:
            schema = parts[1]
            table = parts[2]
        elif len(parts) == 2:
            schema = parts[0]
            table = parts[1]
        else:
            return {
                "error": (
                    f"Invalid table name format: {table_name}. "
                    "Use 'schema.table' or 'database.schema.table'"
                )
            }

        query = f"""
        SELECT
            table_schema,
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        LIMIT 1;
        """

        with self.get_connection().cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()

            if not records:
                return {"error": f"Table {table_name} not found"}

            df = pd.DataFrame(records, columns=colnames)
            return df.to_dict(orient="records")[0]

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

    def get_top_site_issues(self, target_date: str | None = None) -> pd.DataFrame:
        """
        Get top site issues for today and compare with last week and last month.

        Args:
            target_date: Date in YYYYMMDD format (default: today)

        Returns:
            DataFrame with issue_sources, issue_reasons, and counts for today, last week, last month
        """
        if target_date is None:
            target_date = datetime.date.today().strftime("%Y%m%d")

        target = datetime.datetime.strptime(str(target_date), "%Y%m%d").date()
        last_week = (target - datetime.timedelta(days=7)).strftime("%Y%m%d")
        last_month = (target - datetime.timedelta(days=30)).strftime("%Y%m%d")

        query = f"""
        WITH today_issues AS (
            SELECT
                issue_sources,
                issue_reasons,
                sitecode,
                COUNT(*) as today_count
            FROM prod.monitoring.provider_combined_audit
            WHERE sales_date = {target_date}
              AND issue_sources != 'request'
            GROUP BY issue_sources, issue_reasons, sitecode
        ),
        last_week_issues AS (
            SELECT
                issue_sources,
                issue_reasons,
                sitecode,
                COUNT(*) as last_week_count
            FROM prod.monitoring.provider_combined_audit
            WHERE sales_date = {last_week}
              AND issue_sources != 'request'
            GROUP BY issue_sources, issue_reasons, sitecode
        ),
        last_month_issues AS (
            SELECT
                issue_sources,
                issue_reasons,
                sitecode,
                COUNT(*) as last_month_count
            FROM prod.monitoring.provider_combined_audit
            WHERE sales_date = {last_month}
              AND issue_sources != 'request'
            GROUP BY issue_sources, issue_reasons, sitecode
        )
        SELECT
            COALESCE(t.sitecode, lw.sitecode, lm.sitecode) as sitecode,
            COALESCE(t.issue_sources, lw.issue_sources, lm.issue_sources) as issue_sources,
            COALESCE(t.issue_reasons, lw.issue_reasons, lm.issue_reasons) as issue_reasons,
            COALESCE(t.today_count, 0) as today_count,
            COALESCE(lw.last_week_count, 0) as last_week_count,
            COALESCE(lm.last_month_count, 0) as last_month_count,
            COALESCE(t.today_count, 0) - COALESCE(lw.last_week_count, 0) as week_over_week_change,
            COALESCE(t.today_count, 0) - COALESCE(lm.last_month_count, 0) as month_over_month_change
        FROM today_issues t
        FULL OUTER JOIN last_week_issues lw
            ON t.sitecode = lw.sitecode
            AND t.issue_sources = lw.issue_sources
            AND t.issue_reasons = lw.issue_reasons
        FULL OUTER JOIN last_month_issues lm
            ON COALESCE(t.sitecode, lw.sitecode) = lm.sitecode
            AND COALESCE(t.issue_sources, lw.issue_sources) = lm.issue_sources
            AND COALESCE(t.issue_reasons, lw.issue_reasons) = lm.issue_reasons
        WHERE COALESCE(t.today_count, lw.last_week_count, lm.last_month_count) > 0
        ORDER BY today_count DESC
        LIMIT 50;
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



_reader: AnalyticsReader | None = None


def _get_reader() -> AnalyticsReader:
    global _reader
    if _reader is None:
        _reader = AnalyticsReader()
    return _reader


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.to_dict(orient="records")


def _short_json(value: Any, limit: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=True, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated)"


async def _stream_progress(
    ctx: RunContextWrapper[AgentContext],
    icon: str,
    text: str,
) -> None:
    await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))


def monitoring_instructions() -> str:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"You are a database exploration and monitoring agent. Today is {current_date}.\n"
        "Available tables: use describe_table() to verify tables as needed.\n\n"
        "GENERAL DATABASE EXPLORATION:\n"
        "1. Start with describe_table() to understand key columns and table metadata.\n"
        "2. Use read_table_head(limit=...) for quick previews.\n"
        "3. When invoking query_table(), write SELECT/WITH statements only, keep LIMIT clauses, and include partition filters.\n\n"
        "PROVIDER MONITORING TOOLS:\n"
        "4. Use get_top_site_issues(target_date) to identify top site issues for a specific date and compare with last week/month.\n"
        "There are two types of issues: site issues and request issues:\n"
        "site issues are defined as issue_source is tagged as site which is failure to collect data from the site itself\n"
        "request issues are defined as issue_source is tagged as request which refer to the request itself is invalid.\n"
        "   - Accepts date in YYYYMMDD format (e.g., '20251109')\n"
        "   - Returns issue_sources, issue_reasons, and counts with trend analysis\n"
        "5. Use analyze_issue_scope(providercode, sitecode, target_date, lookback_days) to analyze issue dimensions.\n"
        "   - Breaks down issues by POS, triptype, LOS, cabin, O&D, depart dates, day of week, observation hour\n"
        "   - Example: analyze_issue_scope('QL2', 'QF', '20251109', 7) for QL2/QF issues over last 7 days\n\n"
        "Never modify data and cite which tool produced each insight."
    )


@function_tool
async def describe_table(ctx: RunContextWrapper[AgentContext], table_name: str) -> dict:
    """Return table metadata from information_schema."""
    await _stream_progress(ctx, "search", f"describe_table: {_short_json({'table_name': table_name})}")
    return _get_reader().describe_table(table_name)


@function_tool
async def read_table_head(
    ctx: RunContextWrapper[AgentContext],
    table_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return a preview of table rows."""
    await _stream_progress(
        ctx,
        "search",
        f"read_table_head: {_short_json({'table_name': table_name, 'limit': limit})}",
    )
    return _df_records(_get_reader().read_table_head(table_name, limit=limit))


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
        f"query_table: {_short_json({'query': query, 'limit': limit})}",
    )
    return _df_records(_get_reader().query_table(query, limit=limit))


@function_tool
async def get_top_site_issues(
    ctx: RunContextWrapper[AgentContext],
    target_date: str | None = None,
) -> list[dict[str, Any]]:
    """Return top site issues and trend deltas."""
    await _stream_progress(
        ctx,
        "search",
        f"get_top_site_issues: {_short_json({'target_date': target_date})}",
    )
    return _df_records(_get_reader().get_top_site_issues(target_date))


@function_tool
async def analyze_issue_scope(
    ctx: RunContextWrapper[AgentContext],
    providercode: str | None = None,
    sitecode: str | None = None,
    target_date: str | None = None,
    lookback_days: int = 7,
) -> list[dict[str, Any]]:
    """Return issue scope breakdowns by multiple dimensions."""
    await _stream_progress(
        ctx,
        "search",
        f"analyze_issue_scope: {_short_json({'providercode': providercode, 'sitecode': sitecode, 'target_date': target_date, 'lookback_days': lookback_days})}",
    )
    return _df_records(
        _get_reader().analyze_issue_scope(
            providercode=providercode,
            sitecode=sitecode,
            target_date=target_date,
            lookback_days=lookback_days,
        )
    )


def monitoring_tools() -> list[Any]:
    return [
        describe_table,
        read_table_head,
        query_table,
        get_top_site_issues,
        analyze_issue_scope,
    ]
