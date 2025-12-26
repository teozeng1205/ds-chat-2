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
        Get top site issues for a specific date.

        Args:
            target_date: Date in YYYYMMDD format (default: today)

        Returns:
            DataFrame with issue_sources, issue_reasons, providercode, sitecode, and counts
        """
        if target_date is None:
            target_date = datetime.date.today().strftime("%Y%m%d")

        query = f"""
        SELECT
            issue_sources,
            issue_reasons,
            providercode,
            sitecode,
            COUNT(*) as today_count
        FROM prod.monitoring.provider_combined_audit
        WHERE sales_date = {target_date}
          AND issue_sources != 'request'
          AND issue_sources != ''
          AND issue_reasons != ''
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
        "Only use tools for answers. Prioritize any tool that can directly answer the user.\n"
        "If no tool directly answers, start with read_table_head(), then follow with query_table() using your own SQL.\n"
        "Primary table: prod.monitoring.provider_combined_audit (partitioned by sales_date as bigint like 20251205).\n"
        "\n"
        "GENERAL DATABASE EXPLORATION:\n"
        "1. Use read_table_head(limit=...) for quick previews.\n"
        "2. When invoking query_table(), write SELECT/WITH statements only, keep LIMIT clauses, and include partition filters.\n\n"
        "PROVIDER MONITORING TOOLS:\n"
        "3. Use get_top_site_issues(target_date) to identify top site issues for a specific date.\n"
        "There are two types of issues: site issues and request issues:\n"
        "site issues are defined as issue_source is tagged as site which is failure to collect data from the site itself\n"
        "request issues are defined as issue_source is tagged as request which refer to the request itself is invalid.\n"
        "   - Accepts date in YYYYMMDD format (e.g., '20251109')\n"
        "   - Returns issue_sources, issue_reasons, providercode, sitecode, and counts\n"
        "4. Use analyze_issue_scope(providercode, sitecode, target_date, lookback_days) to analyze issue dimensions.\n"
        "   - Breaks down issues by POS, triptype, LOS, cabin, O&D, depart dates, day of week, observation hour\n"
        "   - Example: analyze_issue_scope('QL2', 'QF', '20251109', 7) for QL2/QF issues over last 7 days\n\n"
        "Never modify data and cite which tool produced each insight."
    )


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
    reader = AnalyticsReader()
    try:
        return _df_records(reader.read_table_head(table_name, limit=limit))
    finally:
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
        f"query_table: {_short_json({'query': query, 'limit': limit})}",
    )
    reader = AnalyticsReader()
    try:
        return _df_records(reader.query_table(query, limit=limit))
    finally:
        reader.close()


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
    reader = AnalyticsReader()
    try:
        return _df_records(reader.get_top_site_issues(target_date))
    finally:
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
    await _stream_progress(
        ctx,
        "search",
        f"analyze_issue_scope: {_short_json({'providercode': providercode, 'sitecode': sitecode, 'target_date': target_date, 'lookback_days': lookback_days})}",
    )
    reader = AnalyticsReader()
    try:
        return _df_records(
            reader.analyze_issue_scope(
                providercode=providercode,
                sitecode=sitecode,
                target_date=target_date,
                lookback_days=lookback_days,
            )
        )
    finally:
        reader.close()


def monitoring_tools() -> list[Any]:
    return [
        read_table_head,
        query_table,
        get_top_site_issues,
        analyze_issue_scope,
    ]
