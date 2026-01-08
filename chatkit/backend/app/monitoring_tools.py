"""Custom function tools for the internal monitoring agent."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

import numpy as np
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
        "3. Use get_top_site_issues(target_date, providercode, sitecode) get a simple quick summary for today (not anomalies).\n"
        "   - Accepts date in YYYYMMDD format (e.g., '20251109')\n"
        "   - Optional filters: providercode and/or sitecode (single or comma-separated)\n"
            "   - Returns issue_sources, issue_reasons, providercode, sitecode, and counts\n"
        "4. Use analyze_issue_scope(providercode, sitecode, target_date, lookback_days) to analyze issue dimensions.\n"
        "   - Breaks down issues by POS, triptype, LOS, cabin, O&D, depart dates, day of week, observation hour\n"
        "   - Example: analyze_issue_scope('QL2', 'QF', '20251109', 7) for QL2/QF issues over last 7 days\n\n"
        "5. Use get_provider_anomaly(sales_date) for anomalies\n"
        "   - sales_date is required (YYYY-MM-DD or YYYYMMDD), defaults to today if not provided.\n"
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
    providercode: str | None = None,
    sitecode: str | None = None,
) -> list[dict[str, Any]]:
    """Return top site issues and trend deltas."""
    await _stream_progress(
        ctx,
        "search",
        f"get_top_site_issues: {_short_json({'target_date': target_date, 'providercode': providercode, 'sitecode': sitecode})}",
    )
    reader = AnalyticsReader()
    try:
        return _df_records(
            reader.get_top_site_issues(
                target_date=target_date,
                providercode=providercode,
                sitecode=sitecode,
            )
        )
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


def _query_df(reader: AnalyticsReader, query: str) -> pd.DataFrame:
    with reader.get_connection().cursor() as cursor:
        cursor.execute(query)
        colnames = [desc[0] for desc in cursor.description]
        records = cursor.fetchall()
        return pd.DataFrame(records, columns=colnames)


def _get_provider_anomaly_impl(
    sales_date: str | datetime.date | datetime.datetime,
    lookback_days: int = 22,
    eval_date_delta: int = 0,
) -> list[str]:
    """
    Returns Slack-ready message strings (header + per-provider/per-metric tables),
    for the given sales_date (evaluation date).
    """

    def _parse_sales_date(d):
        if isinstance(d, datetime.datetime):
            return d.date()
        if isinstance(d, datetime.date):
            return d
        if isinstance(d, str):
            s = d.strip()
            if len(s) == 8 and s.isdigit():
                return datetime.datetime.strptime(s, "%Y%m%d").date()
            return datetime.datetime.strptime(s, "%Y-%m-%d").date()
        raise ValueError("sales_date must be YYYY-MM-DD, YYYYMMDD, date, or datetime")

    def get_alert_table(df):
        return "```\n" + df.to_string(index=False) + "\n```"

    def anomaly_detect(df_in, df_iqr, l):
        q1 = df_iqr["metric_val" + l].astype(float).quantile(0.25)
        q3 = df_iqr["metric_val" + l].astype(float).quantile(0.75)
        iqr = q3 - q1
        lower_limit = max(q1 - 1.5 * iqr, 0)
        upper_limit = q3 + 1.5 * iqr

        df_out = df_in.copy()
        if (upper_limit * 0.98) <= lower_limit <= (upper_limit * 1.02):
            lower_limit *= 0.95
            upper_limit *= 1.05

        df_out["anomaly" + l] = np.where(
            (df_out["metric_val" + l] < lower_limit) | (df_out["metric_val" + l] > upper_limit), 1, 0
        )
        df_out["lower" + l] = round(lower_limit, 2)
        df_out["upper" + l] = round(upper_limit, 2)
        return df_out

    def outlier_replace(df_in, t1, t2, eval_date_int):
        df_out = df_in.copy()
        mean = df_out[df_out["obs_date"] != eval_date_int]["metric_val" + t1].mean()

        reorder = df_out.pop("metric_val" + t1)
        df_out.insert(5, "metric_val" + t1, reorder)
        reorder_2 = df_out.pop("metric")
        df_out.insert(4, "metric", reorder_2)

        df_out["metric_val" + t2] = np.where(
            (df_out["anomaly" + t1] == 1) & (df_out["obs_date"] != eval_date_int),
            mean,
            df_out["metric_val" + t1],
        ).astype("float64").round(2)
        return df_out

    def model_main(df_in, mode, eval_dow, eval_date_int):
        if mode == "dow":
            df_in = df_in[df_in["dow"].str.contains(eval_dow)].copy()

        df_iqr = df_in.drop(df_in[df_in["obs_date"] >= eval_date_int].index)
        full_v1 = anomaly_detect(df_in, df_iqr, "_t1")
        full_v1 = outlier_replace(full_v1, "_t1", "_t2", eval_date_int)
        df_out = anomaly_detect(full_v1, full_v1, "_t2")

        anomalies = df_out[(df_out["anomaly_t2"] == 1) & (df_out["obs_date"] == eval_date_int)].copy()
        anomalies["model_type"] = mode
        return df_out, anomalies

    def avg_req(df):
        sum_tr_site = (
            df.groupby(["obs_date", "prov_site"])["total_request"]
            .sum()
            .reset_index()
            .rename(columns={"prov_site": "agg_val"})
        )
        avg_tr_site = sum_tr_site.groupby(["agg_val"])["total_request"].mean().reset_index().round(2)

        sum_tr_prov = (
            df.groupby(["obs_date", "providercode"])["total_request"]
            .sum()
            .reset_index()
            .rename(columns={"providercode": "agg_val"})
        )
        avg_tr_prov = sum_tr_prov.groupby(["agg_val"])["total_request"].mean().reset_index().round(2)

        avg_tr_merge = pd.concat([avg_tr_prov, avg_tr_site], ignore_index=True).rename(
            columns={"total_request": "avg_total_req"}
        )

        sum_tr_total = df.groupby(["obs_date"])["total_request"].sum().reset_index()
        avg_tr_total = sum_tr_total["total_request"].mean().round(2)
        return avg_tr_merge, avg_tr_total

    def get_rank_cust(reader, agg_field, x_days, start_int, end_int):
        if agg_field == "providercode":
            b1 = "TRIM(providercode) AS providercode"
            b2 = "providercode"
            group_by = "TRIM(providercode)"
        elif agg_field == "prov_site":
            b1 = "TRIM(providercode) || '|' || TRIM(sitecode) AS prov_site"
            b2 = "prov_site"
            group_by = "TRIM(providercode), TRIM(sitecode)"
        else:
            raise ValueError("agg_field must be 'providercode' or 'prov_site'")

        query = f"""
            WITH customer_counts AS (
                SELECT
                    {b1},
                    customers,
                    COUNT(*) AS cust_count
                FROM monitoring.provider_combined_audit
                WHERE sales_date >= {start_int}
                  AND sales_date <= {end_int}
                  AND customers NOT IN ('GJ','Sanity','CH','JV')
                GROUP BY
                    {group_by},
                    customers
            ),
            ranked_customers AS (
                SELECT
                    {b2},
                    customers,
                    cust_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY {b2}
                        ORDER BY cust_count DESC
                    ) AS rn
                FROM customer_counts
            )
            SELECT
                {b2},
                LISTAGG(DISTINCT customers, ',') WITHIN GROUP (ORDER BY cust_count DESC) AS impacted_cust
            FROM ranked_customers
            WHERE rn <= 5
            GROUP BY {b2};
        """
        df_cust = _query_df(reader, query).rename(columns={agg_field: "agg_val"})
        return df_cust

    def get_cust_pct(reader, start_int, end_int):
        query = f"""
            WITH c AS (
                SELECT customers, COUNT(DISTINCT id) AS customer_count
                FROM monitoring.provider_combined_audit
                WHERE sales_date >= {start_int}
                  AND sales_date <= {end_int}
                  AND customers NOT IN ('GJ','Sanity','CH','JV')
                GROUP BY customers
            ),
            total AS (
                SELECT COUNT(DISTINCT id) AS total_count
                FROM monitoring.provider_combined_audit
                WHERE sales_date >= {start_int}
                  AND sales_date <= {end_int}
            )
            SELECT
                c.customers,
                ROUND(100.0 * c.customer_count / total.total_count, 2) AS cust_pct
            FROM c, total
            ORDER BY cust_pct DESC;
        """
        return _query_df(reader, query)

    def fill_empty_requests(df, eval_date_int, days_back=8):
        df = df.copy()
        df["obs_date"] = df["obs_date"].astype(int)

        d = pd.to_datetime(str(eval_date_int))
        prev_days = [int((d - pd.Timedelta(days=i)).strftime("%Y%m%d")) for i in range(1, days_back + 1)]

        g = df.groupby(["aggregation_val", "obs_date"], as_index=False)["total_request"].sum()

        have_eval = set(g.loc[(g.obs_date == eval_date_int) & (g.total_request > 0), "aggregation_val"])
        candidates = []

        for s in g["aggregation_val"].unique():
            if s in have_eval:
                continue
            ok_prev = True
            for p in prev_days:
                if g.loc[(g.aggregation_val == s) & (g.obs_date == p), "total_request"].sum() <= 0:
                    ok_prev = False
                    break
            if ok_prev:
                candidates.append(s)

        if not candidates:
            return df

        eval_dow = pd.to_datetime(str(eval_date_int)).strftime("%A")[:3]
        num_cols = [c for c in df.columns if c not in ["obs_date", "dow", "aggregation_val", "aggregation"]]

        new_rows = []
        for s in candidates:
            row = {
                "obs_date": eval_date_int,
                "dow": eval_dow,
                "aggregation_val": s,
            }
            if "aggregation" in df.columns:
                row["aggregation"] = df.loc[df["aggregation_val"] == s, "aggregation"].iloc[0]

            for c in num_cols:
                row[c] = 0

            new_rows.append(row)

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return df

    def get_df(reader, start_int, end_int, utc_hour_cap):
        query = f"""
            SELECT cast(observationtimestamp as date) as obs_date
                , EXTRACT(HOUR FROM observationtimestamp) as obs_hour
                , EXTRACT(HOUR FROM response_timestamp) as response_hour
                , trim(providercode) as providercode
                , trim(providercode) + '|' + trim(sitecode) as prov_site
                , count(distinct id) as total_request
                , Count(DISTINCT CASE when response_status LIKE '%success%' THEN id END) as total_success
                , Count(DISTINCT CASE when issue_source LIKE '%site%' AND response_status NOT LIKE '%success%' THEN id END) as total_site_issue
                , SUM(CASE WHEN response_status IS NOT NULL THEN response_itinerarycount ELSE 0 END) as total_itinerary
                , total_itinerary/(total_success + 1) as response_itin_ratio
            FROM monitoring.combined_audit
            WHERE sales_date >= {start_int}
              AND sales_date <= {end_int}
            GROUP BY 1,2,3,4,5;
        """
        df = _query_df(reader, query)

        df = df[df["obs_hour"] < utc_hour_cap]
        df["total_success"] = np.where(df["response_hour"] >= utc_hour_cap, 0, df["total_success"])
        df["total_site_issue"] = np.where(
            df["response_hour"] >= utc_hour_cap, df["total_request"], df["total_site_issue"]
        )
        df["response_itin_ratio"] = np.where(df["response_hour"] >= utc_hour_cap, 0, df["response_itin_ratio"])

        df["dow"] = df["obs_date"].apply(lambda x: pd.to_datetime(x).strftime("%A")[:3])
        df["obs_date"] = df["obs_date"].apply(lambda x: pd.to_datetime(x).strftime("%Y%m%d")).astype(int)
        return df

    eval_date = _parse_sales_date(sales_date) - datetime.timedelta(days=eval_date_delta)
    eval_date_int = int(eval_date.strftime("%Y%m%d"))
    eval_dow = eval_date.strftime("%A")[:3]

    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    current_utc_hour = datetime.datetime.now(datetime.timezone.utc).hour
    utc_hour_cap = current_utc_hour if eval_date == today_utc else 24

    start_date = eval_date - datetime.timedelta(days=lookback_days)
    start_int = int(start_date.strftime("%Y%m%d"))
    end_int = int(eval_date.strftime("%Y%m%d"))

    reader = AnalyticsReader()
    try:
        df1 = get_df(reader, start_int, end_int, utc_hour_cap)
        avg_req_df, avg_tr_total = avg_req(df1)
        metric_variables = ["total_request", "success_pct", "site_issue_pct", "response_itin_ratio"]
        aggregation_level = ["providercode", "prov_site"]

        exclusions = set()

        df_total = []
        df_dow = []
        df_anomalies = []

        for a in aggregation_level:
            for m in metric_variables:
                df = df1

                df_agg = (
                    df[
                        [
                            "obs_date",
                            "dow",
                            a,
                            "total_request",
                            "total_success",
                            "total_site_issue",
                            "response_itin_ratio",
                        ]
                    ]
                    .groupby(["obs_date", "dow", a])
                    .agg("sum")
                    .reset_index()
                )

                df_agg["success_pct"] = round(100 * (df_agg["total_success"] / df_agg["total_request"]), 2)
                df_agg["site_issue_pct"] = round(100 * (df_agg["total_site_issue"] / df_agg["total_request"]), 2)
                df_agg = df_agg.drop(["total_success", "total_site_issue"], axis=1)

                df_agg["aggregation"] = a
                df_agg["aggregation_val"] = df_agg[a]
                agg_list = df_agg["aggregation_val"].unique()
                df_agg = df_agg.drop(a, axis=1)

                df_agg = fill_empty_requests(df_agg, eval_date_int=eval_date_int)

                extra_m = metric_variables.copy()
                extra_m.remove(m)

                df_model = df_agg.copy()
                df_model = df_model.rename(columns={m: "metric_val_t1"})
                df_model = df_model.drop(extra_m, axis=1)
                df_model["metric"] = m

                for val in agg_list:
                    if val in exclusions:
                        continue

                    df_filter = df_model[df_model["aggregation_val"] == val]
                    model_out, model_anoms = model_main(df_filter, "total", eval_dow, eval_date_int)
                    df_total.append(model_out)
                    df_anomalies.append(model_anoms)

                    model_out_dow, model_anoms_dow = model_main(df_filter, "dow", eval_dow, eval_date_int)
                    df_dow.append(model_out_dow)
                    df_anomalies.append(model_anoms_dow)

        if not df_anomalies:
            return [f"SAME_DAY PROVIDER | Anomalies for: sales_date={eval_date} | (no anomalies)"]

        final_anomalies = pd.concat(df_anomalies, ignore_index=True)

        final_anomalies = final_anomalies[
            ~((final_anomalies["metric"] == "site_issue_pct") & (final_anomalies["metric_val_t1"] < 1))
        ]

        final_anomalies["direction"] = np.where(
            (final_anomalies["metric_val_t1"] < final_anomalies["lower_t2"]),
            "DOWN",
            np.where((final_anomalies["metric_val_t1"] > final_anomalies["upper_t2"]), "UP", None),
        )

        cols_round = ["metric_val_t1", "lower_t1", "upper_t1", "metric_val_t2", "lower_t2", "upper_t2"]
        for c in cols_round:
            if c in final_anomalies.columns:
                final_anomalies[c] = final_anomalies[c].round(0).astype(int)

        for c in ["lower_t1", "upper_t1", "lower_t2", "upper_t2"]:
            final_anomalies[c] = final_anomalies[c].map(lambda x: f"{x:,}")

        final_anomalies["iqr"] = (
            final_anomalies["lower_t2"].astype(str) + " - " + final_anomalies["upper_t2"].astype(str)
        )

        anomalies_dow = final_anomalies[final_anomalies["model_type"] == "dow"].copy()
        anomalies_dow = anomalies_dow[["aggregation", "aggregation_val", "metric", "metric_val_t1", "direction", "iqr"]]
        anomalies_dow.columns = ["aggregation", "agg_val", "metric", "metric_val", "direction", "dow_iqr"]

        anomalies_total = final_anomalies[final_anomalies["model_type"] == "total"].copy()
        anomalies_total = anomalies_total[
            ["aggregation", "aggregation_val", "metric", "metric_val_t1", "direction", "iqr"]
        ]
        anomalies_total.columns = ["aggregation", "agg_val", "metric", "metric_val", "direction", "total_iqr"]

        anomalies_alert = pd.merge(
            anomalies_total,
            anomalies_dow,
            on=("aggregation", "agg_val", "metric", "metric_val", "direction"),
        )

        anomalies_alert["prov"] = anomalies_alert["agg_val"].str.split("|").str[0]
        prov_alert = anomalies_alert["prov"].unique()

        anomalies_merge = pd.merge(anomalies_alert, avg_req_df, on=["agg_val"], how="left")

        df_cust_prov = get_rank_cust(reader, "providercode", lookback_days, start_int, end_int)
        df_cust_ps = get_rank_cust(reader, "prov_site", lookback_days, start_int, end_int)
        final_cust_list = pd.concat([df_cust_prov, df_cust_ps], ignore_index=True)
        anomalies_merge2 = pd.merge(anomalies_merge, final_cust_list, on=["agg_val"], how="left")

        anomalies_merge2 = anomalies_merge2[
            ~((anomalies_merge2["impacted_cust"] == "CHNL") | (anomalies_merge2["impacted_cust"].isnull()))
        ]

        anomalies_merge2 = anomalies_merge2[
            ~((anomalies_merge2["prov"].isin(prov_alert)) & (anomalies_merge2["metric"] == "total_request"))
        ]

        cust_pcts = get_cust_pct(reader, start_int, end_int)
        total_score_sum = float(cust_pcts["cust_pct"].sum())

        anomalies_merge2["cust_score"] = np.nan
        for idx, row in anomalies_merge2.iterrows():
            score_sum = 0
            cust_list = str(row["impacted_cust"]).split(",")
            for c in cust_list:
                match = cust_pcts[cust_pcts["customers"] == c]
                if not match.empty:
                    score_sum += float(match.iloc[0]["cust_pct"])
            anomalies_merge2.at[idx, "cust_score"] = (
                round((score_sum / total_score_sum) * 100, 2) if total_score_sum else 0.0
            )
        anomalies_merge2["cust_score"] = pd.to_numeric(anomalies_merge2["cust_score"], errors="coerce").fillna(0.0)

        anomalies_merge2["volume_score"] = (
            (anomalies_merge2["avg_total_req"] / avg_tr_total) * 100 if avg_tr_total else 0.0
        )

        runtime = datetime.datetime.now().isoformat(timespec="seconds")
        messages = [f"SAME_DAY PROVIDER | Anomalies for: sales_date={eval_date} | Runtime : {runtime}"]

        for p in prov_alert:
            metric_alert = anomalies_merge2[anomalies_merge2["prov"] == p]["metric"].unique()
            for m in metric_alert:
                alert_filtered = anomalies_merge2[
                    (anomalies_merge2["prov"] == p) & (anomalies_merge2["metric"] == m)
                ].drop("prov", axis=1)

                if m == "site_issue_pct":
                    alert_filtered = (
                        alert_filtered[alert_filtered["direction"] != "DOWN"]
                        .sort_values("metric_val", ascending=False)
                        .drop("metric", axis=1)
                        .copy()
                    )
                    alert_filtered["metric_score"] = alert_filtered["metric_val"]

                elif m == "success_pct":
                    alert_filtered = (
                        alert_filtered[alert_filtered["direction"] != "UP"]
                        .sort_values("metric_val")
                        .drop("metric", axis=1)
                        .copy()
                    )
                    alert_filtered["metric_score"] = 100 - alert_filtered["metric_val"]

                elif m == "response_itin_ratio":
                    alert_filtered = (
                        alert_filtered[alert_filtered["direction"] != "UP"]
                        .sort_values("metric_val")
                        .drop("metric", axis=1)
                        .copy()
                    )
                    alert_filtered["metric_score"] = alert_filtered["metric_val"]
                    alert_filtered["metric_val"] = alert_filtered["metric_val"].map(lambda x: f"{x:,}")

                else:
                    alert_filtered = alert_filtered.drop("metric", axis=1).copy()
                    alert_filtered["metric_score"] = 0

                alert_final = alert_filtered.copy()
                alert_final["severity_score"] = round(
                    alert_final["metric_score"] + alert_final["cust_score"] + alert_final["volume_score"], 2
                )
                alert_final = (
                    alert_final.sort_values("severity_score", ascending=False)
                    .drop(["metric_score", "cust_score", "volume_score"], axis=1)
                )

                if alert_final.shape[0] > 0:
                    messages.append(f"{p} - {m}: \n")
                    messages.append(get_alert_table(alert_final))

        return messages
    finally:
        reader.close()


@function_tool
async def get_provider_anomaly(
    ctx: RunContextWrapper[AgentContext],
    sales_date: str,
) -> list[str]:
    """Return provider anomaly Slack-style summaries."""
    await _stream_progress(
        ctx,
        "search",
        f"get_provider_anomaly: {_short_json({'sales_date': sales_date})}",
    )
    return _get_provider_anomaly_impl(
        sales_date=sales_date,
    )


def monitoring_tools() -> list[Any]:
    return [
        read_table_head,
        query_table,
        get_top_site_issues,
        analyze_issue_scope,
        get_provider_anomaly,
    ]
