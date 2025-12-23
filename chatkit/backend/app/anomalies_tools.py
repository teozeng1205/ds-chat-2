"""Custom function tools for the market anomalies agent."""

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

    def get_anomalies_overiew(
        self,
        sales_date: str | None,
        customer: str,
    ) -> pd.DataFrame:
        """
        Return market-level anomalies for a customer on a sales date.

        Args:
            sales_date: Date in YYYYMMDD format (default: today)
            customer: Customer code (e.g., 'B6', 'AA')
        """
        if not customer or not customer.strip():
            raise ValueError("customer is required (e.g., 'B6', 'AA')")
        customer = customer.strip()

        if sales_date is None:
            sales_date = datetime.date.today().strftime("%Y%m%d")

        query = f"""
        SELECT observation_date, mkt, seg, top_offenders, impact_dates, cp, dow
        FROM prod.analytics.market_level_anomalies_v3
        WHERE sales_date = {sales_date}
          AND customer = '{customer}'
          AND any_anomaly = 1
        ORDER BY impact_score;
        """

        log.info("Getting anomalies overview for customer=%s date=%s", customer, sales_date)
        with self.get_connection().cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=colnames)
            log.info("Found %s anomalies", len(df))
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


def anomalies_instructions() -> str:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"You are a market anomalies assistant. Today is {current_date}.\n"
        "Use get_anomalies_overiew(sales_date, customer) to fetch anomalies.\n"
        "customer is required (e.g., 'B6', 'AA'); sales_date defaults to today (YYYYMMDD).\n"
        "Never modify data and cite which tool produced each insight."
    )


@function_tool
async def get_anomalies_overiew(
    ctx: RunContextWrapper[AgentContext],
    customer: str,
    sales_date: str | None = None,
) -> list[dict[str, Any]]:
    """Return market-level anomalies for a customer and date."""
    await _stream_progress(
        ctx,
        "search",
        f"get_anomalies_overiew: {_short_json({'customer': customer, 'sales_date': sales_date})}",
    )
    return _df_records(_get_reader().get_anomalies_overiew(sales_date, customer))


def anomalies_tools() -> list[Any]:
    return [get_anomalies_overiew]
