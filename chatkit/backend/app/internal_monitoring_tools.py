"""Dedicated S3-backed tools for internal monitoring anomalies."""

from __future__ import annotations

import datetime
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent

from .monitoring_tools import (
    DEFAULT_ANOMALY_LIMIT,
    _get_internal_monitoring_anomalies_impl,
    analyze_issue_scope,
    get_top_site_issues,
    query_table,
    read_table_head,
)


async def _stream_progress(
    ctx: RunContextWrapper[AgentContext],
    icon: str,
    text: str,
) -> None:
    await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))


def internal_monitoring_instructions() -> str:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"You are an internal monitoring anomalies assistant. Today is {current_date}.\n"
        "Always use tools to answer.\n"
        "Choose the tool based on user intent, not by defaulting to S3 anomalies.\n"
        "Tool routing rules:\n"
        "1) If user asks for top site issues (e.g., 'top site issues', 'site issues for QL2'), use get_top_site_issues(...) first.\n"
        "2) If user asks for site-issue drilldown/scope for provider/customer/site/date, use analyze_issue_scope(...).\n"
        "3) If user explicitly asks for anomalies, anomaly counts, anomaly records, or anomaly_t1/t2 logic, use get_monitoring_anomalies(...).\n"
        "4) If user asks for custom SQL/table exploration, use read_table_head(...) and query_table(...).\n"
        "For get_monitoring_anomalies(...): fetch from S3 partitions (customer/provider/late-request); treat as anomalies only when anomaly_t1=1 and anomaly_t2=1.\n"
        "Default sales_date to today unless user specifies a date.\n"
        "For large datasets, apply filtering fields (providercode, sitecode, customer, metric_name, model_type, limit).\n"
        "If S3 partitions are missing, state which partitions are missing and continue with available data.\n"
        "When question is about site issues, do not call get_monitoring_anomalies unless user asked for anomalies."
    )


@function_tool
async def get_monitoring_anomalies(
    ctx: RunContextWrapper[AgentContext],
    sales_date: str | None = None,
    providercode: str | None = None,
    sitecode: str | None = None,
    customer: str | None = None,
    metric_name: str | None = None,
    model_type: str | None = None,
    limit: int = DEFAULT_ANOMALY_LIMIT,
) -> dict[str, Any]:
    """Get confirmed anomalies from S3 (customer, provider, and laterequests) for a date."""
    target_date = sales_date or datetime.date.today().strftime("%Y%m%d")
    await _stream_progress(ctx, "search", f"Loading internal monitoring anomalies for {target_date}.")
    await _stream_progress(ctx, "clock", "Checking S3 partitions and reading anomaly CSVs.")
    try:
        result = _get_internal_monitoring_anomalies_impl(
            sales_date=sales_date,
            providercode=providercode,
            sitecode=sitecode,
            customer=customer,
            metric_name=metric_name,
            model_type=model_type,
            limit=limit,
        )
        dataset_counts = result.get("results", {})
        total_confirmed = sum(
            int(dataset.get("confirmed_anomalies", 0)) for dataset in dataset_counts.values()
        )
        available = result.get("available_partitions", {})
        available_names = [name for name, exists in available.items() if exists]
        await _stream_progress(
            ctx,
            "check-circle",
            (
                f"S3 anomaly extraction complete: {total_confirmed} confirmed anomalies across "
                f"{', '.join(available_names) if available_names else 'no'} partitions."
            ),
        )
        missing = result.get("missing_partitions", [])
        if missing:
            await _stream_progress(ctx, "info", f"Missing partitions for {target_date}: {', '.join(missing)}.")
        return result
    except Exception as exc:
        await _stream_progress(ctx, "bug", f"S3 anomaly extraction failed: {type(exc).__name__}.")
        raise


def internal_monitoring_tools() -> list[Any]:
    return [
        get_monitoring_anomalies,
        get_top_site_issues,
        analyze_issue_scope,
        read_table_head,
        query_table,
    ]
