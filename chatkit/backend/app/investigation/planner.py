"""Autonomous plan/act loop for investigation requests."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


_TABLE_ALIASES: dict[str, str] = {
    "combined audit": "prod.monitoring.combined_audit",
    "combined_audit": "prod.monitoring.combined_audit",
    "provider combined audit": "prod.monitoring.provider_combined_audit",
    "provider_combined_audit": "prod.monitoring.provider_combined_audit",
}

_ANOMALY_BUCKET = "s3-atp-3victors-3vdev-use1-collection-anomalies"
_CUSTOMER_PREFIX = "collection-customer/v1"


class RuntimeActions(Protocol):
    def resolve_entities(self, input_text: str, sales_date_hint: str | None = None) -> dict[str, Any]: ...
    def retrieve_knowledge(self, *, intent: str, entities: dict[str, Any], question: str) -> dict[str, Any]: ...
    def inspect_table_metadata(self, table_name: str, datasource: str | None = None, capture_example_row: bool = True) -> dict[str, Any]: ...
    def extract_sql_to_dataset(
        self,
        *,
        thread_id: str,
        query: str,
        datasource: str,
        run_id: str,
        metadata: dict[str, Any] | None,
        dataset_name: str | None,
    ) -> dict[str, Any]: ...
    def extract_s3_to_dataset(
        self,
        *,
        thread_id: str,
        bucket: str,
        key_or_prefix: str,
        run_id: str,
        metadata: dict[str, Any] | None,
        dataset_name: str | None,
    ) -> dict[str, Any]: ...
    def run_dataframe_analysis(self, *, thread_id: str, run_id: str, dataset_ids: list[str], analysis_spec: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class LoopContext:
    thread_id: str
    run_id: str
    question: str
    sales_date: str
    constraints: dict[str, Any]
    entities: dict[str, Any] = field(default_factory=dict)
    knowledge: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    analysis: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    clarification: str | None = None
    strategy: str = "autonomous"


class AutonomousInvestigationEngine:
    """Codex-style iterative loop: PLAN -> ACT -> OBSERVE -> CHECK_DONE -> FINALIZE."""

    def __init__(self, runtime: RuntimeActions) -> None:
        self.runtime = runtime

    @staticmethod
    def _datasource_for_table(table_name: str) -> str:
        if table_name.startswith("priceeye."):
            return "mysql_priceeye"
        if table_name.startswith("prod.monitoring") or table_name.startswith("local.monitoring"):
            return "redshift_core"
        return "redshift_analytics"

    @staticmethod
    def _extract_explicit_table(question: str) -> str | None:
        match = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,2}\b", question)
        return match.group(0) if match else None

    @staticmethod
    def _resolve_alias(question: str) -> str | None:
        lowered = question.lower()
        for alias, table_name in _TABLE_ALIASES.items():
            if alias in lowered:
                return table_name
        if "combined audit" in lowered:
            return "prod.monitoring.combined_audit"
        return None

    @staticmethod
    def _sales_date_from_question(question: str, fallback: str) -> str:
        lowered = question.lower()
        if "yesterday" in lowered or "yersterday" in lowered or "yestarday" in lowered:
            return (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
        if "today" in lowered:
            return dt.date.today().strftime("%Y%m%d")
        compact = re.search(r"\b(20\d{6})\b", question)
        if compact:
            return compact.group(1)
        dashed = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
        if dashed:
            return dt.datetime.strptime(dashed.group(1), "%Y-%m-%d").strftime("%Y%m%d")
        return fallback

    @staticmethod
    def _pipe_codes(question: str) -> tuple[str | None, str | None]:
        match = re.search(r"\b([A-Za-z0-9]{2,})\|([A-Za-z0-9]{1,})\b", question)
        if not match:
            return None, None
        return match.group(1).upper(), match.group(2).upper()

    def _plan_actions(self, ctx: LoopContext) -> list[dict[str, Any]]:
        question = ctx.question
        lowered = question.lower()
        sales_date = ctx.sales_date
        entities = ctx.entities

        explicit = self._extract_explicit_table(question)
        aliased = self._resolve_alias(question)
        table_name = explicit or aliased

        candidates = [str(item) for item in ctx.knowledge.get("candidate_tables", []) if isinstance(item, str)]
        if not table_name and candidates:
            table_name = candidates[0]

        provider_from_pipe, site_from_pipe = self._pipe_codes(question)
        provider = provider_from_pipe or (entities.get("providers", [None])[0] if entities.get("providers") else None)
        site = site_from_pipe or (entities.get("sites", [None])[0] if entities.get("sites") else None)
        customer = entities.get("customers", [None])[0] if entities.get("customers") else None

        if not ctx.entities:
            return [{"action": "resolve_entities"}]

        actions: list[dict[str, Any]] = []

        asks_eda = any(token in lowered for token in (" eda", "profile", "explore table", "analyze table"))
        asks_head = "head of" in lowered or "head table" in lowered or "query the head" in lowered
        asks_distribution = "distribution" in lowered
        asks_issues = "site issue" in lowered or "top site" in lowered
        asks_collection = "collection anomalies" in lowered
        asks_market = "market anomal" in lowered or "impact score" in lowered

        if asks_eda and not table_name:
            ctx.clarification = "Please specify the table name for EDA (for example: `prod.monitoring.combined_audit`)."
            return actions

        if asks_eda and table_name:
            datasource = self._datasource_for_table(table_name)
            sample_rows = max(1000, min(int(ctx.constraints.get("sample_rows", 100000)), 200000))
            ctx.strategy = "table_eda"
            actions.extend(
                [
                    {"action": "inspect_metadata", "table": table_name, "datasource": datasource},
                    {
                        "action": "extract_sql",
                        "query": f"SELECT * FROM {table_name} LIMIT 200",
                        "datasource": datasource,
                        "dataset_name": "table_preview",
                    },
                    {
                        "action": "extract_sql",
                        "query": f"SELECT * FROM {table_name} LIMIT {sample_rows}",
                        "datasource": datasource,
                        "dataset_name": "table_profile_sample",
                    },
                    {
                        "action": "analyze",
                        "analysis_spec": {"type": "table_eda", "table_name": table_name, "sample_rows": sample_rows},
                    },
                ]
            )
            return actions

        if asks_head and table_name:
            datasource = self._datasource_for_table(table_name)
            ctx.strategy = "table_head"
            actions.extend(
                [
                    {"action": "inspect_metadata", "table": table_name, "datasource": datasource},
                    {
                        "action": "extract_sql",
                        "query": f"SELECT * FROM {table_name} LIMIT 200",
                        "datasource": datasource,
                        "dataset_name": "table_head",
                    },
                    {"action": "analyze", "analysis_spec": {"type": "summary"}},
                ]
            )
            return actions

        if asks_collection:
            ctx.strategy = "collection_anomaly_fetch"
            yyyy, mm, dd = sales_date[0:4], sales_date[4:6], sales_date[6:8]
            key = f"{_CUSTOMER_PREFIX}/{yyyy}/{mm}/{dd}/"
            actions.extend(
                [
                    {
                        "action": "extract_s3",
                        "bucket": _ANOMALY_BUCKET,
                        "key_or_prefix": key,
                        "dataset_name": "customer_collection_anomalies",
                    },
                    {"action": "analyze", "analysis_spec": {"type": "anomaly_summary", "confirmed_only": True, "top_n": 15}},
                ]
            )
            return actions

        if asks_issues:
            ctx.strategy = "site_issue_investigation"
            provider_filter = f"AND providercode = '{provider}'" if provider else ""
            site_filter = f"AND sitecode = '{site}'" if site else ""
            if provider is None:
                ctx.warnings.append("Provider code not detected; query is broader than requested.")
            query_top = (
                "SELECT issue_sources, issue_reasons, providercode, sitecode, COUNT(*) AS issue_count "
                "FROM prod.monitoring.provider_combined_audit "
                f"WHERE sales_date = {sales_date} {provider_filter} {site_filter} "
                "AND issue_sources <> 'request' AND issue_sources <> '' AND issue_reasons <> '' "
                "GROUP BY issue_sources, issue_reasons, providercode, sitecode "
                "ORDER BY issue_count DESC"
            )
            query_impact = (
                "SELECT providercode, sitecode, COUNT(*) AS total_requests, "
                "SUM(CASE WHEN (issue_sources <> '' OR filterreason <> '') THEN 1 ELSE 0 END) AS issue_requests, "
                "ROUND(100.0 * SUM(CASE WHEN (issue_sources <> '' OR filterreason <> '') THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS issue_rate_pct "
                "FROM prod.monitoring.provider_combined_audit "
                f"WHERE sales_date = {sales_date} {provider_filter} {site_filter} "
                "GROUP BY providercode, sitecode ORDER BY issue_rate_pct DESC"
            )
            actions.extend(
                [
                    {"action": "inspect_metadata", "table": "prod.monitoring.provider_combined_audit", "datasource": "redshift_core"},
                    {"action": "extract_sql", "query": query_top, "datasource": "redshift_core", "dataset_name": "site_issue_groups"},
                    {"action": "extract_sql", "query": query_impact, "datasource": "redshift_core", "dataset_name": "issue_impact"},
                    {"action": "analyze", "analysis_spec": {"type": "issue_impact"}},
                ]
            )
            return actions

        if asks_market:
            ctx.strategy = "market_anomaly_distribution"
            customer_filter = f"AND customer = '{customer}'" if customer else ""
            if customer is None:
                ctx.warnings.append("Customer code not detected; query runs across all customers.")
            query = (
                "SELECT observation_date, mkt, seg, top_offenders, cp, dow, impact_score, customer, sales_date "
                "FROM prod.analytics.market_level_anomalies_v3 "
                f"WHERE sales_date = {sales_date} {customer_filter} AND any_anomaly = 1 "
                "ORDER BY impact_score DESC"
            )
            analysis = {"type": "distribution", "column": "impact_score", "bucket_count": int(ctx.constraints.get("bucket_count", 12))}
            if not asks_distribution:
                analysis = {"type": "summary"}
            actions.extend(
                [
                    {"action": "inspect_metadata", "table": "prod.analytics.market_level_anomalies_v3", "datasource": "redshift_analytics"},
                    {"action": "extract_sql", "query": query, "datasource": "redshift_analytics", "dataset_name": "market_anomalies"},
                    {"action": "analyze", "analysis_spec": analysis},
                ]
            )
            return actions

        if table_name:
            datasource = self._datasource_for_table(table_name)
            ctx.strategy = "generic_table_preview"
            actions.extend(
                [
                    {"action": "inspect_metadata", "table": table_name, "datasource": datasource},
                    {
                        "action": "extract_sql",
                        "query": f"SELECT * FROM {table_name} LIMIT 200",
                        "datasource": datasource,
                        "dataset_name": "table_preview",
                    },
                    {"action": "analyze", "analysis_spec": {"type": "summary"}},
                ]
            )
            return actions

        ctx.clarification = (
            "I need one concrete table, dataset, or business scope to investigate. "
            "Example: `EDA of prod.monitoring.combined_audit` or `top site issues for QL2|AV`."
        )
        return actions

    def _execute_action(self, ctx: LoopContext, action: dict[str, Any]) -> None:
        name = action.get("action")

        if name == "resolve_entities":
            ctx.entities = self.runtime.resolve_entities(ctx.question, sales_date_hint=ctx.sales_date)
            ctx.knowledge = self.runtime.retrieve_knowledge(intent="autonomous", entities=ctx.entities, question=ctx.question)
            return

        if name == "inspect_metadata":
            self.runtime.inspect_table_metadata(
                table_name=str(action.get("table")),
                datasource=str(action.get("datasource")) if action.get("datasource") else None,
                capture_example_row=True,
            )
            return

        if name == "extract_sql":
            record = self.runtime.extract_sql_to_dataset(
                thread_id=ctx.thread_id,
                query=str(action.get("query", "")),
                datasource=str(action.get("datasource", "redshift_analytics")),
                run_id=ctx.run_id,
                metadata={"strategy": ctx.strategy, "query": str(action.get("query", ""))},
                dataset_name=str(action.get("dataset_name") or "sql_extract"),
            )
            ctx.datasets.append(record)
            return

        if name == "extract_s3":
            record = self.runtime.extract_s3_to_dataset(
                thread_id=ctx.thread_id,
                bucket=str(action.get("bucket", _ANOMALY_BUCKET)),
                key_or_prefix=str(action.get("key_or_prefix", "")),
                run_id=ctx.run_id,
                metadata={"strategy": ctx.strategy, "bucket": str(action.get("bucket", _ANOMALY_BUCKET))},
                dataset_name=str(action.get("dataset_name") or "s3_extract"),
            )
            ctx.datasets.append(record)
            return

        if name == "analyze":
            dataset_ids = [item["dataset_id"] for item in ctx.datasets if item.get("dataset_id")]
            if not dataset_ids:
                ctx.warnings.append("No datasets available for analysis.")
                return
            ctx.analysis = self.runtime.run_dataframe_analysis(
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                dataset_ids=dataset_ids,
                analysis_spec=dict(action.get("analysis_spec", {"type": "summary"})),
            )
            return

        if name == "ask_clarification":
            ctx.clarification = str(action.get("message") or "Need more details to continue.")
            return

        if name == "finish":
            return

        raise ValueError(f"Unsupported action: {name}")

    @staticmethod
    def _done(ctx: LoopContext) -> bool:
        question = ctx.question.lower()
        data_task = any(
            token in question
            for token in ["query", "table", "eda", "impact", "anomal", "issue", "distribution", "head"]
        )
        has_dataset = len(ctx.datasets) > 0
        has_analysis = ctx.analysis is not None
        if not data_task:
            return True
        if ctx.clarification:
            return True
        return has_dataset and has_analysis

    def run(self, *, thread_id: str, run_id: str, question: str, sales_date: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = LoopContext(
            thread_id=thread_id,
            run_id=run_id,
            question=question,
            sales_date=self._sales_date_from_question(question, sales_date),
            constraints=constraints or {},
        )

        state = "PLAN"
        while True:
            if state == "PLAN":
                ctx.actions = self._plan_actions(ctx)
                state = "ACT"
                continue

            if state == "ACT":
                for action in ctx.actions:
                    if ctx.clarification:
                        break
                    self._execute_action(ctx, action)
                state = "OBSERVE"
                continue

            if state == "OBSERVE":
                state = "CHECK_DONE"
                continue

            if state == "CHECK_DONE":
                if self._done(ctx):
                    state = "FINALIZE"
                elif not ctx.clarification and not ctx.datasets:
                    state = "PLAN"
                else:
                    ctx.actions = [{"action": "ask_clarification", "message": "Please narrow the request to a table or business scope."}]
                    state = "ACT"
                continue

            if state == "FINALIZE":
                return {
                    "strategy": ctx.strategy,
                    "sales_date": ctx.sales_date,
                    "entities": ctx.entities,
                    "knowledge": ctx.knowledge,
                    "datasets": ctx.datasets,
                    "analysis": ctx.analysis,
                    "warnings": ctx.warnings,
                    "clarification": ctx.clarification,
                }


__all__ = ["AutonomousInvestigationEngine"]
