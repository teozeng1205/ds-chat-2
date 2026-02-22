"""Generic knowledge-driven tools for DS Chat Next-Gen."""

from __future__ import annotations

import datetime
import json
import logging
import re
import uuid
from typing import Any

import numpy as np
import pandas as pd
from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent

from .entity_resolver import EntityResolver
from .knowledge_base import KnowledgeBaseService
from .nextgen_types import AnalysisResult, EntityResolution, InvestigationPlan, PlanFilter, PlanStep
from .partition_policy import apply_default_required_filters, build_where_clause, ensure_partition_filters
from .threevictors_client import ThreeVictorsClient
from .workspace_manager import TurnWorkspace

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
stream_handler.setFormatter(formatter)
if not any(getattr(handler, "name", "") == "nextgen_tools" for handler in log.handlers):
    stream_handler.name = "nextgen_tools"
    log.addHandler(stream_handler)
log.propagate = False


_SELECT_COL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _today_yyyymmdd() -> str:
    return datetime.date.today().strftime("%Y%m%d")


def _coerce_sales_date(value: str | None) -> str:
    if not value:
        return _today_yyyymmdd()
    raw = value.strip()
    if raw.lower() == "today":
        return _today_yyyymmdd()
    if raw.lower() == "yesterday":
        return (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y%m%d")
    if len(raw) == 8 and raw.isdigit():
        return raw
    return datetime.datetime.strptime(raw, "%Y-%m-%d").strftime("%Y%m%d")


def _request_context(ctx: RunContextWrapper[AgentContext]) -> dict[str, Any]:
    if not isinstance(ctx.context.request_context, dict):
        raise RuntimeError("request_context must be dict for DS Chat Next-Gen tools")
    return ctx.context.request_context


def _kb(ctx: RunContextWrapper[AgentContext]) -> KnowledgeBaseService:
    req = _request_context(ctx)
    kb = req.get("nextgen_kb")
    if not isinstance(kb, KnowledgeBaseService):
        raise RuntimeError("nextgen_kb is not initialized in request_context")
    kb.refresh_if_needed()
    return kb


def _workspace(ctx: RunContextWrapper[AgentContext]) -> TurnWorkspace:
    req = _request_context(ctx)
    workspace = req.get("nextgen_workspace")
    if not isinstance(workspace, TurnWorkspace):
        raise RuntimeError("nextgen_workspace is not initialized in request_context")
    return workspace


def _tv_client(ctx: RunContextWrapper[AgentContext]) -> ThreeVictorsClient:
    req = _request_context(ctx)
    client = req.get("nextgen_threevictors")
    if not isinstance(client, ThreeVictorsClient):
        raise RuntimeError("nextgen_threevictors is not initialized in request_context")
    return client


def _resolver(ctx: RunContextWrapper[AgentContext]) -> EntityResolver:
    req = _request_context(ctx)
    resolver = req.get("nextgen_entity_resolver")
    if isinstance(resolver, EntityResolver):
        return resolver

    resolver = EntityResolver(_kb(ctx), _tv_client(ctx))
    req["nextgen_entity_resolver"] = resolver
    return resolver


async def _stream_progress(ctx: RunContextWrapper[AgentContext], icon: str, text: str) -> None:
    await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))


def _parse_filters_json(raw: str | None) -> list[PlanFilter]:
    if not raw:
        return []
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("filters_json must be a JSON list")
    return [PlanFilter.model_validate(item) for item in payload]


def _columns_list(raw: str) -> list[str]:
    if raw.strip() == "*":
        return ["*"]
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("columns must be '*' or a comma-separated list")
    for value in values:
        if not _SELECT_COL_RE.match(value):
            raise ValueError(f"Unsupported column name: {value}")
    return values


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("s3_uri must start with s3://")
    tail = uri[5:]
    if "/" not in tail:
        return tail, ""
    bucket, key = tail.split("/", 1)
    return bucket, key


def _collect_partition_values(filters: list[PlanFilter], required: list[str]) -> dict[str, Any]:
    by_col = {f.column: f.value for f in filters}
    return {predicate: by_col.get(predicate) for predicate in required}


def _resolve_entities_impl(
    ctx: RunContextWrapper[AgentContext],
    codes: str,
) -> list[EntityResolution]:
    resolver = _resolver(ctx)
    req = _request_context(ctx)
    cache = req.setdefault("nextgen_entity_cache", {})
    return resolver.resolve_codes(codes, cache=cache)


@function_tool
async def search_kb(
    ctx: RunContextWrapper[AgentContext],
    query: str,
    top_k: int = 8,
) -> dict[str, Any]:
    """Search local knowledge base docs, playbooks, and table metadata."""
    await _stream_progress(ctx, "search", f"Searching knowledge base for: {query}")
    kb = _kb(ctx)
    results = kb.search(query, top_k=max(1, min(top_k, 20)))
    await _stream_progress(ctx, "check-circle", f"Knowledge base search complete: {len(results)} results.")
    return {
        "query": query,
        "top_k": top_k,
        "results": results,
    }


@function_tool
async def resolve_entities(
    ctx: RunContextWrapper[AgentContext],
    codes: str,
) -> list[dict[str, Any]]:
    """Resolve provider/site/customer codes via common list then MySQL fallback."""
    await _stream_progress(ctx, "search", "Resolving provider/site/customer codes.")
    records = _resolve_entities_impl(ctx, codes)
    await _stream_progress(ctx, "check-circle", f"Entity resolution complete: {len(records)} input code(s).")
    return [record.model_dump(mode="json") for record in records]


@function_tool
async def build_investigation_plan(
    ctx: RunContextWrapper[AgentContext],
    user_question: str,
    sales_date: str | None = None,
    lookback_days: int = 7,
    entity_codes: str | None = None,
    customer_code: str | None = None,
) -> dict[str, Any]:
    """Build strict InvestigationPlan JSON from KB metadata and entities (no raw SQL)."""
    kb = _kb(ctx)
    target_date = _coerce_sales_date(sales_date)
    playbook = kb.match_playbook(user_question)

    entity_records: list[EntityResolution] = []
    if entity_codes:
        entity_records = _resolve_entities_impl(ctx, entity_codes)

    if playbook is None:
        plan = InvestigationPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:10]}",
            intent="generic_investigation",
            question=user_question,
            requires_clarification=True,
            clarification_prompt=(
                "I couldn't match a playbook. Please specify the metric/intent and target entity "
                "(provider/site/customer) so I can generate a partition-safe plan."
            ),
            assumptions=["No playbook matched from local KB."],
            steps=[
                PlanStep(
                    step_id="step_search_kb",
                    step_type="summarize",
                    description="Search knowledge base for matching table/playbook context.",
                    options={"query": user_question},
                )
            ],
        )
        return plan.model_dump(mode="json")

    table_id = str(playbook.get("default_table_id"))
    table_spec = kb.get_table(table_id)

    filters: list[PlanFilter] = []
    if target_date:
        date_column = table_spec.default_date_column or "sales_date"
        filters.append(PlanFilter(column=date_column, operator="=", value=target_date))

    if customer_code:
        customer_column = table_spec.default_customer_column or "customer"
        filters.append(PlanFilter(column=customer_column, operator="=", value=customer_code.upper()))

    for record in entity_records:
        if record.entity_type == "unknown" or not record.canonical_value:
            continue
        target_column = table_spec.entity_columns.get(record.entity_type)
        if target_column:
            filters.append(PlanFilter(column=target_column, operator="=", value=record.canonical_value))

    default_map = {
        (table_spec.default_date_column or "sales_date"): target_date,
    }
    filters = apply_default_required_filters(table_spec, filters, defaults=default_map)

    missing: list[str] = []
    try:
        ensure_partition_filters(table_spec, filters)
    except ValueError as exc:
        text = str(exc)
        if ":" in text:
            missing = [item.strip() for item in text.split(":", 1)[1].split(",") if item.strip()]

    plan = InvestigationPlan(
        plan_id=f"plan_{uuid.uuid4().hex[:10]}",
        intent=str(playbook.get("intent", playbook.get("playbook_id", "investigation"))),
        question=user_question,
        requires_clarification=bool(missing),
        clarification_prompt=(
            f"Please provide partition value(s) for: {', '.join(missing)}"
            if missing
            else None
        ),
        missing_predicates=missing,
        assumptions=[
            f"Environment fixed to 3VDEV.",
            f"Lookback days set to {max(1, lookback_days)}.",
        ],
        steps=[
            PlanStep(
                step_id="step_sql_extract",
                step_type="sql_extract",
                description="Extract partition-safe dataset from KB table.",
                table_id=table_id,
                filters=filters,
                output_dataset_id=f"{table_id}_{target_date}",
                options={
                    "columns": ",".join(playbook.get("default_columns", ["*"])),
                    "limit": int(playbook.get("default_limit", table_spec.default_limit)),
                },
            ),
            PlanStep(
                step_id="step_python_analysis",
                step_type="python_analysis",
                description="Run offline pandas analysis on extracted dataset.",
                input_datasets=[f"{table_id}_{target_date}"],
                output_dataset_id=f"{table_id}_{target_date}_analysis",
                options={
                    "group_by": ",".join(playbook.get("analysis", {}).get("group_by", [])),
                    "top_n": int(playbook.get("analysis", {}).get("top_n", 20)),
                },
            ),
            PlanStep(
                step_id="step_summarize",
                step_type="summarize",
                description="Summarize findings and scope.",
                input_datasets=[f"{table_id}_{target_date}_analysis"],
            ),
        ],
    )
    return plan.model_dump(mode="json")


@function_tool
async def execute_sql_extract(
    ctx: RunContextWrapper[AgentContext],
    table_id: str,
    filters_json: str,
    columns: str = "*",
    limit: int = 5000,
    dataset_id: str | None = None,
    source_step_id: str | None = None,
) -> dict[str, Any]:
    """Extract rows from KB tables with enforced partition predicates."""
    kb = _kb(ctx)
    workspace = _workspace(ctx)
    tv = _tv_client(ctx)

    spec = kb.get_table(table_id)
    filters = _parse_filters_json(filters_json)
    ensure_partition_filters(spec, filters)

    cols = _columns_list(columns)
    select_clause = "*" if cols == ["*"] else ", ".join(cols)
    where_clause = build_where_clause(filters)
    clamped_limit = max(1, min(int(limit), 200_000))

    query = f"SELECT {select_clause} FROM {spec.physical_name} WHERE {where_clause} LIMIT {clamped_limit}"

    await _stream_progress(ctx, "clock", f"Executing {spec.source_system} extract for table {table_id}.")
    if spec.source_system == "redshift":
        frame = tv.query_redshift(query)
    elif spec.source_system == "mysql":
        frame = tv.query_mysql(query)
    else:
        raise ValueError(f"Table '{table_id}' is source_system={spec.source_system}; use execute_s3_extract instead")

    handle, manifest = workspace.write_dataset(
        df=frame,
        dataset_id=dataset_id or f"{table_id}_{uuid.uuid4().hex[:8]}",
        source_type="sql",
        source_ref=spec.physical_name,
        query=query,
        partitions=_collect_partition_values(filters, spec.partition_policy.required_predicates),
        lineage=[source_step_id] if source_step_id else [],
        source_step_id=source_step_id,
    )
    await _stream_progress(ctx, "check-circle", f"SQL extract complete: {handle.row_count} rows.")

    return {
        "dataset": handle.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json", by_alias=True),
        "query": query,
    }


@function_tool
async def execute_s3_extract(
    ctx: RunContextWrapper[AgentContext],
    s3_uri: str,
    format_hint: str = "auto",
    dataset_id: str | None = None,
    source_step_id: str | None = None,
    max_files: int = 20,
    limit_rows: int = 200_000,
) -> dict[str, Any]:
    """Extract from S3 (prefer parquet, fallback csv), normalize to local parquet dataset."""
    workspace = _workspace(ctx)
    tv = _tv_client(ctx)

    bucket, key = _parse_s3_uri(s3_uri)
    if not bucket:
        raise ValueError("Invalid s3_uri: missing bucket")

    if key.endswith("/") or key == "":
        keys = tv.list_s3_keys(bucket, key)
        parquet_keys = [item for item in keys if item.lower().endswith(".parquet")]
        csv_keys = [item for item in keys if item.lower().endswith(".csv")]
        selected = parquet_keys if parquet_keys else csv_keys
    else:
        selected = [key]

    selected = selected[: max(1, min(max_files, 100))]
    if not selected:
        raise ValueError(f"No parquet/csv objects found under {s3_uri}")

    await _stream_progress(ctx, "clock", f"Reading {len(selected)} object(s) from S3.")
    frames: list[pd.DataFrame] = []
    for object_key in selected:
        frame = tv.read_s3_table(bucket, object_key, format_hint=format_hint)
        if frame.empty:
            continue
        tagged = frame.copy()
        tagged["source_key"] = object_key
        frames.append(tagged)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(merged) > limit_rows:
        merged = merged.head(limit_rows).copy()

    handle, manifest = workspace.write_dataset(
        df=merged,
        dataset_id=dataset_id or f"s3_{uuid.uuid4().hex[:8]}",
        source_type="s3",
        source_ref=s3_uri,
        s3_keys=selected,
        lineage=[source_step_id] if source_step_id else [],
        source_step_id=source_step_id,
    )
    await _stream_progress(ctx, "check-circle", f"S3 extract complete: {handle.row_count} rows.")

    return {
        "dataset": handle.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json", by_alias=True),
        "selected_keys": selected,
    }


@function_tool
async def join_datasets(
    ctx: RunContextWrapper[AgentContext],
    left_dataset_id: str,
    right_dataset_id: str,
    left_on: str,
    right_on: str,
    how: str = "inner",
    dataset_id: str | None = None,
    source_step_id: str | None = None,
) -> dict[str, Any]:
    """Join two local datasets offline in pandas and persist a joined dataset artifact."""
    workspace = _workspace(ctx)
    supported_how = {"inner", "left", "right", "outer"}
    if how.lower() not in supported_how:
        raise ValueError(f"Unsupported join type '{how}'. Use one of: {', '.join(sorted(supported_how))}")

    left_frame = workspace.read_dataset(left_dataset_id)
    right_frame = workspace.read_dataset(right_dataset_id)

    left_cols = [value.strip() for value in left_on.split(",") if value.strip()]
    right_cols = [value.strip() for value in right_on.split(",") if value.strip()]
    if len(left_cols) != len(right_cols):
        raise ValueError("left_on and right_on must include the same number of comma-separated columns")

    joined = left_frame.merge(
        right_frame,
        left_on=left_cols,
        right_on=right_cols,
        how=how.lower(),
        suffixes=("_left", "_right"),
    )

    handle, manifest = workspace.write_dataset(
        df=joined,
        dataset_id=dataset_id or f"join_{uuid.uuid4().hex[:8]}",
        source_type="join",
        source_ref=f"join:{left_dataset_id}:{right_dataset_id}",
        lineage=[left_dataset_id, right_dataset_id] + ([source_step_id] if source_step_id else []),
        source_step_id=source_step_id,
    )

    return {
        "dataset": handle.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json", by_alias=True),
        "join": {
            "left_dataset_id": left_dataset_id,
            "right_dataset_id": right_dataset_id,
            "left_on": left_cols,
            "right_on": right_cols,
            "how": how.lower(),
        },
    }


@function_tool
async def run_python_analysis(
    ctx: RunContextWrapper[AgentContext],
    dataset_id: str,
    python_code: str | None = None,
    group_by: str | None = None,
    metric_columns: str | None = None,
    top_n: int = 20,
    output_dataset_id: str | None = None,
    analysis_name: str = "analysis",
    source_step_id: str | None = None,
) -> dict[str, Any]:
    """Run offline pandas analysis on a local dataset and optionally persist output dataframe."""
    workspace = _workspace(ctx)
    frame = workspace.read_dataset(dataset_id)
    clamped_top_n = max(1, min(top_n, 200))

    output_df: pd.DataFrame | None = None
    warnings: list[str] = []
    metrics: dict[str, Any] = {"input_rows": int(len(frame))}

    if python_code and python_code.strip():
        await _stream_progress(ctx, "clock", "Running custom python analysis code.")
        scope: dict[str, Any] = {
            "pd": pd,
            "np": np,
            "df": frame.copy(),
            "result": {},
            "output_df": None,
        }
        exec(compile(python_code, "<agent-python-analysis>", "exec"), scope, scope)

        result = scope.get("result", {})
        if isinstance(result, dict):
            metrics.update(result)
        else:
            metrics["result"] = str(result)

        candidate_output = scope.get("output_df")
        if isinstance(candidate_output, pd.DataFrame):
            output_df = candidate_output

    else:
        if group_by:
            groups = [col.strip() for col in group_by.split(",") if col.strip()]
            missing = [col for col in groups if col not in frame.columns]
            if missing:
                raise ValueError(f"group_by column(s) missing in dataset {dataset_id}: {', '.join(missing)}")
            output_df = (
                frame.groupby(groups, dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False, kind="stable")
                .head(clamped_top_n)
            )
            metrics["group_by"] = groups
            metrics["top_groups"] = output_df.to_dict(orient="records")
        else:
            metrics["columns"] = [str(col) for col in frame.columns]
            metrics["null_counts"] = {
                str(col): int(count)
                for col, count in frame.isnull().sum().to_dict().items()
            }
            numeric_cols = [str(col) for col in frame.select_dtypes(include=["number"]).columns]
            metrics["numeric_columns"] = numeric_cols
            if metric_columns:
                selected = [col.strip() for col in metric_columns.split(",") if col.strip()]
                selected_numeric = [col for col in selected if col in numeric_cols]
                if selected and not selected_numeric:
                    warnings.append("metric_columns provided but none are numeric columns.")
                if selected_numeric:
                    metrics["metric_describe"] = (
                        frame[selected_numeric].describe().fillna("").to_dict()
                    )

    artifacts: list[str] = []
    if isinstance(output_df, pd.DataFrame) and output_dataset_id:
        handle, _manifest = workspace.write_dataset(
            df=output_df,
            dataset_id=output_dataset_id,
            source_type="python",
            source_ref=f"analysis:{analysis_name}:{dataset_id}",
            lineage=[dataset_id] + ([source_step_id] if source_step_id else []),
            source_step_id=source_step_id,
        )
        artifacts.append(handle.dataset_id)

    analysis = AnalysisResult(
        analysis_name=analysis_name,
        summary=f"Analysis complete for dataset '{dataset_id}' with {len(frame)} input rows.",
        metrics=metrics,
        artifacts=artifacts,
        warnings=warnings,
    )
    await _stream_progress(ctx, "check-circle", "Python analysis complete.")
    return analysis.model_dump(mode="json")


@function_tool
async def summarize_findings(
    ctx: RunContextWrapper[AgentContext],
    question: str | None = None,
    analysis_json: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Summarize current findings with dataset lineage and key metrics."""
    workspace = _workspace(ctx)
    parts: list[str] = []
    payload: dict[str, Any] = {}

    if question:
        parts.append(f"Question: {question}")

    if analysis_json:
        parsed = json.loads(analysis_json)
        payload["analysis"] = parsed
        if isinstance(parsed, dict):
            summary = parsed.get("summary")
            if summary:
                parts.append(str(summary))
            metrics = parsed.get("metrics")
            if isinstance(metrics, dict):
                if "top_groups" in metrics:
                    top_groups = metrics.get("top_groups")
                    parts.append(f"Top grouped findings rows: {len(top_groups) if isinstance(top_groups, list) else 0}")
                input_rows = metrics.get("input_rows")
                if input_rows is not None:
                    parts.append(f"Input rows analyzed: {input_rows}")

    if dataset_id:
        manifest = workspace.read_manifest(dataset_id)
        payload["dataset_manifest"] = manifest.model_dump(mode="json", by_alias=True)
        parts.append(
            f"Dataset {dataset_id}: {manifest.row_count} rows from {manifest.source_type} source ({manifest.source_ref})."
        )

    if not parts:
        parts.append("No findings payload was provided; run extract/analysis tools first.")

    return {
        "summary": " ".join(parts),
        "details": payload,
    }


@function_tool
async def list_workspace_datasets(
    ctx: RunContextWrapper[AgentContext],
) -> dict[str, Any]:
    """List local dataset artifacts currently available in this user-turn workspace."""
    workspace = _workspace(ctx)
    dataset_ids = workspace.list_dataset_ids()
    manifests: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        manifests.append(workspace.read_manifest(dataset_id).model_dump(mode="json", by_alias=True))
    return {
        "workspace": str(workspace.root_path),
        "dataset_ids": dataset_ids,
        "manifests": manifests,
    }


def knowledge_planner_instructions() -> str:
    today = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"You are the Knowledge Planner Agent. Today is {today}. "
        "Always search local KB first, resolve entities, and produce strict InvestigationPlan JSON. "
        "Never generate raw SQL directly. If required partition values are missing, ask a concise clarification."
    )


def data_access_instructions() -> str:
    today = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"You are the Data Access Agent. Today is {today}. "
        "Use only execute_sql_extract/execute_s3_extract/join_datasets/list_workspace_datasets. "
        "Enforce partition-safe extraction and keep everything in local turn workspace artifacts."
    )


def analysis_instructions() -> str:
    today = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"You are the Analysis Agent. Today is {today}. "
        "Use run_python_analysis and join_datasets over local dataset artifacts. "
        "Do offline pandas analysis and preserve lineage in outputs."
    )


def synthesis_instructions() -> str:
    today = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"You are the Synthesis Agent. Today is {today}. "
        "Use summarize_findings and list_workspace_datasets to produce concise evidence-based conclusions with caveats."
    )


def planner_tools() -> list[Any]:
    return [search_kb, resolve_entities, build_investigation_plan]


def data_access_tools() -> list[Any]:
    return [execute_sql_extract, execute_s3_extract, join_datasets, list_workspace_datasets, search_kb]


def analysis_tools() -> list[Any]:
    return [run_python_analysis, join_datasets, list_workspace_datasets, summarize_findings]


def synthesis_tools() -> list[Any]:
    return [summarize_findings, list_workspace_datasets, search_kb]
