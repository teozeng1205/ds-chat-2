"""Function tools exposed to the autonomous investigation agent."""

from __future__ import annotations

import datetime
import json
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent

from ..investigation.runtime import cleanup_thread_workspace, get_runtime, is_investigation_engine_enabled


def _thread_id(ctx: RunContextWrapper[AgentContext]) -> str:
    thread = getattr(ctx.context, "thread", None)
    thread_id = getattr(thread, "id", None)
    return str(thread_id) if thread_id else "default-thread"


async def _stream_progress(ctx: RunContextWrapper[AgentContext], icon: str, text: str) -> None:
    await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))


def _parse_json_object(raw: str | None, *, field_name: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} must be a JSON object string") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return parsed


def investigation_instructions() -> str:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"You are an autonomous investigation operator. Today is {current_date}.\n"
        "Operate in iterative plan/act/observe loops using tools until done criteria are met.\n"
        "Do not rely on predefined intent labels; infer strategy from evidence and KB context.\n"
        "Always ground conclusions in local artifacts and include lineage with run_id and dataset IDs.\n"
        "Prefer extract_sql_to_dataset/extract_s3_to_dataset, then run_dataframe_analysis/operator_run_python.\n"
        "Use run_table_eda when the user explicitly asks for EDA/profile exploration of a table.\n"
        "Use inspect_table_metadata for unknown or discovered tables.\n"
        "Use browse_knowledge_files for local docs and KB sources.\n"
        "Keep SQL read-only and call out caveats for sampled or broad scans.\n"
    )


@function_tool
async def browse_knowledge_files(ctx: RunContextWrapper[AgentContext], path_or_glob: str) -> dict[str, Any]:
    """Browse local KB source files directly (tables/docs/codes/practices)."""
    await _stream_progress(ctx, "search", f"Browsing knowledge files for: {path_or_glob}")
    runtime = get_runtime()
    result = runtime.browse_knowledge_files(path_or_glob)
    await _stream_progress(ctx, "check-circle", f"Knowledge browse complete: {result.get('count', 0)} file(s).")
    return result


@function_tool
async def resolve_entities(
    ctx: RunContextWrapper[AgentContext],
    input_text: str,
    sales_date_hint: str | None = None,
) -> dict[str, Any]:
    """Resolve provider/site/customer entities via common codes + MySQL fallback."""
    await _stream_progress(ctx, "search", "Resolving entities and code references.")
    runtime = get_runtime()
    result = runtime.resolve_entities(input_text, sales_date_hint=sales_date_hint)
    await _stream_progress(
        ctx,
        "check-circle",
        (
            "Entity resolution complete: "
            f"providers={len(result.get('providers', []))}, "
            f"sites={len(result.get('sites', []))}, "
            f"customers={len(result.get('customers', []))}."
        ),
    )
    return result


@function_tool
async def retrieve_knowledge(
    ctx: RunContextWrapper[AgentContext],
    intent: str,
    entities: str,
    question: str,
) -> dict[str, Any]:
    """Retrieve candidate tables/metadata hints from local KB index."""
    await _stream_progress(ctx, "search", "Retrieving KB context.")
    runtime = get_runtime()
    parsed_entities = _parse_json_object(entities, field_name="entities")
    result = runtime.retrieve_knowledge(intent=intent, entities=parsed_entities, question=question)
    await _stream_progress(
        ctx,
        "check-circle",
        f"Knowledge retrieval complete: tables={len(result.get('candidate_tables', []))}.",
    )
    return result


@function_tool
async def inspect_table_metadata(
    ctx: RunContextWrapper[AgentContext],
    table_name: str,
    datasource: str | None = None,
    capture_example_row: bool = True,
) -> dict[str, Any]:
    """Inspect schema/partitions and cache unknown tables as discovered metadata."""
    await _stream_progress(ctx, "search", f"Inspecting metadata for table {table_name}.")
    runtime = get_runtime()
    result = runtime.inspect_table_metadata(
        table_name=table_name,
        datasource=datasource,
        capture_example_row=capture_example_row,
    )
    await _stream_progress(
        ctx,
        "check-circle",
        f"Table inspection complete: columns={len(result.get('columns', []))}.",
    )
    return result


@function_tool
async def extract_sql_to_dataset(
    ctx: RunContextWrapper[AgentContext],
    query: str,
    datasource: str,
    run_id: str | None = None,
    metadata: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Execute read-only SQL and persist result to local dataset artifact."""
    runtime = get_runtime()
    thread_id = _thread_id(ctx)
    parsed_metadata = _parse_json_object(metadata, field_name="metadata") if metadata else {}

    await _stream_progress(ctx, "clock", f"Running SQL extract on {datasource}.")
    result = runtime.extract_sql_to_dataset(
        thread_id=thread_id,
        query=query,
        datasource=datasource,
        run_id=run_id,
        metadata=parsed_metadata,
        dataset_name=dataset_name,
    )
    await _stream_progress(
        ctx,
        "check-circle",
        f"SQL extraction complete: dataset_id={result.get('dataset_id')}, rows={result.get('row_count')}.",
    )
    return result


@function_tool
async def extract_s3_to_dataset(
    ctx: RunContextWrapper[AgentContext],
    bucket: str,
    key_or_prefix: str,
    run_id: str | None = None,
    metadata: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Fetch CSV object(s) from S3 and persist result to local dataset artifact."""
    runtime = get_runtime()
    thread_id = _thread_id(ctx)
    parsed_metadata = _parse_json_object(metadata, field_name="metadata") if metadata else {}

    await _stream_progress(ctx, "clock", f"Running S3 extraction from {bucket}.")
    result = runtime.extract_s3_to_dataset(
        thread_id=thread_id,
        bucket=bucket,
        key_or_prefix=key_or_prefix,
        run_id=run_id,
        metadata=parsed_metadata,
        dataset_name=dataset_name,
    )
    await _stream_progress(
        ctx,
        "check-circle",
        f"S3 extraction complete: dataset_id={result.get('dataset_id')}, rows={result.get('row_count')}.",
    )
    return result


@function_tool
async def run_dataframe_analysis(
    ctx: RunContextWrapper[AgentContext],
    run_id: str,
    dataset_ids: list[str],
    analysis_spec: str,
) -> dict[str, Any]:
    """Run built-in dataframe analyses on dataset artifacts."""
    runtime = get_runtime()
    thread_id = _thread_id(ctx)
    parsed_analysis_spec = _parse_json_object(analysis_spec, field_name="analysis_spec")

    await _stream_progress(ctx, "clock", "Running dataframe analysis.")
    result = runtime.run_dataframe_analysis(
        thread_id=thread_id,
        run_id=run_id,
        dataset_ids=dataset_ids,
        analysis_spec=parsed_analysis_spec,
    )
    await _stream_progress(ctx, "check-circle", f"Dataframe analysis complete: analysis_id={result.get('analysis_id')}.")
    return result


@function_tool
async def operator_run_python(
    ctx: RunContextWrapper[AgentContext],
    code: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run pandas-focused Python analysis over saved dataset artifacts."""
    runtime = get_runtime()
    thread_id = _thread_id(ctx)
    await _stream_progress(ctx, "clock", "Running custom Python operator code.")
    result = runtime.operator_run_python(thread_id=thread_id, code=code, run_id=run_id)
    await _stream_progress(
        ctx,
        "check-circle",
        f"Python operator complete: created_datasets={len(result.get('created_datasets', []))}, run_id={result.get('run_id')}.",
    )
    return result


@function_tool
async def cleanup_session_workspace(
    ctx: RunContextWrapper[AgentContext],
    thread_id: str | None = None,
    mode: str = "ephemeral_manifest",
) -> dict[str, Any]:
    """Clean local workspace artifacts while retaining compact manifests for lineage."""
    target_thread = thread_id or _thread_id(ctx)
    await _stream_progress(ctx, "search", f"Cleaning workspace for thread {target_thread}.")
    result = cleanup_thread_workspace(thread_id=target_thread, mode=mode)
    await _stream_progress(
        ctx,
        "check-circle",
        f"Workspace cleanup complete: deleted_files={result.get('deleted_files')}, manifest_retained={result.get('manifest_retained')}.",
    )
    return result


@function_tool
async def run_table_eda(
    ctx: RunContextWrapper[AgentContext],
    table_name: str,
    datasource: str | None = None,
    constraints: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run deep autonomous EDA for a table and return markdown report + lineage."""
    runtime = get_runtime()
    thread_id = _thread_id(ctx)
    parsed_constraints = _parse_json_object(constraints, field_name="constraints") if constraints else {}

    await _stream_progress(ctx, "search", f"Running deep table EDA for {table_name}.")
    result = runtime.run_table_eda(
        thread_id=thread_id,
        table_name=table_name,
        datasource=datasource,
        constraints=parsed_constraints,
        run_id=run_id,
    )
    await _stream_progress(
        ctx,
        "check-circle",
        f"EDA complete: datasets={len(result.get('datasets', []))}, run_id={result.get('run_id')}.",
    )
    return result


@function_tool
async def investigate_issue(
    ctx: RunContextWrapper[AgentContext],
    question: str,
    sales_date: str | None = None,
    constraints: str | None = None,
) -> dict[str, Any]:
    """Autonomous investigation dispatcher (no predefined intent recipes)."""
    runtime = get_runtime()
    thread_id = _thread_id(ctx)
    parsed_constraints = _parse_json_object(constraints, field_name="constraints") if constraints else None

    await _stream_progress(ctx, "search", "Starting autonomous investigation.")
    result = runtime.investigate_issue(
        thread_id=thread_id,
        question=question,
        sales_date=sales_date,
        constraints=parsed_constraints,
    )
    await _stream_progress(
        ctx,
        "check-circle",
        f"Investigation complete: strategy={result.get('strategy')}, datasets={len(result.get('datasets', []))}, run_id={result.get('run_id')}.",
    )
    return result


def investigation_tools() -> list[Any]:
    return [
        investigate_issue,
        run_table_eda,
        resolve_entities,
        retrieve_knowledge,
        browse_knowledge_files,
        inspect_table_metadata,
        extract_sql_to_dataset,
        extract_s3_to_dataset,
        run_dataframe_analysis,
        operator_run_python,
        cleanup_session_workspace,
    ]


__all__ = [
    "investigation_instructions",
    "investigation_tools",
    "cleanup_thread_workspace",
    "is_investigation_engine_enabled",
]
