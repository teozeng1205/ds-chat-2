"""Function tools exposed to the autonomous investigation agent."""

from __future__ import annotations

import mimetypes
import base64
import datetime
import json
import re
from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import AttachmentCreateParams, ProgressUpdateEvent
from chatkit.widgets import Card

from ..attachment_store import LocalDiskAttachmentStore, default_attachment_dir
from ..investigation.runtime import cleanup_thread_workspace, get_runtime, is_investigation_engine_enabled

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
IMAGE_PATH_PATTERN = re.compile(r"(/[^)\s]+\.(?:png|jpg|jpeg|gif|webp|svg))", re.IGNORECASE)


def _allowed_plot_roots() -> tuple[Path, ...]:
    backend_root = Path(__file__).resolve().parents[2]
    return (Path("/tmp").resolve(), (backend_root / ".work").resolve())


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


def _path_allowed_for_publish(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    for root in _allowed_plot_roots():
        if resolved == root or root in resolved.parents:
            return True
    return False


def _normalize_image_path_token(raw: str) -> str:
    token = raw.strip()
    if not token:
        return ""
    # Handle markdown link wrappers like [label](/tmp/plot.png)
    link_match = re.search(r"\((/[^)\s]+\.(?:png|jpg|jpeg|gif|webp|svg))\)", token, flags=re.IGNORECASE)
    if link_match:
        token = link_match.group(1)
    for _ in range(3):
        token = token.strip().strip("`'\"<>[](){}")
    token = re.sub(r"[,;:.]+$", "", token)
    return token


def _image_path_candidates(raw: str) -> list[str]:
    candidates: list[str] = []
    if raw:
        candidates.append(raw)
        candidates.extend(IMAGE_PATH_PATTERN.findall(raw))
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = _normalize_image_path_token(item)
        if not normalized or normalized in seen:
            continue
        if Path(normalized).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _extract_image_paths(payload: Any) -> list[str]:
    found: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {"plot_path", "image_path", "chart_path", "figure_path"} and isinstance(item, str):
                    found.extend(_image_path_candidates(item))
                _walk(item)
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)
            return
        if isinstance(value, str):
            found.extend(_image_path_candidates(value))

    _walk(payload)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        candidate = _normalize_image_path_token(item)
        if not candidate or candidate in seen:
            continue
        if Path(candidate).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


async def _publish_image_widget(
    ctx: RunContextWrapper[AgentContext],
    *,
    path: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    backend_root = Path(__file__).resolve().parents[2]
    image_path: Path | None = None
    for candidate in _image_path_candidates(path):
        raw_candidate = Path(candidate).expanduser()
        candidate_paths: list[Path] = []
        if raw_candidate.is_absolute():
            candidate_paths.append(raw_candidate.resolve())
        else:
            candidate_paths.append((backend_root / raw_candidate).resolve())
            candidate_paths.append((Path.cwd() / raw_candidate).resolve())
        for resolved in candidate_paths:
            if resolved.exists() and resolved.is_file():
                image_path = resolved
                break
        if image_path is not None:
            break
    if image_path is None:
        raise ValueError(f"Image path does not exist or is unreadable: {path}")
    if not _path_allowed_for_publish(image_path):
        raise ValueError(f"Image path is outside allowed roots (/tmp, .work): {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"File must be an image: {image_path}")

    file_bytes = image_path.read_bytes()
    if not file_bytes:
        raise ValueError(f"Image file is empty: {image_path}")

    local_attachment_store = LocalDiskAttachmentStore(default_attachment_dir())
    attachment = await local_attachment_store.create_attachment(
        AttachmentCreateParams(
            name=(display_name or image_path.name),
            size=len(file_bytes),
            mime_type=mime_type,
        ),
        context=ctx.context.request_context,
    )
    await ctx.context.store.save_attachment(attachment, context=ctx.context.request_context)
    await local_attachment_store.write_attachment_bytes(attachment.id, file_bytes)

    image_url = getattr(attachment, "preview_url", None) or (
        attachment.upload_descriptor.url if attachment.upload_descriptor else None
    )
    if not image_url:
        raise RuntimeError("Failed to build image URL for published plot.")

    label = display_name or image_path.name
    inline_data_url = f"data:{mime_type};base64,{base64.b64encode(file_bytes).decode('ascii')}"
    await ctx.context.stream_widget(
        Card(
            size="full",
            children=[
                {"type": "Title", "value": "Generated Plot"},
                {
                    "type": "Image",
                    "src": inline_data_url,
                    "alt": label,
                    "fit": "contain",
                    "frame": True,
                    "radius": "lg",
                    "width": "100%",
                    "maxHeight": "78vh",
                    "minHeight": 360,
                },
                {"type": "Caption", "value": label},
                {
                    "type": "Row",
                    "gap": "sm",
                    "children": [
                        {
                            "type": "Button",
                            "label": "Open Full Size",
                            "style": "secondary",
                            "onClickAction": {
                                "type": "open_url",
                                "payload": {"url": inline_data_url},
                            },
                        },
                        {
                            "type": "Button",
                            "label": "Download PNG",
                            "style": "secondary",
                            "onClickAction": {
                                "type": "download_url",
                                "payload": {"url": inline_data_url, "filename": image_path.name},
                            },
                        },
                    ],
                },
            ]
        ),
        copy_text=f"Image: {label}",
    )

    return {
        "published": True,
        "attachment_id": attachment.id,
        "image_url": image_url,
        "path": str(image_path),
        "mime_type": mime_type,
    }


async def _auto_publish_images_from_result(
    ctx: RunContextWrapper[AgentContext],
    *,
    result: dict[str, Any],
    max_images: int = 4,
) -> list[dict[str, Any]]:
    paths = _extract_image_paths(result)
    published: list[dict[str, Any]] = []
    for path in paths[:max_images]:
        try:
            record = await _publish_image_widget(ctx, path=path)
            published.append(record)
        except Exception:
            continue
    return published


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
        "If analysis produces plot/image paths, publish them inline with publish_plot_image.\n"
        "When a plot path is wrapped in markdown or punctuation, normalize the path and still publish it.\n"
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
    query: str,
    entities: str,
    top_k: int = 8,
) -> dict[str, Any]:
    """Retrieve candidate tables/metadata hints from local KB index."""
    await _stream_progress(ctx, "search", "Retrieving KB context.")
    runtime = get_runtime()
    parsed_entities = _parse_json_object(entities, field_name="entities")
    result = runtime.retrieve_knowledge(query=query, entities=parsed_entities, top_k=top_k)
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
    result["published_images"] = await _auto_publish_images_from_result(ctx, result=result)
    await _stream_progress(
        ctx,
        "check-circle",
        (
            "Dataframe analysis complete: "
            f"analysis_id={result.get('analysis_id')}, "
            f"images={len(result.get('published_images', []))}."
        ),
    )
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
    result["published_images"] = await _auto_publish_images_from_result(ctx, result=result)
    await _stream_progress(
        ctx,
        "check-circle",
        (
            "Python operator complete: "
            f"created_datasets={len(result.get('created_datasets', []))}, "
            f"images={len(result.get('published_images', []))}, "
            f"run_id={result.get('run_id')}."
        ),
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
async def publish_plot_image(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Publish an image file from investigation workspace into chat as an inline image widget."""
    await _stream_progress(ctx, "search", f"Publishing plot image: {path}")
    result = await _publish_image_widget(ctx, path=path, display_name=display_name)
    await _stream_progress(ctx, "check-circle", f"Published image to chat: {Path(result['path']).name}")
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
    result["published_images"] = await _auto_publish_images_from_result(ctx, result=result)
    await _stream_progress(
        ctx,
        "check-circle",
        (
            "EDA complete: "
            f"datasets={len(result.get('datasets', []))}, "
            f"images={len(result.get('published_images', []))}, "
            f"run_id={result.get('run_id')}."
        ),
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
    result["published_images"] = await _auto_publish_images_from_result(ctx, result=result)
    await _stream_progress(
        ctx,
        "check-circle",
        (
            "Investigation complete: "
            f"strategy={result.get('strategy')}, "
            f"datasets={len(result.get('datasets', []))}, "
            f"images={len(result.get('published_images', []))}, "
            f"run_id={result.get('run_id')}."
        ),
    )
    return result


def investigation_tools() -> list[Any]:
    return [
        investigate_issue,
        run_table_eda,
        publish_plot_image,
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
