"""Six atomic tools for the DS Chat investigation agent."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import AttachmentCreateParams, ProgressUpdateEvent
from chatkit.widgets import Card

from ..attachment_store import LocalDiskAttachmentStore, default_attachment_dir
from ..investigation.runtime import cleanup_thread_workspace, get_runtime

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
log = logging.getLogger(__name__)

# ── Run-id cache: one run per thread per agent turn ──
# Entries expire after _RUN_TTL_SECONDS so a new conversation turn gets a fresh run.
_RUN_TTL_SECONDS = 120
_run_cache: dict[str, tuple[str, float]] = {}
_run_cache_lock = threading.Lock()


def _get_or_create_run_id(thread_id: str) -> str:
    """Return the current run_id for *thread_id*, creating one if needed or expired."""
    now = time.monotonic()
    with _run_cache_lock:
        entry = _run_cache.get(thread_id)
        if entry is not None:
            run_id, created_at = entry
            if now - created_at < _RUN_TTL_SECONDS:
                return run_id
        runtime = get_runtime()
        run_id = runtime.start_run(thread_id)
        _run_cache[thread_id] = (run_id, now)
        return run_id
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
                                "handler": "client",
                                "loadingBehavior": "none",
                                "payload": {"url": inline_data_url},
                            },
                        },
                        {
                            "type": "Button",
                            "label": "Download PNG",
                            "style": "secondary",
                            "onClickAction": {
                                "type": "download_url",
                                "handler": "client",
                                "loadingBehavior": "none",
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


# ── Tool 1: execute_sql ──

@function_tool
async def execute_sql(
    ctx: RunContextWrapper[AgentContext],
    query: str,
    datasource: str | None = None,
) -> dict[str, Any]:
    """Execute a read-only SQL query and save the result as a dataset.

    Args:
        query: The SQL SELECT/WITH query to execute. Must include partition filters.
        datasource: One of 'redshift_analytics', 'redshift_core', 'mysql_priceeye'. Auto-detected if omitted.

    Returns: columns, row_count, preview rows (first 20), dataset_id, partition_warnings.
    """
    try:
        runtime = get_runtime()
        thread_id = _thread_id(ctx)
        run_id = _get_or_create_run_id(thread_id)
        await _stream_progress(ctx, "clock", f"Running SQL on {datasource or 'auto-detected datasource'}.")
        result = runtime.execute_sql(thread_id=thread_id, run_id=run_id, query=query, datasource=datasource)
        await _stream_progress(
            ctx, "check-circle",
            f"SQL complete: {result.get('row_count')} rows, dataset_id={result.get('dataset_id')}.",
        )
        return result
    except Exception as exc:
        log.exception("execute_sql failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Tool 2: fetch_s3 ──

@function_tool
async def fetch_s3(
    ctx: RunContextWrapper[AgentContext],
    bucket: str,
    key_or_prefix: str,
) -> dict[str, Any]:
    """Fetch CSV/Parquet/JSONL data from S3 and save as a dataset.

    Args:
        bucket: S3 bucket name (e.g. 's3-atp-3victors-3vdev-use1-collection-anomalies').
        key_or_prefix: S3 key or prefix path (e.g. 'collection-customer/v1/2026/02/26/').

    Returns: columns, row_count, preview rows (first 20), dataset_id, s3_keys.
    """
    try:
        runtime = get_runtime()
        thread_id = _thread_id(ctx)
        run_id = _get_or_create_run_id(thread_id)
        await _stream_progress(ctx, "clock", f"Fetching S3 data from {bucket}.")
        result = runtime.fetch_s3(thread_id=thread_id, run_id=run_id, bucket=bucket, key_or_prefix=key_or_prefix)
        await _stream_progress(
            ctx, "check-circle",
            f"S3 fetch complete: {result.get('row_count')} rows, {len(result.get('s3_keys', []))} files.",
        )
        return result
    except Exception as exc:
        log.exception("fetch_s3 failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Tool 3: run_python ──

@function_tool
async def run_python(
    ctx: RunContextWrapper[AgentContext],
    code: str,
) -> dict[str, Any]:
    """Execute Python/pandas code against workspace datasets.

    Available in scope: load_dataset(id), list_datasets(), save_dataframe(df, name),
    save_plot(fig, name), pd, np, plt, sns, json, Path.

    Args:
        code: Python code to execute. Use load_dataset(dataset_id) to load saved data.

    Returns: stdout output, created_datasets, created_analyses, and any published images.
    """
    try:
        runtime = get_runtime()
        thread_id = _thread_id(ctx)
        run_id = _get_or_create_run_id(thread_id)
        await _stream_progress(ctx, "clock", "Running Python code.")
        result = runtime.run_python(thread_id=thread_id, run_id=run_id, code=code)
        result["published_images"] = await _auto_publish_images_from_result(ctx, result=result)
        await _stream_progress(
            ctx, "check-circle",
            f"Python complete: {len(result.get('created_datasets', []))} datasets, {len(result.get('published_images', []))} images.",
        )
        return result
    except Exception as exc:
        log.exception("run_python failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Tool 4: inspect_table ──

@function_tool
async def inspect_table(
    ctx: RunContextWrapper[AgentContext],
    table_name: str,
    datasource: str | None = None,
) -> dict[str, Any]:
    """Get schema, partition info, and a masked sample row for a table.

    Args:
        table_name: Fully qualified table name (e.g. 'prod.monitoring.provider_combined_audit').
        datasource: Optional datasource override. Auto-detected if omitted.

    Returns: columns, partitions, sample_row_masked, tier.
    """
    try:
        await _stream_progress(ctx, "search", f"Inspecting table {table_name}.")
        runtime = get_runtime()
        result = runtime.inspect_table(table_name=table_name, datasource=datasource)
        await _stream_progress(
            ctx, "check-circle",
            f"Table inspection complete: {len(result.get('columns', []))} columns.",
        )
        return result
    except Exception as exc:
        log.exception("inspect_table failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Tool 5: search_kb ──

@function_tool
async def search_kb(
    ctx: RunContextWrapper[AgentContext],
    query: str,
) -> dict[str, Any]:
    """Search the local knowledge base for matching tables, docs, and investigation patterns.

    Args:
        query: Natural language search query (e.g. 'market anomalies', 'site issues', 'combined audit').

    Returns: candidate_tables, table_hints (with partition info).
    """
    try:
        await _stream_progress(ctx, "search", f"Searching KB for: {query}")
        runtime = get_runtime()
        result = runtime.search_kb(query=query)
        await _stream_progress(
            ctx, "check-circle",
            f"KB search complete: {len(result.get('candidate_tables', []))} tables found.",
        )
        return result
    except Exception as exc:
        log.exception("search_kb failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Tool 6: resolve_codes ──

@function_tool
async def resolve_codes(
    ctx: RunContextWrapper[AgentContext],
    text: str,
) -> dict[str, Any]:
    """Resolve provider/site/customer codes from natural language text.

    Handles airline codes (AA, DL, B6), names (JetBlue, Delta), and pipe-separated pairs (QL2|AV).

    Args:
        text: Text containing entity references to resolve.

    Returns: {providers: [...], sites: [...], customers: [...], unknown_tokens: [...]}.
    """
    try:
        await _stream_progress(ctx, "search", "Resolving entity codes.")
        runtime = get_runtime()
        result = runtime.resolve_codes(text=text)
        await _stream_progress(
            ctx, "check-circle",
            f"Resolved: providers={len(result.get('providers', []))}, sites={len(result.get('sites', []))}, customers={len(result.get('customers', []))}.",
        )
        return result
    except Exception as exc:
        log.exception("resolve_codes failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


@function_tool
async def browse_repo_files(
    ctx: RunContextWrapper[AgentContext],
    path_or_glob: str,
) -> dict[str, Any]:
    """Browse source files and documentation under ~/git/ for code explanation.

    Args:
        path_or_glob: Path relative to ~/git/ or a glob pattern.
          Examples: 'documentations/ds-priceeye-analytics.md'
                    'ds-priceeye-analytics/src/**/*.py'
                    'documentations/*.md'  (list all docs)

    Returns: {count, files: [{path, size, content (first 8000 chars), truncated}]}
    """
    import glob as _glob
    base = Path("~/git").expanduser().resolve()
    pattern = path_or_glob.strip() or "*"
    if any(ch in pattern for ch in "*?[]"):
        matches = [Path(p) for p in _glob.glob(str(base / pattern), recursive=True)]
    else:
        target = (base / pattern).resolve()
        matches = [target] if target.exists() else []

    entries: list[dict[str, Any]] = []
    for p in matches[:20]:
        if not p.is_file():
            continue
        try:
            p.relative_to(base)
        except ValueError:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        entries.append({
            "path": str(p.relative_to(base)),
            "size": p.stat().st_size,
            "content": text[:8000],
            "truncated": len(text) > 8000,
        })

    return {"count": len(entries), "files": entries}


def investigation_tools() -> list[Any]:
    """Return the 7 atomic tools for the investigation agent."""
    return [
        execute_sql,
        fetch_s3,
        run_python,
        inspect_table,
        search_kb,
        resolve_codes,
        browse_repo_files,
    ]


__all__ = [
    "cleanup_thread_workspace",
    "investigation_tools",
]
