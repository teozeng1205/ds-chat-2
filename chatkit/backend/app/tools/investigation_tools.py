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
from ._common import (
    TIMEOUT_DB_QUERY,
    TIMEOUT_FAST,
    TIMEOUT_S3_FETCH,
    TIMEOUT_SHORT_NET,
    tool_error,
)

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

def _date_bucket() -> str:
    """Day-resolution partition for the cache key. Stale queries expire at midnight UTC."""
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


@function_tool(timeout=TIMEOUT_DB_QUERY, failure_error_function=tool_error)
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
    If the identical query was run within the last 15 minutes, a cached preview
    is returned with `cached: True` and `dataset_id: None` — call execute_sql
    again (e.g., with a slightly different LIMIT or WHERE) if you need a fresh
    per-thread dataset for Python analysis.
    """
    try:
        runtime = get_runtime()
        thread_id = _thread_id(ctx)
        run_id = _get_or_create_run_id(thread_id)

        # ── Cache lookup ────────────────────────────────────────
        # Cache only the preview-shaped payload; strip dataset_id because it
        # points into a per-thread workspace that's cleaned up each turn.
        cache_payload: dict[str, Any] | None = None
        cache_hit_age: float | None = None
        try:
            from app.investigation.query_cache import get_query_cache
            hit = get_query_cache().get(query, datasource or "_auto", extra=[_date_bucket()])
            if hit is not None:
                cache_payload = dict(hit.payload)
                cache_hit_age = hit.age_seconds
        except Exception as exc:  # noqa: BLE001 — telemetry, never crash
            log.debug("query cache read failed: %s", exc)

        if cache_payload is not None:
            await _stream_progress(
                ctx, "check-circle",
                f"SQL cached ({int(cache_hit_age or 0)}s old): {cache_payload.get('row_count')} rows.",
            )
            cache_payload.update({
                "cached": True,
                "cache_age_seconds": int(cache_hit_age or 0),
                "dataset_id": None,
                "dataset_note": (
                    "Cached preview — dataset not materialized this turn. "
                    "If you need a dataset for Python analysis, tweak the query "
                    "(e.g., adjust LIMIT) to force a fresh run."
                ),
            })
            return cache_payload

        # ── Fresh execution ─────────────────────────────────────
        await _stream_progress(ctx, "clock", f"Running SQL on {datasource or 'auto-detected datasource'}.")
        t0 = time.monotonic()
        result = runtime.execute_sql(thread_id=thread_id, run_id=run_id, query=query, datasource=datasource)
        elapsed = time.monotonic() - t0
        await _stream_progress(
            ctx, "check-circle",
            f"SQL complete: {result.get('row_count')} rows in {elapsed:.1f}s, dataset_id={result.get('dataset_id')}.",
        )

        # ── Cache write ─────────────────────────────────────────
        try:
            from app.investigation.query_cache import get_query_cache
            if isinstance(result, dict) and result.get("ok", True):
                # Don't cache dataset_id (it's per-thread-ephemeral) or errors.
                to_cache = {k: v for k, v in result.items() if k != "dataset_id"}
                get_query_cache().put(query, to_cache, datasource or "_auto", extra=[_date_bucket()])
        except Exception as exc:  # noqa: BLE001
            log.debug("query cache write failed: %s", exc)

        return result
    except Exception as exc:
        log.exception("execute_sql failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Tool 2: fetch_s3 ──

@function_tool(timeout=TIMEOUT_S3_FETCH, failure_error_function=tool_error)
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
        t0 = time.monotonic()
        result = runtime.fetch_s3(thread_id=thread_id, run_id=run_id, bucket=bucket, key_or_prefix=key_or_prefix)
        elapsed = time.monotonic() - t0
        await _stream_progress(
            ctx, "check-circle",
            f"S3 fetch complete: {result.get('row_count')} rows, {len(result.get('s3_keys', []))} files in {elapsed:.1f}s.",
        )
        return result
    except Exception as exc:
        log.exception("fetch_s3 failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Tool 3: inspect_table ──

@function_tool(timeout=TIMEOUT_DB_QUERY, failure_error_function=tool_error)
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

def _pipeline_context(
    *,
    candidate_tables: list[str] | None = None,
    semantic_hits: list[dict[str, Any]] | None = None,
    max_entries: int = 6,
) -> dict[str, Any]:
    """Attach 1-hop lineage to KB hits that resolve to graph nodes.

    Best-effort — returns {} when the graph DB hasn't been built or
    when no KB hit maps to a known node. Each entry in the returned
    dict is keyed by graph node id and has at most ~2 upstream /
    ~2 downstream neighbors plus the stage that produces the entity.
    """
    try:
        from pathlib import Path
        from ..pipelines.graph_store import GraphStore, default_graph_db_path

        db_path = default_graph_db_path()
        if not Path(db_path).exists():
            return {}

        store = GraphStore(db_path)
        try:
            if store.stats()["total_nodes"] == 0:
                return {}

            seeds: list[str] = []
            for t in candidate_tables or []:
                if isinstance(t, str) and t:
                    seeds.append(t)
            for hit in semantic_hits or []:
                sid = hit.get("id") if isinstance(hit, dict) else None
                if isinstance(sid, str) and sid:
                    seeds.append(sid)
                meta = (hit or {}).get("metadata") if isinstance(hit, dict) else None
                if isinstance(meta, dict):
                    for k in ("table_name", "code", "source"):
                        v = meta.get(k)
                        if isinstance(v, str) and v:
                            seeds.append(v)

            out: dict[str, Any] = {}
            seen: set[str] = set()
            for seed in seeds:
                if len(out) >= max_entries:
                    break
                resolved = store.resolve(seed)
                if resolved is None or resolved in seen:
                    continue
                seen.add(resolved)
                node = store.get_node(resolved)
                if node is None:
                    continue
                edges = store.get_edges(resolved, direction="both")
                entry: dict[str, Any] = {"kind": node.kind, "name": node.name}

                # Bucket edges by their semantic role relative to `resolved`:
                # - produced_by: some source writes INTO us
                # - consumed_by: some source reads FROM us
                # - reads_from:  we (as source) read a thing
                # - writes_to:   we (as source) write a thing
                produced_by: list[str] = []
                consumed_by: list[str] = []
                reads_from: list[str] = []
                writes_to: list[str] = []
                triggers: list[str] = []
                for e in edges:
                    if e.rel == "writes" and e.target_id == resolved:
                        produced_by.append(e.source_id)
                    elif e.rel == "reads" and e.target_id == resolved:
                        consumed_by.append(e.source_id)
                    elif e.rel == "reads" and e.source_id == resolved:
                        reads_from.append(e.target_id)
                    elif e.rel == "writes" and e.source_id == resolved:
                        writes_to.append(e.target_id)
                    elif e.rel == "triggers":
                        triggers.append(
                            e.source_id if e.target_id == resolved else e.target_id
                        )

                if produced_by:
                    entry["produced_by"] = produced_by[:3]
                if consumed_by:
                    entry["consumed_by"] = consumed_by[:3]
                if reads_from:
                    entry["reads_from"] = reads_from[:6]
                if writes_to:
                    entry["writes_to"] = writes_to[:6]
                if triggers:
                    entry["triggers"] = triggers[:3]
                out[resolved] = entry
            return out
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001 — graph lookup is best-effort
        log.debug("pipeline_context lookup skipped: %s", exc)
        return {}


def _semantic_hits(query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Best-effort semantic search over the pre-built embedding index.

    Returns an empty list when:
      - the index file doesn't exist yet (run
        `python -m app.investigation.knowledge.build_embeddings` to build),
      - the OpenAI embedding call fails.
    Never raises.
    """
    try:
        from pathlib import Path
        backend_root = Path(__file__).resolve().parents[2]
        index_path = backend_root / "app" / ".data" / "ds-chat-semantic.sqlite"
        if not index_path.exists():
            return []

        from openai import OpenAI
        from app.investigation.semantic_index import SemanticIndex, tokenize
        client = OpenAI()
        resp = client.embeddings.create(model="text-embedding-3-large", input=[query])
        q_vec = list(resp.data[0].embedding)

        idx = SemanticIndex(index_path)
        try:
            hits = idx.hybrid_search(q_vec, lexical_terms=tokenize(query), top_k=top_k)
        finally:
            idx.close()

        return [
            {
                "id": h.id,
                "kind": h.kind,
                "score": round(h.score, 4),
                "cosine": round(h.cosine, 4),
                "lexical": round(h.lexical, 4),
                "snippet": h.text[:600],
                "source": (h.metadata or {}).get("source"),
            }
            for h in hits
        ]
    except Exception as exc:  # noqa: BLE001 — semantic path is best-effort
        log.debug("semantic KB search skipped: %s", exc)
        return []


@function_tool(timeout=TIMEOUT_SHORT_NET, failure_error_function=tool_error)
async def search_kb(
    ctx: RunContextWrapper[AgentContext],
    query: str,
) -> dict[str, Any]:
    """Search the local knowledge base for matching tables, docs, and investigation patterns.

    Args:
        query: Natural language search query (e.g. 'market anomalies', 'site issues', 'combined audit').

    Returns: candidate_tables, table_hints (with partition info), document_hints.
    When the embedding index has been built (run
    `python -m app.investigation.knowledge.build_embeddings`), the result also
    carries `semantic_hits` — hybrid cosine+lexical ranked chunks with score,
    snippet, and source path.
    """
    try:
        await _stream_progress(ctx, "search", f"Searching KB for: {query}")
        runtime = get_runtime()
        result = runtime.search_kb(query=query)

        # Best-effort semantic overlay. Additive: never replaces the
        # lexical path, just adds a `semantic_hits` field when available.
        sem = _semantic_hits(query)
        if sem:
            result["semantic_hits"] = sem

        # Best-effort pipeline-lineage overlay. When a candidate_table or
        # semantic_hit resolves to a graph node, attach its 1-hop
        # neighborhood so the agent sees upstream producers / downstream
        # consumers without needing a separate trace_pipeline call.
        pipe_ctx = _pipeline_context(
            candidate_tables=result.get("candidate_tables") or [],
            semantic_hits=sem,
        )
        if pipe_ctx:
            result["pipeline_context"] = pipe_ctx

        await _stream_progress(
            ctx, "check-circle",
            f"KB search complete: {len(result.get('candidate_tables', []))} tables"
            + (f", {len(sem)} semantic hits" if sem else "")
            + (f", {len(pipe_ctx)} lineage nodes" if pipe_ctx else "")
            + ".",
        )
        return result
    except Exception as exc:
        log.exception("search_kb failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Tool 6: resolve_codes ──

@function_tool(timeout=TIMEOUT_FAST, failure_error_function=tool_error)
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


@function_tool(timeout=TIMEOUT_FAST, failure_error_function=tool_error)
async def publish_image(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Publish a saved image file as a Card widget with fullscreen and download buttons.

    Args:
        path: Absolute or /tmp/-relative path to the image file (PNG, JPG, SVG, etc).
        display_name: Optional human-readable label shown on the card.

    Returns: {published: true, attachment_id, image_url, path, mime_type}
    """
    try:
        await _stream_progress(ctx, "images", f"Publishing image: {path}")
        result = await _publish_image_widget(ctx, path=path, display_name=display_name)
        return result
    except Exception as exc:
        log.exception("publish_image failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


def investigation_tools_core() -> list[Any]:
    """Return the 6 core data tools for the coding agent (excludes run_python, browse_repo_files)."""
    return [
        execute_sql,
        fetch_s3,
        inspect_table,
        search_kb,
        resolve_codes,
        publish_image,
    ]


__all__ = [
    "cleanup_thread_workspace",
    "investigation_tools_core",
    "publish_image",
]
