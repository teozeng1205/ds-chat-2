"""@function_tool wrappers for Glue catalog + QuickSight.

These unlock two capabilities:
  - Live Glue lookups (glue_get_table, glue_get_partitions) — source of
    truth for schema + partitions. Complements the existing lexical KB.
  - QuickSight dashboards (quicksight_list_dashboards,
    quicksight_get_embed_url) — the agent can list the 25+ existing
    per-airline monitoring dashboards and return an embed URL the UI
    renders as a dashboard artifact.
"""

from __future__ import annotations

import logging
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent

from ..investigation.glue_catalog import get_default_catalog
from ..ops import quicksight_client as qs

log = logging.getLogger(__name__)


async def _stream(ctx: RunContextWrapper[AgentContext], icon: str, text: str) -> None:  # type: ignore[type-arg]
    try:
        await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))
    except Exception:
        pass


def _err(exc: Exception) -> dict[str, Any]:
    log.exception("catalog tool failed: %s", exc)
    return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── Glue tools ──


@function_tool
async def glue_get_table(
    ctx: RunContextWrapper[AgentContext],
    database: str,
    name: str,
) -> dict[str, Any]:
    """Return Glue's authoritative metadata for a single table.

    Includes columns (name/type/comment), partition keys, S3 location,
    table type, parameters, and created/updated timestamps.
    """
    try:
        await _stream(ctx, "clock", f"Looking up Glue table {database}.{name}.")
        catalog = get_default_catalog()
        t = catalog.get_table(database, name)
        if t is None:
            await _stream(ctx, "circle-x", "Table not found.")
            return {"ok": False, "error": "table not found", "error_type": "NotFound"}
        await _stream(
            ctx,
            "check-circle",
            f"{t.qualified}: {len(t.columns)} cols, {len(t.partition_keys)} partition keys.",
        )
        return {
            "ok": True,
            "database": t.database,
            "name": t.name,
            "qualified": t.qualified,
            "table_type": t.table_type,
            "location": t.location,
            "owner": t.owner,
            "columns": [{"name": c.name, "type": c.type, "comment": c.comment} for c in t.columns],
            "partition_keys": [
                {"name": c.name, "type": c.type, "comment": c.comment} for c in t.partition_keys
            ],
            "created": t.created,
            "updated": t.updated,
            "parameters": t.parameters,
        }
    except Exception as exc:
        return _err(exc)


@function_tool
async def glue_get_partitions(
    ctx: RunContextWrapper[AgentContext],
    database: str,
    name: str,
    expression: str | None = None,
    max_results: int = 200,
) -> dict[str, Any]:
    """List partitions for a Glue table (optionally filtered by expression).

    Example expression: `sales_date > '20260101'`. Returns up to
    max_results partitions; each carries values, location, and timestamps.
    """
    try:
        await _stream(ctx, "clock", f"Fetching partitions for {database}.{name}.")
        catalog = get_default_catalog()
        parts = catalog.get_partitions(database, name, expression=expression, max_results=max_results)
        await _stream(ctx, "check-circle", f"{len(parts)} partitions.")
        return {
            "ok": True,
            "database": database,
            "name": name,
            "expression": expression,
            "partitions": [
                {
                    "values": list(p.values),
                    "location": p.location,
                    "created": p.created,
                    "updated": p.updated,
                }
                for p in parts
            ],
        }
    except Exception as exc:
        return _err(exc)


# ── QuickSight tools ──


@function_tool
async def quicksight_list_dashboards(
    ctx: RunContextWrapper[AgentContext],
    name_substring: str | None = None,
    max_results: int = 100,
) -> dict[str, Any]:
    """List available QuickSight dashboards in this account.

    Optionally filter by a case-insensitive name substring (e.g. 'B6').
    """
    try:
        await _stream(ctx, "clock", "Listing QuickSight dashboards.")
        dashboards = qs.list_dashboards(name_substring=name_substring, max_results=max_results)
        await _stream(ctx, "check-circle", f"{len(dashboards)} dashboards.")
        return {"ok": True, "dashboards": dashboards}
    except Exception as exc:
        return _err(exc)


@function_tool
async def quicksight_get_embed_url(
    ctx: RunContextWrapper[AgentContext],
    dashboard_id: str,
    session_lifetime_minutes: int = 60,
    allowed_domain: str | None = None,
) -> dict[str, Any]:
    """Generate a short-lived embed URL for a QuickSight dashboard.

    Requires QuickSight Enterprise + IAM permission to call
    GenerateEmbedUrlForAnonymousUser. The returned URL is signed and
    expires after `session_lifetime_minutes` (default 60, min 15,
    max 600). The frontend renders the URL as a dashboard Artifact.
    """
    try:
        await _stream(ctx, "clock", f"Generating embed URL for dashboard {dashboard_id}.")
        result = qs.generate_anonymous_embed_url(
            dashboard_id,
            session_lifetime_minutes=session_lifetime_minutes,
            allowed_domain=allowed_domain,
        )
        if not result.get("ok"):
            return result
        await _stream(ctx, "check-circle", "Embed URL ready.")
        return result
    except Exception as exc:
        return _err(exc)


def catalog_tools() -> list[Any]:
    """Return all catalog + viz tools."""
    return [
        glue_get_table,
        glue_get_partitions,
        quicksight_list_dashboards,
        quicksight_get_embed_url,
    ]


__all__ = [
    "glue_get_table",
    "glue_get_partitions",
    "quicksight_list_dashboards",
    "quicksight_get_embed_url",
    "catalog_tools",
]
