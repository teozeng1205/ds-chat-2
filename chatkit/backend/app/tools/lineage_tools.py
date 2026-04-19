"""Agent-facing pipeline-lineage tool(s).

`trace_pipeline(entity)` walks the cross-repo lineage graph built by
`scripts/build_pipeline_graph.py` and returns the entity's upstream
/ downstream neighborhood — the stages that read it, the stages that
write it, the triggers that fire those stages, and the Lambda / ECS /
Step Functions that deploy them.

Example calls the agent can make:

    trace_pipeline("market_level_anomalies_v4", direction="upstream", depth=3)
    trace_pipeline("competitive-position")
    trace_pipeline("s3-atp-3victors-3vprod-use1-anomaly-datasets")
    trace_pipeline("DCO")                  # alias → "derived-common-output"

When the graph hasn't been built yet, the tool returns a clear
"graph empty — run scripts/build_pipeline_graph.py" message so the
agent can suggest the next step to the user.
"""

from __future__ import annotations

import logging
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent

from ..pipelines.graph_store import Direction, GraphStore

log = logging.getLogger(__name__)


async def _stream(ctx: RunContextWrapper[AgentContext], icon: str, text: str) -> None:  # type: ignore[type-arg]
    try:
        await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))
    except Exception:
        pass


@function_tool
async def trace_pipeline(
    ctx: RunContextWrapper[AgentContext],
    entity: str,
    direction: str = "both",
    depth: int = 3,
) -> dict[str, Any]:
    """Walk the cross-repo pipeline lineage graph from any entity.

    Use this to understand how data flows between apps. Especially
    valuable when debugging bad values in a Redshift table or S3
    output — `trace_pipeline(table, direction="upstream")` tells you
    which stages produced it, so you can spot-check each one in
    order.

    Args:
        entity: Any reference the graph might know: a Redshift table
          name ("market_level_anomalies_v4", "prod.analytics.foo"),
          an S3 bucket prefix ("s3-atp-3victors-3vprod-use1-anomaly-
          datasets"), an app / stage name ("competitive-position"),
          or a short alias ("DCO", "MLG").
        direction: "upstream" | "downstream" | "both". Upstream walks
          the chain of stages whose outputs feed this entity;
          downstream walks the consumers.
        depth: How many hops to traverse (default 3). Bounded so
          large graphs don't blow up the response.

    Returns a dict with:
        origin       — the resolved node (id, kind, name, metadata)
        upstream     — ordered list of nodes reached by walking upstream
        downstream   — ordered list of nodes reached by walking downstream
        stages       — every stage/app node in the neighborhood
        tables       — every Redshift / Glue table in the neighborhood
        s3_prefixes  — every S3 prefix in the neighborhood
        edges        — every edge traversed, each with {source, target, rel}
        summary      — human-readable 1-3 sentence chain description

    If the entity can't be resolved, returns {ok: false, error: ...}.
    If the graph hasn't been built yet, returns a {ok: false, error: ...}
    message that hints at running scripts/build_pipeline_graph.py.
    """
    try:
        await _stream(ctx, "route", f"Tracing lineage from {entity!r}.")

        store = GraphStore()
        try:
            stats = store.stats()
            if stats["total_nodes"] == 0:
                return {
                    "ok": False,
                    "error": (
                        "Pipeline graph is empty. Run "
                        "`python scripts/build_pipeline_graph.py` to build it."
                    ),
                    "error_type": "GraphEmpty",
                }

            resolved_id = store.resolve(entity)
            if resolved_id is None:
                return {
                    "ok": False,
                    "error": f"No pipeline node found for {entity!r}.",
                    "error_type": "NotFound",
                    "hint": "Try a Redshift table, S3 bucket, app name, or alias.",
                }

            dir_ = direction if direction in ("upstream", "downstream", "both") else "both"
            depth_ = max(1, min(int(depth or 3), 6))

            # Walk upstream + downstream separately so the response can
            # label each side cleanly.
            if dir_ == "both":
                up = store.neighbors(resolved_id, direction="upstream",   depth=depth_)
                dn = store.neighbors(resolved_id, direction="downstream", depth=depth_)
            elif dir_ == "upstream":
                up = store.neighbors(resolved_id, direction="upstream", depth=depth_)
                dn = {"origin": up["origin"], "nodes": [], "edges": [], "reached_by_depth": {}}
            else:
                dn = store.neighbors(resolved_id, direction="downstream", depth=depth_)
                up = {"origin": dn["origin"], "nodes": [], "edges": [], "reached_by_depth": {}}

            origin = up["origin"] or dn["origin"]
            origin_dict = _node_to_dict(origin) if origin else None

            # Build union of reached nodes / edges with side labels
            nodes_by_id: dict[str, dict] = {}
            for n in up["nodes"]:
                nodes_by_id.setdefault(n.id, _node_to_dict(n) | {"side": "upstream"})
            for n in dn["nodes"]:
                # Origin shows up in both; leave whichever side labeled it first
                nodes_by_id.setdefault(n.id, _node_to_dict(n) | {"side": "downstream"})
            if origin is not None and origin.id in nodes_by_id:
                nodes_by_id[origin.id]["side"] = "origin"

            edges_out: list[dict] = []
            seen_edge_keys: set[tuple[str, str, str]] = set()
            for e in up["edges"] + dn["edges"]:
                key = (e.source_id, e.target_id, e.rel)
                if key in seen_edge_keys:
                    continue
                seen_edge_keys.add(key)
                edges_out.append({
                    "source": e.source_id,
                    "target": e.target_id,
                    "rel": e.rel,
                    "provenance": e.source,
                })

            # Buckets by kind for quick UI rendering / agent reading
            stages = [n for n in nodes_by_id.values() if n["kind"] in ("stage", "app")]
            tables = [n for n in nodes_by_id.values() if n["kind"] in ("redshift_table", "glue_table")]
            s3_prefixes = [n for n in nodes_by_id.values() if n["kind"] == "s3_prefix"]

            upstream_ordered = [
                _node_to_dict(store.get_node(nid))
                for nids in sorted(up["reached_by_depth"].items())
                for nid in nids[1]
                if store.get_node(nid) is not None
            ]
            downstream_ordered = [
                _node_to_dict(store.get_node(nid))
                for nids in sorted(dn["reached_by_depth"].items())
                for nid in nids[1]
                if store.get_node(nid) is not None
            ]

            summary = _format_summary(
                origin=origin_dict,
                upstream=upstream_ordered,
                downstream=downstream_ordered,
                stages=stages,
                tables=tables,
            )

            await _stream(
                ctx, "check-circle",
                f"Traced {origin_dict['id'] if origin_dict else entity}: "
                f"{len(upstream_ordered)} upstream, {len(downstream_ordered)} downstream.",
            )

            return {
                "ok": True,
                "origin": origin_dict,
                "upstream": upstream_ordered,
                "downstream": downstream_ordered,
                "stages": stages,
                "tables": tables,
                "s3_prefixes": s3_prefixes,
                "edges": edges_out,
                "summary": summary,
            }
        finally:
            store.close()
    except Exception as exc:
        log.exception("trace_pipeline failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


def _node_to_dict(node: Any) -> dict:
    if node is None:
        return {}
    return {
        "id": node.id,
        "kind": node.kind,
        "name": node.name,
        "aliases": list(node.aliases or []),
        "metadata": node.metadata or {},
        "source": node.source,
    }


def _format_summary(
    *,
    origin: dict | None,
    upstream: list[dict],
    downstream: list[dict],
    stages: list[dict],
    tables: list[dict],
) -> str:
    if origin is None:
        return ""
    parts: list[str] = []
    parts.append(f"`{origin.get('name')}` (kind={origin.get('kind')})")
    if upstream:
        up_stage_names = [n["name"] for n in upstream if n.get("kind") in ("stage", "app")]
        if up_stage_names:
            parts.append("upstream stages: " + " ← ".join(up_stage_names[:6]))
    if downstream:
        dn_stage_names = [n["name"] for n in downstream if n.get("kind") in ("stage", "app")]
        if dn_stage_names:
            parts.append("downstream stages: " + " → ".join(dn_stage_names[:6]))
    table_names = [n["name"] for n in tables][:4]
    if table_names:
        parts.append("related tables: " + ", ".join(table_names))
    return "; ".join(parts)


def lineage_tools() -> list[Any]:
    return [trace_pipeline]


__all__ = ["trace_pipeline", "lineage_tools"]
