"""Task-first KB V2 retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ingest import build_kb
from .models import SearchResult
from .store import KnowledgeStore


def default_kb_db_path() -> Path:
    backend_root = Path(__file__).resolve().parents[3]
    data_dir = backend_root / "app" / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "ds-chat-kb-v2.sqlite"


class KnowledgeRetriever:
    def __init__(self, db_path: Path | None = None) -> None:
        self.store = KnowledgeStore(db_path or default_kb_db_path())

    def ensure_ready(self, *, force: bool = False) -> dict[str, Any]:
        return build_kb(self.store, force=force)

    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        ready = self.ensure_ready(force=False)
        task_hits = self.store.match_tasks(query, top_k=3)
        task = task_hits[0] if task_hits else None
        chunk_hits = self.store.search_chunks(query, top_k=top_k)
        preferred_names = list((task.get("metadata") or {}).get("preferred_tables") or []) if task else []
        q_lower = query.lower()
        if "monitoring" in q_lower and ("schema" in q_lower or "tables" in q_lower or "collection" in q_lower):
            preferred_names.extend(["prod.monitoring.provider_combined_audit", "prod.monitoring.combined_audit"])
        if "market" in q_lower and "anomal" in q_lower:
            preferred_names.extend(["prod.analytics.market_level_anomalies_v4", "prod.analytics.market_level_anomalies_v3"])
        preferred_items = self.store.items_by_names(preferred_names)

        seen_items: set[str] = set()
        items: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        for hit in chunk_hits:
            item = dict(hit["item"])
            chunk = dict(hit["chunk"])
            item["score"] = hit["score"]
            item["matched_chunk_id"] = chunk["id"]
            item["matched_text"] = chunk["text"][:900]
            if item["id"] not in seen_items:
                seen_items.add(item["id"])
                items.append(item)
                if item["type"] == "table":
                    tables.append(_table_payload(item))
            citations.append(
                {
                    "source": chunk.get("citation") or chunk.get("source_path") or item.get("source_path") or item["id"],
                    "item_id": item["id"],
                    "chunk_id": chunk["id"],
                    "title": item.get("title"),
                    "excerpt": chunk.get("text", "")[:500],
                }
            )

        for item in preferred_items:
            if item["id"] in seen_items:
                continue
            seen_items.add(item["id"])
            items.append(item)
            if item["type"] == "table":
                tables.append(_table_payload(item))

        lineage = self.store.edges_for_items(list(seen_items), limit=80)
        tool_plan = _tool_plan(task, items, lineage)
        confidence = _confidence(task, items, citations, lineage)
        result = SearchResult(
            query=query,
            task=task,
            items=items[:top_k],
            tables=tables[:8],
            lineage=lineage,
            tool_plan=tool_plan,
            citations=_dedupe_citations(citations)[:8],
            confidence=confidence,
            retrieval_trace={
                "kb_ready": ready,
                "task_candidates": task_hits,
                "chunk_hits": len(chunk_hits),
                "item_hits": len(items),
                "lineage_edges": len(lineage),
                "contract": "kb_v2",
            },
        )
        return result

    def close(self) -> None:
        self.store.close()


def _table_payload(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata") or {}
    return {
        "id": item["id"],
        "name": item["name"],
        "datasource": meta.get("datasource"),
        "tier": meta.get("tier"),
        "partitions": meta.get("partitions") or [],
        "columns": meta.get("columns") or [],
        "sample_columns": meta.get("sample_columns") or [],
        "max_sales_date": meta.get("max_sales_date"),
        "s3_location": meta.get("s3_location"),
        "git_repo": meta.get("git_repo"),
        "git_path": meta.get("git_path"),
        "score": item.get("score"),
        "summary": item.get("summary"),
    }


def _tool_plan(task: dict[str, Any] | None, items: list[dict[str, Any]], lineage: list[dict[str, Any]]) -> list[str]:
    if task and task.get("tool_plan"):
        plan = list(task["tool_plan"])
    else:
        plan = ["search_kb"]
    if task and (task.get("metadata") or {}).get("bounded"):
        return _dedupe(plan)
    item_types = {item.get("type") for item in items}
    if "table" in item_types and "inspect_table" not in plan:
        plan.append("inspect_table")
    if "table" in item_types and "execute_sql" not in plan and task and not (task.get("metadata") or {}).get("bounded"):
        plan.append("execute_sql")
    if lineage and "trace_pipeline" not in plan:
        plan.append("trace_pipeline")
    if "code" in item_types and "read_file" not in plan:
        plan.append("read_file")
    if "s3_prefix" in item_types and "list_s3" not in plan:
        plan.append("list_s3")
    return _dedupe(plan)


def _confidence(task: dict[str, Any] | None, items: list[dict[str, Any]], citations: list[dict[str, Any]], lineage: list[dict[str, Any]]) -> float:
    score = 0.2
    if task:
        score += 0.25
    if items:
        score += min(0.25, len(items) * 0.04)
    if citations:
        score += 0.15
    if lineage:
        score += 0.15
    return round(min(score, 0.95), 3)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for citation in citations:
        key = f"{citation.get('source')}|{citation.get('chunk_id')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
    return out


__all__ = ["KnowledgeRetriever", "default_kb_db_path"]
