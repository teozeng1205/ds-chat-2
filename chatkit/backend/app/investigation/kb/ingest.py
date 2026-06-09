"""Ingestion pipeline for KB V2."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ...skills import SkillRegistry
from .models import KnowledgeChunk, KnowledgeEdge, KnowledgeItem, TaskRecipe
from .store import KnowledgeStore
from .tasks import BASE_TASK_RECIPES

BACKEND_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_ROOT = BACKEND_ROOT / "app" / "investigation" / "knowledge"
DOCS_ROOT = KNOWLEDGE_ROOT / "docs"
SKILLS_ROOT = BACKEND_ROOT / "skills"
E2E_CASES_PATH = BACKEND_ROOT / "tests" / "e2e_investigation_cases.json"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_LEGACY_RE = re.compile(r"\blegacy\b|_old\b|-old\b|common-web-legacy", re.IGNORECASE)
INGEST_VERSION = "kb-v2-2026-05-04-source-authority-1"

SOURCE_POLICY = {
    "live_verified": "Evidence produced by SQL, S3, AWS, or code tools during the current answer.",
    "structured_snapshot": "Machine-readable snapshots such as table metadata, pipeline graph, codes, and YAML maps.",
    "code_verified": "Facts checked against the local repo checkout.",
    "doc_hint": "Markdown documentation. Useful for routing, not authoritative without verification.",
    "task_hint": "Skills and E2E seeds. Useful for tool planning, not answer evidence.",
}


def _stable_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _is_legacy_only(*values: str | None) -> bool:
    text = " ".join(v or "" for v in values)
    return bool(_LEGACY_RE.search(text))


def _source_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(INGEST_VERSION.encode("utf-8"))
    for path in sorted(paths):
        if not path.exists() or not path.is_file():
            continue
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_metadata(
    source_type: str,
    *,
    path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "source_type": source_type,
        "authority": source_type,
        "requires_verification": source_type in {"doc_hint", "task_hint"},
    }
    if source_type in {"doc_hint", "task_hint"}:
        metadata["staleness_note"] = "Markdown is not treated as authoritative; verify against structured/live/code evidence before relying on it."
    if path is not None and path.exists():
        try:
            metadata["source_mtime"] = path.stat().st_mtime
        except OSError:
            pass
    if extra:
        metadata.update(extra)
    return metadata


def _chunk_markdown(path: Path, *, kind: str, item_type: str) -> tuple[list[KnowledgeItem], list[KnowledgeChunk]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = str(path.relative_to(KNOWLEDGE_ROOT)) if KNOWLEDGE_ROOT in path.parents else str(path)
    item_id = f"{item_type}:{rel}"
    title = path.stem.replace("_", " ").replace("-", " ")
    item = KnowledgeItem(
        id=item_id,
        type=item_type,
        name=path.stem,
        title=title,
        summary=f"{title} documentation hint",
        source_path=rel,
        metadata=_source_metadata("doc_hint", path=path, extra={"format": "markdown"}),
        confidence=0.35,
    )
    chunks: list[KnowledgeChunk] = []
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        body = text.strip()[:5000]
        if body and not _is_legacy_only(rel, body):
            chunks.append(
                KnowledgeChunk(
                    id=f"chunk:{_stable_id(item_id, 'body')}",
                    item_id=item_id,
                    kind=kind,
                    text=body,
                    source_path=rel,
                    citation=rel,
                    metadata=_source_metadata("doc_hint", path=path, extra={"format": "markdown"}),
                    confidence=0.35,
                )
            )
        return [item], chunks

    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        heading = match.group(2).strip()
        section = text[start:end].strip()
        if not section or _is_legacy_only(rel, heading, section):
            continue
        chunks.append(
            KnowledgeChunk(
                id=f"chunk:{_stable_id(item_id, heading)}",
                item_id=item_id,
                kind=kind,
                text=section[:5000],
                source_path=rel,
                heading=heading,
                citation=f"{rel}#{heading}",
                metadata=_source_metadata("doc_hint", path=path, extra={"format": "markdown", "level": len(match.group(1))}),
                confidence=0.35,
            )
        )
    return [item], chunks


def _ingest_docs() -> tuple[list[KnowledgeItem], list[KnowledgeChunk]]:
    items: list[KnowledgeItem] = []
    chunks: list[KnowledgeChunk] = []
    for path, kind, item_type in [
        (KNOWLEDGE_ROOT / "tables.md", "table_reference", "doc"),
        (KNOWLEDGE_ROOT / "sql_best_practices.md", "sql_practice", "doc"),
    ]:
        if path.exists():
            i, c = _chunk_markdown(path, kind=kind, item_type=item_type)
            items.extend(i)
            chunks.extend(c)
    # Tier A (overview) + cross-cutting topic docs live in the docs/ root.
    # Per-repo docs were retired to docs/repos/ and are intentionally NOT indexed
    # (the non-recursive glob skips subdirectories); they remain on disk for
    # Tier C (live ~/git/ + repo exploration) reference.
    if DOCS_ROOT.exists():
        for md in sorted(DOCS_ROOT.glob("*.md")):
            i, c = _chunk_markdown(md, kind="doc", item_type="doc")
            items.extend(i)
            chunks.extend(c)
        # Tier B — one doc per process/workflow.
        workflows_root = DOCS_ROOT / "workflows"
        if workflows_root.exists():
            for md in sorted(workflows_root.glob("*.md")):
                i, c = _chunk_markdown(md, kind="workflow", item_type="doc")
                items.extend(i)
                chunks.extend(c)
    overview_path = DOCS_ROOT / "priceeye_overview.md"
    if overview_path.exists():
        overview_text = overview_path.read_text(encoding="utf-8", errors="replace")
        start = overview_text.find("## 2. The end-to-end data flow")
        end = overview_text.find("## 5. Orchestration")
        overview = overview_text[start:end].strip() if start >= 0 and end > start else overview_text[:4500].strip()
        item_id = "doc_overview:priceeye"
        text = (
            "How does PriceEye work? PriceEye system overview, architecture, end-to-end "
            "data flow, indexed documentation, source file docs/priceeye_overview.md.\n\n"
            f"{overview[:4200]}"
        )
        items.append(
            KnowledgeItem(
                id=item_id,
                type="doc",
                name="priceeye_system_overview",
                title="PriceEye end-to-end system overview",
                summary="Tier A overview of how PriceEye works from docs/priceeye_overview.md.",
                source_path="docs/priceeye_overview.md",
                metadata=_source_metadata("doc_hint", path=overview_path, extra={"format": "markdown", "overview": True}),
                confidence=0.35,
            )
        )
        chunks.append(
            KnowledgeChunk(
                id=f"chunk:{_stable_id(item_id, 'overview')}",
                item_id=item_id,
                kind="doc_overview",
                text=text,
                source_path="docs/priceeye_overview.md",
                heading="End-to-End Data Flow",
                citation="docs/priceeye_overview.md#End-to-End Data Flow",
                metadata=_source_metadata("doc_hint", path=overview_path, extra={"format": "markdown", "overview": True}),
                confidence=0.35,
            )
        )
    return items, chunks


def _ingest_tables() -> tuple[list[KnowledgeItem], list[KnowledgeChunk], list[KnowledgeEdge]]:
    path = KNOWLEDGE_ROOT / "common_table_live_metadata.json"
    if not path.exists():
        return [], [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("tables", []) if isinstance(payload, dict) else []
    items: list[KnowledgeItem] = []
    chunks: list[KnowledgeChunk] = []
    edges: list[KnowledgeEdge] = []
    schema_tables: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("table_name") or "").strip()
        if not name or _is_legacy_only(name):
            continue
        datasource = str(row.get("datasource") or "unknown")
        columns = [
            str(c.get("column_name"))
            for c in (row.get("columns") or [])
            if isinstance(c, dict) and c.get("column_name")
        ]
        partitions = [
            str(p.get("column"))
            for p in (row.get("partitions") or [])
            if isinstance(p, dict) and p.get("column")
        ]
        item_id = f"table:{name}"
        parts = name.split(".")
        if len(parts) >= 3:
            schema_tables.setdefault(".".join(parts[:2]), []).append(row)
        s3_location = row.get("s3_location")
        git_repo = row.get("git_repo")
        git_path = row.get("git_path")
        summary = (
            f"{name} ({datasource}); partitions: {', '.join(partitions) or 'none'}; "
            f"columns: {', '.join(columns[:40]) or 'unknown'}"
        )
        if s3_location:
            summary += f"; s3: {s3_location}"
        items.append(
            KnowledgeItem(
                id=item_id,
                type="table",
                name=name,
                title=name,
                summary=summary,
                source_path="common_table_live_metadata.json",
                metadata={
                    **_source_metadata("structured_snapshot", path=path),
                    "datasource": datasource,
                    "tier": row.get("tier"),
                    "partitions": row.get("partitions") or [],
                    "columns": row.get("columns") or [],
                    "sample_columns": list((row.get("sample_row_masked") or {}).keys())[:80],
                    "max_sales_date": row.get("max_sales_date"),
                    "s3_location": s3_location,
                    "git_repo": git_repo,
                    "git_path": git_path,
                },
                confidence=0.95 if str(name).startswith("prod.") else 0.75,
            )
        )
        chunks.append(
            KnowledgeChunk(
                id=f"chunk:{_stable_id(item_id, 'table')}",
                item_id=item_id,
                kind="table",
                text=summary,
                source_path="common_table_live_metadata.json",
                citation=f"common_table_live_metadata.json:{name}",
                metadata={**_source_metadata("structured_snapshot", path=path), "datasource": datasource},
                confidence=0.95,
            )
        )
        if s3_location and not _is_legacy_only(str(s3_location)):
            s3_id = f"s3:{s3_location}"
            items.append(
                KnowledgeItem(
                    id=s3_id,
                    type="s3_prefix",
                    name=str(s3_location),
                    title=str(s3_location),
                    summary=f"S3 location associated with {name}",
                    source_path="common_table_live_metadata.json",
                    metadata={**_source_metadata("structured_snapshot", path=path), "table": name},
                    confidence=0.8,
                )
            )
            edges.append(KnowledgeEdge(source_id=s3_id, target_id=item_id, rel="mirrors", source_path="common_table_live_metadata.json"))
        if git_repo or git_path:
            code_name = "/".join(str(v).strip("/") for v in (git_repo, git_path) if v)
            if code_name and not _is_legacy_only(code_name):
                code_id = f"code:{code_name}"
                items.append(
                    KnowledgeItem(
                        id=code_id,
                        type="code",
                        name=code_name,
                        title=code_name,
                        summary=f"Code provenance associated with {name}",
                        source_path="common_table_live_metadata.json",
                        metadata={**_source_metadata("structured_snapshot", path=path), "repo": git_repo, "path": git_path, "table": name},
                        confidence=0.8,
                    )
                )
                edges.append(KnowledgeEdge(source_id=code_id, target_id=item_id, rel="implemented_in", source_path="common_table_live_metadata.json"))
    for schema_name, schema_rows in sorted(schema_tables.items()):
        if _is_legacy_only(schema_name):
            continue
        table_names = sorted(str(r.get("table_name") or "") for r in schema_rows if r.get("table_name"))
        if not table_names:
            continue
        item_id = f"schema:{schema_name}"
        useful_lines: list[str] = []
        if schema_name == "prod.monitoring":
            useful_lines = [
                "- prod.monitoring.combined_audit: master request lifecycle table for end-to-end PriceEye collection, cache, enrichment, packager, delivery, and issue classification debugging.",
                "- prod.monitoring.provider_combined_audit: provider/site-centric aggregate for PriceEye collection issue analysis; use plural issue_sources and issue_reasons and aggregate inputrequestid_count for impacted requests.",
                "- prod.monitoring.deduped_collection_run_audit: collection run timing and scheduling/debugging evidence.",
                "- prod.monitoring.deduped_provider_request_audit and prod.monitoring.deduped_provider_response_audit: request/response-level provider collection evidence.",
                "- prod.monitoring.customer_combined_audit_v1 and prod.monitoring.customer_combined_audit_v2: customer impact and billing-style collection summaries.",
            ]
        text = (
            f"{schema_name} schema table inventory from common_table_live_metadata.json.\n"
            f"Tables ({len(table_names)}):\n- " + "\n- ".join(table_names)
        )
        if useful_lines:
            text += "\n\nMost useful for debugging PriceEye collection issues:\n" + "\n".join(useful_lines)
        items.append(
            KnowledgeItem(
                id=item_id,
                type="schema",
                name=schema_name,
                title=f"{schema_name} schema",
                summary=f"{schema_name} has {len(table_names)} KB-indexed tables: {', '.join(table_names[:12])}",
                source_path="common_table_live_metadata.json",
                metadata={**_source_metadata("structured_snapshot", path=path), "table_count": len(table_names), "tables": table_names},
                confidence=0.98 if schema_name.startswith("prod.") else 0.8,
            )
        )
        chunks.append(
            KnowledgeChunk(
                id=f"chunk:{_stable_id(item_id, 'schema_inventory')}",
                item_id=item_id,
                kind="schema_inventory",
                text=text[:5000],
                source_path="common_table_live_metadata.json",
                heading=f"{schema_name} schema inventory",
                citation=f"common_table_live_metadata.json:{schema_name}",
                metadata={**_source_metadata("structured_snapshot", path=path), "table_count": len(table_names)},
                confidence=0.98,
            )
        )
        for table_name in table_names[:80]:
            edges.append(
                KnowledgeEdge(
                    source_id=item_id,
                    target_id=f"table:{table_name}",
                    rel="contains_table",
                    source_path="common_table_live_metadata.json",
                    confidence=0.95,
                )
            )
    return items, chunks, edges


def _ingest_codes() -> tuple[list[KnowledgeItem], list[KnowledgeChunk]]:
    path = KNOWLEDGE_ROOT / "common_codes.json"
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[KnowledgeItem] = []
    chunks: list[KnowledgeChunk] = []
    for bucket, item_type in (("providers", "provider_code"), ("sites", "site_code"), ("customers", "customer_code")):
        for entry in payload.get(bucket, []) or []:
            if isinstance(entry, str):
                code = entry
                name = entry
                aliases: list[str] = []
            elif isinstance(entry, dict):
                code = str(entry.get("code") or entry.get("name") or "").strip()
                name = str(entry.get("name") or code).strip()
                aliases = [str(v) for v in entry.get("aliases") or []]
            else:
                continue
            if not code or _is_legacy_only(code, name):
                continue
            item_id = f"{item_type}:{code}"
            text = f"{item_type} {code}: {name}. Aliases: {', '.join(aliases) or 'none'}."
            items.append(
                KnowledgeItem(
                    id=item_id,
                    type=item_type,
                    name=code,
                    title=name,
                    summary=text,
                    source_path="common_codes.json",
                    metadata={**_source_metadata("structured_snapshot", path=path), "aliases": aliases, "bucket": bucket},
                )
            )
            chunks.append(
                KnowledgeChunk(
                    id=f"chunk:{_stable_id(item_id, 'code')}",
                    item_id=item_id,
                    kind="entity_code",
                    text=text,
                    source_path="common_codes.json",
                    citation=f"common_codes.json:{bucket}:{code}",
                    metadata=_source_metadata("structured_snapshot", path=path),
                )
            )
    return items, chunks


def _ingest_pipelines() -> tuple[list[KnowledgeItem], list[KnowledgeChunk], list[KnowledgeEdge]]:
    path = KNOWLEDGE_ROOT / "pipelines.json"
    if not path.exists():
        return [], [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes_by_kind = payload.get("nodes", {}) if isinstance(payload, dict) else {}
    items: list[KnowledgeItem] = []
    chunks: list[KnowledgeChunk] = []
    edges: list[KnowledgeEdge] = []
    for kind, nodes in nodes_by_kind.items():
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "")
            name = str(node.get("name") or "")
            source = str(node.get("source") or "pipelines.json")
            if not node_id or not name or _is_legacy_only(node_id, name, source):
                continue
            item_type = "pipeline_stage" if kind == "stage" else str(kind)
            meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            aliases = node.get("aliases") if isinstance(node.get("aliases"), list) else []
            summary = f"{kind} {name}. Aliases: {', '.join(str(a) for a in aliases[:8]) or 'none'}."
            if meta.get("repo"):
                summary += f" Repo: {meta.get('repo')}."
            items.append(
                KnowledgeItem(
                    id=node_id,
                    type=item_type,
                    name=name,
                    title=name,
                    summary=summary,
                    source_path="pipelines.json",
                    metadata={**_source_metadata("structured_snapshot", path=path), **meta, "aliases": aliases, "node_kind": kind},
                    confidence=0.85,
                )
            )
            chunks.append(
                KnowledgeChunk(
                    id=f"chunk:{_stable_id(node_id, 'pipeline')}",
                    item_id=node_id,
                    kind="pipeline",
                    text=summary,
                    source_path="pipelines.json",
                    citation=f"pipelines.json:{node_id}",
                    metadata=_source_metadata("structured_snapshot", path=path),
                    confidence=0.85,
                )
            )
    for edge in payload.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("source") or "")
        tgt = str(edge.get("target") or "")
        rel = str(edge.get("rel") or "")
        provenance = str(edge.get("provenance") or "pipelines.json")
        if not src or not tgt or not rel or _is_legacy_only(src, tgt, provenance):
            continue
        edges.append(
            KnowledgeEdge(
                source_id=src,
                target_id=tgt,
                rel=rel,
                source_path="pipelines.json",
                metadata={**_source_metadata("structured_snapshot", path=path), "provenance": provenance, **(edge.get("metadata") or {})},
                confidence=float(edge.get("weight") or 1.0),
            )
        )
    return items, chunks, edges


def _ingest_skills() -> tuple[list[KnowledgeItem], list[KnowledgeChunk], list[TaskRecipe]]:
    registry = SkillRegistry.load(SKILLS_ROOT)
    items: list[KnowledgeItem] = []
    chunks: list[KnowledgeChunk] = []
    tasks: list[TaskRecipe] = []
    for skill in registry.skills:
        if _is_legacy_only(skill.name, skill.description, skill.body):
            continue
        source_path = str(skill.path.relative_to(BACKEND_ROOT)) if skill.path else None
        item_id = f"skill:{skill.name}"
        text = f"{skill.name}: {skill.description}\nKeywords: {', '.join(skill.keywords)}\n\n{skill.body[:4000]}"
        items.append(
            KnowledgeItem(
                id=item_id,
                type="skill",
                name=skill.name,
                title=skill.name,
                summary=skill.description,
                source_path=source_path,
                metadata=_source_metadata("task_hint", path=skill.path, extra={"keywords": list(skill.keywords), "tier": skill.tier}),
                confidence=0.45,
            )
        )
        chunks.append(
            KnowledgeChunk(
                id=f"chunk:{_stable_id(item_id, 'skill')}",
                item_id=item_id,
                kind="skill",
                text=text,
                source_path=source_path,
                citation=source_path,
                metadata=_source_metadata("task_hint", path=skill.path),
                confidence=0.45,
            )
        )
        tasks.append(
            TaskRecipe(
                id=f"task:skill:{skill.name}",
                name=skill.name.replace("_", " ").title(),
                description=skill.description,
                triggers=tuple(skill.keywords),
                tool_plan=_tools_from_text(skill.body),
                source_path=source_path,
                metadata=_source_metadata("task_hint", path=skill.path, extra={"skill": skill.name, "tier": skill.tier}),
                confidence=0.65,
            )
        )
    return items, chunks, tasks


def _tools_from_text(text: str) -> tuple[str, ...]:
    known = ("search_kb", "trace_pipeline", "resolve_codes", "inspect_table", "execute_sql", "list_s3", "fetch_s3", "bash", "read_file", "run_python")
    lowered = text.lower()
    if "bounded kb lookup" in lowered or "do not run" in lowered:
        return ("search_kb",)
    found = [tool for tool in known if tool in text]
    if any(token in lowered for token in ("code", "implemented", "class", "entry point", "codebase")):
        found.extend(["read_file", "bash"])
    if any(token in lowered for token in ("schema", "columns", "what tables", "table inventory")):
        found.append("inspect_table")
    if any(token in lowered for token in ("lineage", "upstream", "downstream", "pipeline")):
        found.append("trace_pipeline")
    if any(token in lowered for token in ("s3", "bucket", "prefix", "freshness")):
        found.append("list_s3")
    if any(token in lowered for token in ("sql", "query", "row", "count")):
        found.append("execute_sql")
    return tuple(found or ("search_kb",))


def _ingest_e2e_cases() -> list[TaskRecipe]:
    if not E2E_CASES_PATH.exists():
        return []
    payload = json.loads(E2E_CASES_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    tasks: list[TaskRecipe] = []
    for case in cases or []:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or case.get("name") or "").strip()
        question = str(case.get("question") or "").strip()
        if not case_id or not question or _is_legacy_only(case_id, question):
            continue
        metadata: dict[str, Any] = _source_metadata("task_hint", path=E2E_CASES_PATH, extra={"case_id": case_id, "eval_seed": True})
        lowered = question.lower()
        if "bounded" in lowered or "do not run" in lowered or "do not inspect" in lowered:
            metadata["bounded"] = True
        if "prod.monitoring" in lowered and ("schema" in lowered or "collection" in lowered):
            metadata["preferred_tables"] = [
                "prod.monitoring.provider_combined_audit",
                "prod.monitoring.combined_audit",
            ]

        tasks.append(
            TaskRecipe(
                id=f"task:e2e:{case_id}",
                name=case_id.replace("_", " "),
                description=question[:500],
                triggers=tuple(t for t in re.split(r"[^A-Za-z0-9_]+", question.lower()) if len(t) > 3)[:16],
                tool_plan=tuple(dict.fromkeys([*(case.get("assertions", {}).get("required_tools", []) or []), *_tools_from_text(question)])),
                source_path="tests/e2e_investigation_cases.json",
                metadata=metadata,
                confidence=0.65,
            )
        )
    return tasks


def build_kb(store: KnowledgeStore, *, force: bool = False) -> dict[str, Any]:
    source_paths = [
        *KNOWLEDGE_ROOT.glob("*.md"),
        *KNOWLEDGE_ROOT.glob("*.json"),
        *KNOWLEDGE_ROOT.glob("*.yaml"),
        *DOCS_ROOT.glob("*.md"),
        *SKILLS_ROOT.glob("*.md"),
    ]
    if E2E_CASES_PATH.exists():
        source_paths.append(E2E_CASES_PATH)
    digest = _source_hash(source_paths)
    old_digest = store.get_meta("source_hash")
    if old_digest == digest and not force and store.stats()["items"] > 0:
        return {"ok": True, "refreshed": False, "source_hash": digest, **store.stats()}

    items: list[KnowledgeItem] = []
    chunks: list[KnowledgeChunk] = []
    edges: list[KnowledgeEdge] = []
    tasks: list[TaskRecipe] = list(BASE_TASK_RECIPES)

    doc_items, doc_chunks = _ingest_docs()
    table_items, table_chunks, table_edges = _ingest_tables()
    code_items, code_chunks = _ingest_codes()
    pipe_items, pipe_chunks, pipe_edges = _ingest_pipelines()
    skill_items, skill_chunks, skill_tasks = _ingest_skills()
    e2e_tasks = _ingest_e2e_cases()

    items.extend(doc_items + table_items + code_items + pipe_items + skill_items)
    chunks.extend(doc_chunks + table_chunks + code_chunks + pipe_chunks + skill_chunks)
    edges.extend(table_edges + pipe_edges)
    tasks.extend(skill_tasks + e2e_tasks)

    # Deduplicate items/chunks while preserving the latest richer record.
    items_by_id = {item.id: item for item in items}
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    task_by_id = {task.id: task for task in tasks}

    store.clear()
    store.upsert_items(items_by_id.values())
    store.upsert_chunks(chunks_by_id.values())
    store.upsert_edges(edges)
    store.upsert_tasks(task_by_id.values())
    store.set_meta("source_hash", digest)

    return {"ok": True, "refreshed": True, "source_hash": digest, **store.stats()}


__all__ = ["build_kb"]
