"""Build the semantic KB index from local knowledge files.

Chunks the knowledge-base markdown + JSON into small, semantically
coherent pieces and embeds each with OpenAI text-embedding-3-large,
persisting results in the SemanticIndex at
`app/.data/ds-chat-semantic.sqlite` (override via env).

Sources covered:
  - `knowledge/tables.md`               (chunked per-section + per-table row)
  - `knowledge/docs/*.md`                (chunked per-heading)
  - `knowledge/sql_best_practices.md`    (chunked per-heading)
  - `knowledge/common_codes.json`        (one chunk per provider/site/customer)

Dry-run mode (`--dry-run`) skips the OpenAI call and the upsert, and
prints chunk stats so the chunking pipeline can be validated without
spend.

CLI:
  python -m app.investigation.knowledge.build_embeddings            # full build
  python -m app.investigation.knowledge.build_embeddings --dry-run  # no OpenAI call
  python -m app.investigation.knowledge.build_embeddings --kinds docs,tables
  python -m app.investigation.knowledge.build_embeddings --clear    # wipe before rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_DIR = BACKEND_ROOT / "app" / "investigation" / "knowledge"
DOCS_DIR = KNOWLEDGE_DIR / "docs"

DEFAULT_EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100
MAX_CHUNK_CHARS = 4000


# ── Chunking ──

@dataclass
class Chunk:
    id: str
    text: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _chunk_markdown(path: Path, kind: str) -> list[Chunk]:
    """Split markdown by heading boundaries and emit one Chunk per section."""
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    headings = list(_HEADING.finditer(text))

    if not headings:
        chunks.append(Chunk(id=f"{kind}:{path.name}", text=text[:MAX_CHUNK_CHARS], kind=kind,
                            metadata={"source": str(path.relative_to(KNOWLEDGE_DIR))}))
        return chunks

    for i, m in enumerate(headings):
        start = m.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[start:end].strip()
        if not section:
            continue
        heading = m.group(2).strip()
        section_id = f"{kind}:{path.name}#{heading[:80]}"
        chunks.append(
            Chunk(
                id=section_id,
                text=section[:MAX_CHUNK_CHARS],
                kind=kind,
                metadata={
                    "source": str(path.relative_to(KNOWLEDGE_DIR)),
                    "heading": heading,
                    "level": len(m.group(1)),
                },
            )
        )
    return chunks


def _chunk_tables_md() -> list[Chunk]:
    """tables.md is a structured reference with tier tables + code-path tables.
    Chunk per heading for context, plus split especially long table blocks."""
    path = KNOWLEDGE_DIR / "tables.md"
    if not path.exists():
        return []
    return _chunk_markdown(path, kind="tables")


def _chunk_docs() -> list[Chunk]:
    out: list[Chunk] = []
    if not DOCS_DIR.exists():
        return out
    for md in sorted(DOCS_DIR.glob("*.md")):
        out.extend(_chunk_markdown(md, kind="doc"))
    return out


def _chunk_sql_best_practices() -> list[Chunk]:
    path = KNOWLEDGE_DIR / "sql_best_practices.md"
    if not path.exists():
        return []
    return _chunk_markdown(path, kind="sql_best_practices")


def _chunk_common_codes() -> list[Chunk]:
    path = KNOWLEDGE_DIR / "common_codes.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[Chunk] = []
    for category in ("providers", "sites", "customers"):
        entries = payload.get(category) or []
        for entry in entries:
            if isinstance(entry, str):
                code = entry
                text = f"{category[:-1]} code {code}"
                aliases: list[str] = []
                name = None
            else:
                code = entry.get("code") or entry.get("name") or ""
                name = entry.get("name")
                aliases = entry.get("aliases") or []
                bits: list[str] = [f"{category[:-1]} code: {code}"]
                if name:
                    bits.append(f"name: {name}")
                if aliases:
                    bits.append(f"aliases: {', '.join(aliases)}")
                text = " | ".join(bits)
            cid = f"code:{category}:{code}"
            out.append(
                Chunk(
                    id=cid,
                    text=text,
                    kind="code",
                    metadata={"category": category, "code": code, "name": name, "aliases": aliases},
                )
            )
    return out


def _chunk_pipelines() -> list[Chunk]:
    """Emit one chunk per app/stage from `pipelines.json` with its
    1-hop neighborhood rendered as a short paragraph.

    When the agent embeds a question like "how does market-level get
    generated?", the semantic hit is a pre-rendered chain summary —
    graph precision + semantic recall. Safe no-op when the graph
    hasn't been built yet.
    """
    path = KNOWLEDGE_DIR / "pipelines.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    nodes_by_kind: dict[str, list[dict[str, Any]]] = payload.get("nodes", {}) or {}
    edges: list[dict[str, Any]] = payload.get("edges", []) or []

    by_id: dict[str, dict[str, Any]] = {}
    for kind, items in nodes_by_kind.items():
        for n in items:
            by_id[n["id"]] = n

    reads_by_src: dict[str, list[str]] = {}
    writes_by_src: dict[str, list[str]] = {}
    for e in edges:
        src, tgt, rel = e.get("source"), e.get("target"), e.get("rel")
        if not src or not tgt:
            continue
        if rel == "reads":
            reads_by_src.setdefault(src, []).append(tgt)
        elif rel == "writes":
            writes_by_src.setdefault(src, []).append(tgt)

    def _pretty(nid: str) -> str:
        n = by_id.get(nid)
        return n["name"] if n else nid

    out: list[Chunk] = []
    for kind in ("app", "stage"):
        for node in nodes_by_kind.get(kind, []):
            nid = node["id"]
            name = node["name"]
            meta = node.get("metadata") or {}
            reads = [_pretty(x) for x in reads_by_src.get(nid, [])][:6]
            writes = [_pretty(x) for x in writes_by_src.get(nid, [])][:6]
            aliases = [a for a in (node.get("aliases") or []) if a != name]

            parts: list[str] = [f"`{name}` — pipeline {kind}."]
            repo = meta.get("repo")
            if repo:
                parts.append(f"Repo: `{repo}`.")
            if aliases:
                parts.append("Aliases: " + ", ".join(aliases[:6]) + ".")
            if reads:
                parts.append("Reads from: " + ", ".join(reads) + ".")
            if writes:
                parts.append("Writes to: " + ", ".join(writes) + ".")
            if not reads and not writes:
                continue  # isolated node: no embedding signal worth adding

            text = " ".join(parts)
            out.append(
                Chunk(
                    id=f"pipeline:{nid}",
                    text=text[:MAX_CHUNK_CHARS],
                    kind="pipeline",
                    metadata={
                        "node_id": nid,
                        "node_kind": kind,
                        "name": name,
                        "repo": repo,
                        "aliases": aliases,
                        "reads": reads,
                        "writes": writes,
                    },
                )
            )
    return out


ALL_BUILDERS = {
    "tables": _chunk_tables_md,
    "docs": _chunk_docs,
    "sql_best_practices": _chunk_sql_best_practices,
    "codes": _chunk_common_codes,
    "pipelines": _chunk_pipelines,
}


def build_chunks(kinds: Iterable[str] | None = None) -> list[Chunk]:
    wanted = set(kinds) if kinds else set(ALL_BUILDERS)
    chunks: list[Chunk] = []
    for key, fn in ALL_BUILDERS.items():
        if key not in wanted:
            continue
        chunks.extend(fn())
    return chunks


# ── Embedding ──

def _batched(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(texts: list[str], model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    """Batch-embed via OpenAI. Returns one vector per input text."""
    from openai import OpenAI  # lazy import so unit tests don't need a key

    client = OpenAI()
    out: list[list[float]] = []
    for batch in _batched(texts, BATCH_SIZE):
        resp = client.embeddings.create(model=model, input=batch)
        # Responses come back in input order.
        for item in resp.data:
            out.append(list(item.embedding))
    return out


# ── Main builder ──

def run_build(
    kinds: Iterable[str] | None = None,
    *,
    dry_run: bool = False,
    clear: bool = False,
    model: str = DEFAULT_EMBED_MODEL,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Chunk → embed → upsert. Returns a summary dict."""
    from app.investigation.semantic_index import SemanticIndex

    chunks = build_chunks(kinds)
    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1

    summary: dict[str, Any] = {
        "chunks_total": len(chunks),
        "chunks_by_kind": by_kind,
        "model": model,
        "dry_run": dry_run,
    }

    if dry_run:
        return summary

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set; run with --dry-run to validate chunking only.")

    if index_path is None:
        index_path = BACKEND_ROOT / "app" / ".data" / "ds-chat-semantic.sqlite"
        index_path.parent.mkdir(parents=True, exist_ok=True)
    index = SemanticIndex(index_path)
    if clear:
        index.clear()

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, model=model)
    for chunk, vec in zip(chunks, vectors):
        index.upsert(chunk.id, chunk.text, vec, kind=chunk.kind, metadata=chunk.metadata)
    summary["index_path"] = str(index_path)
    summary["index_count"] = index.count()
    index.close()
    return summary


# ── CLI ──

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the DS Chat semantic KB index.")
    p.add_argument("--kinds", default=None,
                   help=f"Comma-separated: {','.join(ALL_BUILDERS)}. Default = all.")
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--dry-run", action="store_true", help="Chunk only; no OpenAI call / upsert.")
    p.add_argument("--clear", action="store_true", help="Wipe the index before rebuilding.")
    p.add_argument("--index", type=Path, default=None, help="Override index SQLite path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    args = _parse_args(argv)
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    summary = run_build(
        kinds,
        dry_run=args.dry_run,
        clear=args.clear,
        model=args.model,
        index_path=args.index,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
