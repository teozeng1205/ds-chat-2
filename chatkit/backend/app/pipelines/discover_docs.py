"""Pass 5 — ASCII-DAG mining from human-authored docs.

Walks every `*.md` file under `~/git/documentations/` (and the top-
level READMEs of every enumerated repo) looking for inline pipeline
references:

  X → Y          (unicode arrow, common in ASCII DAGs)
  X --> Y        (mermaid-style arrow)
  X writes to Y
  X reads from Y
  X feeds into Y

For each such pair, if both sides resolve to a known canonical name
(via aliases.yaml) OR look like an S3 bucket literal / table ref, we
emit an edge with `weight=0.5` (medium confidence) tagged
`source=doc:<file>:<line>`.

This pass depends on the canonicalizer + alias table; it produces no
new entity classes — it just promotes hand-written flow descriptions
into machine-readable edges.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .canonicalize import (
    AliasTable,
    Edge,
    Node,
    RepoEntry,
    canonical_redshift_table,
    canonical_s3_prefix,
    node_id,
)

log = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    nodes: list[Node]
    edges: list[Edge]
    files_scanned: int
    files_with_signal: int


# ── Arrow + prose patterns ─────────────────────────────────────────────

# Each pattern captures (left, right). Written to be conservative: both
# sides must be short token-like strings so we don't swallow whole
# sentences or table-header cells.
_TOKENISH = r"[A-Za-z0-9_][A-Za-z0-9._/\-]{2,80}"

_PATTERNS = (
    ("arrow_unicode", re.compile(rf"`?({_TOKENISH})`?\s*→\s*`?({_TOKENISH})`?")),
    ("arrow_ascii",   re.compile(rf"`?({_TOKENISH})`?\s*-->\s*`?({_TOKENISH})`?")),
    ("writes_to",     re.compile(rf"`?({_TOKENISH})`?\s+writes\s+to\s+`?({_TOKENISH})`?", re.I)),
    ("reads_from",    re.compile(rf"`?({_TOKENISH})`?\s+reads\s+from\s+`?({_TOKENISH})`?", re.I)),
    ("feeds_into",    re.compile(rf"`?({_TOKENISH})`?\s+feeds\s+into\s+`?({_TOKENISH})`?", re.I)),
)


def discover(
    doc_roots: Iterable[Path] | None = None,
    *,
    repos: Iterable[RepoEntry] | None = None,
    aliases: AliasTable | None = None,
    max_file_bytes: int = 256 * 1024,
) -> DiscoveryResult:
    """Scan the canonical documentations directory and every repo's
    top-level README + docs/ tree.
    """
    aliases = aliases or AliasTable.load()
    nodes: list[Node] = []
    edges: list[Edge] = []
    scanned = 0
    with_signal = 0

    paths = _enumerate_doc_files(doc_roots, repos, max_file_bytes)
    for path, repo_ctx in paths:
        scanned += 1
        text = _safe_read(path)
        if not text:
            continue
        rel = str(path)
        if repo_ctx:
            try:
                rel = f"{repo_ctx.name}/{path.relative_to(repo_ctx.local_path)}"
            except ValueError:
                pass
        got = _extract_from_doc(
            text=text,
            source_tag=f"doc:{rel}",
            aliases=aliases,
            nodes=nodes,
            edges=edges,
        )
        if got:
            with_signal += 1

    return DiscoveryResult(
        nodes=nodes, edges=edges,
        files_scanned=scanned, files_with_signal=with_signal,
    )


def _enumerate_doc_files(
    doc_roots: Iterable[Path] | None,
    repos: Iterable[RepoEntry] | None,
    max_file_bytes: int,
) -> list[tuple[Path, RepoEntry | None]]:
    out: list[tuple[Path, RepoEntry | None]] = []

    # 1. Global documentations directory (canonical ATPCO pattern)
    if doc_roots:
        roots = list(doc_roots)
    else:
        roots = [Path("~/git/documentations").expanduser()]
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.md")):
            if p.is_file():
                try:
                    if p.stat().st_size <= max_file_bytes:
                        out.append((p, None))
                except OSError:
                    pass

    # 2. Per-repo READMEs + docs/*.md
    for repo in (repos or []):
        if not repo.local_path.exists():
            continue
        for candidate in ("README.md", "Readme.md", "readme.md"):
            p = repo.local_path / candidate
            if p.exists() and p.is_file():
                try:
                    if p.stat().st_size <= max_file_bytes:
                        out.append((p, repo))
                except OSError:
                    pass
        docs = repo.local_path / "docs"
        if docs.exists() and docs.is_dir():
            for p in sorted(docs.rglob("*.md")):
                if not p.is_file():
                    continue
                try:
                    if p.stat().st_size <= max_file_bytes:
                        out.append((p, repo))
                except OSError:
                    pass
    return out


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _resolve_entity(raw: str, aliases: AliasTable) -> tuple[str, str] | None:
    """Return (kind, canonical_name) for a raw token, or None if we can't
    confidently resolve it to a node the graph already tracks.

    Accepts:
      - a name that's in the alias table                        → ("stage", canonical)
      - a `s3://…` or `s3-atp-…` bucket-prefix literal         → ("s3_prefix", canonical)
      - a dotted table ref like `schema.table` / `db.schema.t`  → ("redshift_table", canonical)
    """
    token = raw.strip().strip("`").strip('"').strip("'").rstrip(".,;:")
    if not token:
        return None
    low = token.lower()

    # 1. Alias match → stage
    canon = aliases.resolve(low)
    if canon in aliases.by_canonical:
        return ("stage", canon)

    # 2. S3 literal
    if low.startswith("s3://") or low.startswith("s3-atp-"):
        path = low[5:] if low.startswith("s3://") else low
        bucket, _, prefix = path.partition("/")
        return ("s3_prefix", canonical_s3_prefix(bucket, prefix))

    # 3. Dotted table ref (need ≥1 dot AND all segments look like identifiers)
    if "." in token and re.fullmatch(r"[\w]+(?:\.[\w]+)+", token):
        return ("redshift_table", canonical_redshift_table(token))

    return None


def _extract_from_doc(
    *,
    text: str,
    source_tag: str,
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
) -> bool:
    emitted = False
    # Iterate over each line so regex matches don't cross paragraph boundaries
    for _lineno, line in enumerate(text.splitlines(), 1):
        # Skip obvious table-header / header-underline lines
        if line.lstrip().startswith(("#", "|--", "|===", "```")):
            continue
        for rel_name, pat in _PATTERNS:
            for m in pat.finditer(line):
                left_raw, right_raw = m.group(1), m.group(2)
                left = _resolve_entity(left_raw, aliases)
                right = _resolve_entity(right_raw, aliases)
                if not left or not right:
                    continue
                if left == right:
                    continue
                # Rel mapping:
                #   arrow / feeds_into / writes_to → writes
                #   reads_from                     → reads
                rel = "reads" if rel_name == "reads_from" else "writes"
                src_id = node_id(*left)
                tgt_id = node_id(*right)
                nodes.append(Node(kind=left[0],  name=left[1],  source=source_tag))
                nodes.append(Node(kind=right[0], name=right[1], source=source_tag))
                edges.append(Edge(source_id=src_id, target_id=tgt_id, rel=rel,
                                  weight=0.5, source=source_tag))
                emitted = True
    return emitted


__all__ = ["discover", "DiscoveryResult"]
