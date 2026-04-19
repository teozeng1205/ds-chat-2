"""Pass 3 — code-pattern extraction (medium confidence, deterministic).

Grep-and-scan pass over every repo's source tree. We emit edges
whenever we see a literal that unambiguously names an S3 location, a
Redshift / MySQL table, or a Glue table. Attribution:

  - if the file's path contains a known app / stage folder name (or
    a filename like `market-level-generator.py`), the edge is
    attributed to that stage,
  - otherwise the edge is attributed to the repo — still useful for
    resolve() and for `trace_pipeline("<bucket>")`, just coarser.

Patterns detected:
  - S3 literals           `s3://<bucket>/<prefix>`
  - Redshift UNLOAD TO    `UNLOAD (...) TO 's3://...'`
  - Redshift COPY FROM    `COPY <table> FROM 's3://...'`
  - INSERT INTO <table>   (write)
  - UPDATE <table>        (write)
  - FROM <schema>.<table> (read, SQL files only — Python FROM would
    require AST and is deferred to a later pass)
  - @ConfigurationProperties(prefix="…") — Java config bindings

Noise filters:
  - skip s3://aws-glue-assets-*/temporary/* (ephemeral)
  - skip test / fixture directories
  - skip venv / node_modules / .git
  - skip vendored deps
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


# ── Extension / path filters ───────────────────────────────────────────


CODE_EXTENSIONS = frozenset({".py", ".java", ".scala", ".sql", ".sh"})

SKIP_DIR_SEGMENTS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".tox", "dist", "build", "target",
    "site-packages", "tests", "test", "__tests__", "fixtures",
    "smoke_reports",
})

# Literal S3 paths we treat as noise
S3_NOISE_PREFIXES = (
    "s3://aws-glue-assets-",
    "s3://config-server-",   # the 3v config bucket, not a data path
)


# ── Public API ─────────────────────────────────────────────────────────


def discover(
    repos: Iterable[RepoEntry],
    *,
    aliases: AliasTable | None = None,
    max_file_bytes: int = 256 * 1024,
) -> DiscoveryResult:
    aliases = aliases or AliasTable.load()
    nodes: list[Node] = []
    edges: list[Edge] = []
    scanned = 0
    with_signal = 0

    for repo in repos:
        if not repo.local_path.exists():
            continue
        known_stages = set(aliases.by_canonical.keys())
        for path in _iter_code_files(repo.local_path, max_file_bytes, repo_root=repo.local_path):
            scanned += 1
            text = _safe_read(path)
            if not text:
                continue
            stage_name = _attribute_to_stage(path, repo, known_stages, aliases)
            source_tag = f"code:{repo.name}/{path.relative_to(repo.local_path)}"
            got = _extract_from_text(
                text=text,
                stage_name=stage_name,
                repo=repo,
                source_tag=source_tag,
                is_sql=path.suffix == ".sql",
                nodes=nodes,
                edges=edges,
            )
            if got:
                with_signal += 1

    return DiscoveryResult(
        nodes=nodes, edges=edges,
        files_scanned=scanned, files_with_signal=with_signal,
    )


# ── Helpers ────────────────────────────────────────────────────────────


def _iter_code_files(root: Path, max_bytes: int, *, repo_root: Path | None = None) -> Iterable[Path]:
    """Walk `root`, yielding code files that aren't in skip-dirs and
    aren't too large to read.

    Skip-dir segments are matched relative to `repo_root` so a test
    fixture whose absolute path happens to include `tests/` above the
    repo root isn't skipped wholesale.
    """
    anchor = (repo_root or root).resolve()
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in CODE_EXTENSIONS:
                continue
            try:
                rel_parts = p.resolve().relative_to(anchor).parts
            except ValueError:
                rel_parts = p.parts
            if any(seg in SKIP_DIR_SEGMENTS for seg in rel_parts):
                continue
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield p
    except (OSError, PermissionError) as exc:
        log.debug("could not walk %s: %s", root, exc)


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return ""


def _attribute_to_stage(
    path: Path, repo: RepoEntry, known_stages: set[str], aliases: AliasTable,
) -> str | None:
    """Pick a stage name for this file if its path looks like it belongs
    to a known app/stage; else return None (attribute to repo)."""
    parts_lower = tuple(seg.lower() for seg in path.parts)
    file_stem = aliases.resolve(path.stem)
    if file_stem in known_stages:
        return file_stem
    for seg in parts_lower:
        canon = aliases.resolve(seg)
        if canon in known_stages:
            return canon
    return None


# ── Pattern matchers ───────────────────────────────────────────────────


_S3_LITERAL = re.compile(r"s3://([a-zA-Z0-9][\w.-]{2,63})/([^\s\"'`)]*)", re.I)

_UNLOAD_TO = re.compile(
    r"UNLOAD\s*\([^)]{5,400}?\)\s*TO\s*['\"]s3://([\w.-]+)/([^\s'\"]*)",
    re.I | re.S,
)

_COPY_FROM = re.compile(
    r"COPY\s+([\w.]+)\s+(?:\([^)]*\)\s+)?FROM\s*['\"]s3://([\w.-]+)/([^\s'\"]*)",
    re.I | re.S,
)

# Insert / update / from patterns. The table identifier is one or more
# dot-separated word segments so we capture the full `db.schema.table`
# form instead of truncating at two segments.
_TABLE_REF = r"\w+(?:\.\w+)+"
_INSERT_INTO = re.compile(rf"\bINSERT\s+INTO\s+({_TABLE_REF})", re.I)

_UPDATE_TABLE = re.compile(rf"\bUPDATE\s+({_TABLE_REF})\s+SET\b", re.I)

_SQL_FROM_SCHEMA_TABLE = re.compile(rf"\bFROM\s+({_TABLE_REF})\b", re.I)


def _strip_sql_comments(text: str) -> str:
    """Drop `-- line` and `/* block */` comments so regex scans don't
    match table-name-looking text inside docstrings."""
    # Block comments
    out = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    # Line comments
    out = re.sub(r"--[^\n]*", "", out)
    return out

_JAVA_CFG_PROPS = re.compile(r"@ConfigurationProperties\s*\(\s*prefix\s*=\s*\"([^\"]+)\"")


def _extract_from_text(
    *,
    text: str,
    stage_name: str | None,
    repo: RepoEntry,
    source_tag: str,
    is_sql: bool,
    nodes: list[Node],
    edges: list[Edge],
) -> bool:
    """Scan `text` for our patterns; append nodes + edges. Returns True
    if any signal was found."""
    emitted = False

    # Source node: stage if we attributed, otherwise repo
    if stage_name:
        src_id = node_id("stage", stage_name)
        nodes.append(Node(
            kind="stage", name=stage_name,
            metadata={"repo": repo.name},
            source=source_tag,
        ))
    else:
        src_id = node_id("repo", repo.name)
        nodes.append(Node(
            kind="repo", name=repo.name,
            metadata={"local_path": str(repo.local_path)},
            source=source_tag,
        ))

    # ── S3 literals ────────────────────────────────────────────────
    for m in _S3_LITERAL.finditer(text):
        uri = m.group(0).rstrip("/")
        if any(uri.startswith(noise) for noise in S3_NOISE_PREFIXES):
            continue
        bucket = m.group(1)
        prefix = m.group(2).rstrip("/")
        canon = canonical_s3_prefix(bucket, prefix)
        sp_id = node_id("s3_prefix", canon)
        nodes.append(Node(
            kind="s3_prefix", name=canon,
            metadata={"bucket": bucket, "prefix": prefix} if prefix else {"bucket": bucket},
            source=source_tag,
        ))
        # We don't know reads vs writes from a bare literal — conservative:
        # mark as "reads" unless the surrounding line hints at a write.
        rel = "writes" if _looks_like_write_context(text, m.start()) else "reads"
        edges.append(Edge(
            source_id=src_id, target_id=sp_id, rel=rel,
            weight=0.6,  # medium confidence
            source=source_tag,
        ))
        emitted = True

    # ── UNLOAD TO s3 (Redshift writes) ─────────────────────────────
    for m in _UNLOAD_TO.finditer(text):
        bucket, prefix = m.group(1), m.group(2).rstrip("/")
        canon = canonical_s3_prefix(bucket, prefix)
        sp_id = node_id("s3_prefix", canon)
        nodes.append(Node(kind="s3_prefix", name=canon,
                          metadata={"bucket": bucket, "prefix": prefix}, source=source_tag))
        edges.append(Edge(source_id=src_id, target_id=sp_id, rel="writes",
                          weight=0.9, source=source_tag))
        emitted = True

    # ── COPY table FROM s3 (Redshift writes a table; reads the s3) ─
    for m in _COPY_FROM.finditer(text):
        tbl, bucket, prefix = m.group(1), m.group(2), m.group(3).rstrip("/")
        tbl_canon = canonical_redshift_table(tbl)
        rt_id = node_id("redshift_table", tbl_canon)
        nodes.append(Node(kind="redshift_table", name=tbl_canon, source=source_tag))
        edges.append(Edge(source_id=src_id, target_id=rt_id, rel="writes",
                          weight=0.9, source=source_tag))
        canon = canonical_s3_prefix(bucket, prefix)
        sp_id = node_id("s3_prefix", canon)
        nodes.append(Node(kind="s3_prefix", name=canon,
                          metadata={"bucket": bucket, "prefix": prefix}, source=source_tag))
        edges.append(Edge(source_id=src_id, target_id=sp_id, rel="reads",
                          weight=0.9, source=source_tag))
        emitted = True

    # Strip SQL comments for the table-ref scans so regex doesn't trip
    # on docstrings / line comments that mention SQL keywords.
    text_for_sql = _strip_sql_comments(text) if is_sql else text

    # ── INSERT INTO <table> (write) ────────────────────────────────
    for m in _INSERT_INTO.finditer(text_for_sql):
        tbl = m.group(1)
        if "." not in tbl:    # require schema-qualified to cut noise
            continue
        canon = canonical_redshift_table(tbl)
        rt_id = node_id("redshift_table", canon)
        nodes.append(Node(kind="redshift_table", name=canon, source=source_tag))
        edges.append(Edge(source_id=src_id, target_id=rt_id, rel="writes",
                          weight=0.75, source=source_tag))
        emitted = True

    # ── UPDATE <table> SET (write) ─────────────────────────────────
    for m in _UPDATE_TABLE.finditer(text_for_sql):
        tbl = m.group(1)
        if "." not in tbl:
            continue
        canon = canonical_redshift_table(tbl)
        rt_id = node_id("redshift_table", canon)
        nodes.append(Node(kind="redshift_table", name=canon, source=source_tag))
        edges.append(Edge(source_id=src_id, target_id=rt_id, rel="writes",
                          weight=0.7, source=source_tag))
        emitted = True

    # ── SQL FROM <schema>.<table> (read) — SQL files only ─────────
    if is_sql:
        for m in _SQL_FROM_SCHEMA_TABLE.finditer(text_for_sql):
            tbl = m.group(1)
            canon = canonical_redshift_table(tbl)
            rt_id = node_id("redshift_table", canon)
            nodes.append(Node(kind="redshift_table", name=canon, source=source_tag))
            edges.append(Edge(source_id=src_id, target_id=rt_id, rel="reads",
                              weight=0.7, source=source_tag))
            emitted = True

    return emitted


_WRITE_HINTS = re.compile(
    r"\b(put_object|upload_file|upload|write|to_parquet|to_csv|to_json|save|sink|unload|output|dest)\b",
    re.I,
)


def _looks_like_write_context(text: str, position: int) -> bool:
    """Heuristic: the ~80 chars around `position` contain a write verb."""
    start = max(0, position - 80)
    end = min(len(text), position + 80)
    window = text[start:end]
    return bool(_WRITE_HINTS.search(window))


__all__ = ["discover", "DiscoveryResult"]
