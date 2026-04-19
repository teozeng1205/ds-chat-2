"""Canonical-name helpers for the pipeline lineage graph.

Every node in the graph has a stable ID of the form

    "<kind>:<canonical_name>"

where `kind` is one of app / stage / redshift_table / s3_prefix /
glue_table / step_fn / lambda / ecs_service / event_rule, and
`canonical_name` is the lowercased, hyphenated, environment-expanded
short name.

This module owns:
  - alias resolution  (so "DCO" / "derived_common_output" / "Derived
    Common Output" all collapse to "derived-common-output"),
  - `${environment}` expansion  (so one config line produces both a
    `3vdev` and a `3vprod` node),
  - S3 / Glue / Redshift reference parsing into canonical forms.

Kept intentionally free of network / boto3 calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# yaml is already a pyproject dep (pyyaml>=6)
import yaml

NodeKind = str  # "app" | "stage" | "redshift_table" | "s3_prefix" | …

BACKEND_ROOT = Path(__file__).resolve().parents[2]
ALIASES_PATH = Path(__file__).resolve().parent / "aliases.yaml"
REPOS_PATH = Path(__file__).resolve().parent / "repos.yaml"

# Environments we expand ${environment} to. Order matters only for
# report readability.
ENVIRONMENTS = ("3vdev", "3vprod")

_ENV_PLACEHOLDER = re.compile(r"\$\{\s*environment\s*\}", re.IGNORECASE)


# ── Loaders ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AliasTable:
    """Canonical → set of known alternate spellings (lowercased)."""

    by_canonical: dict[str, frozenset[str]]
    by_alias: dict[str, str]  # alternate (lowercased) → canonical

    @classmethod
    def load(cls, path: Path | None = None) -> "AliasTable":
        p = path or ALIASES_PATH
        if not p.exists():
            return cls(by_canonical={}, by_alias={})
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        by_canonical: dict[str, frozenset[str]] = {}
        by_alias: dict[str, str] = {}
        for canonical, alts in data.items():
            if not isinstance(canonical, str):
                continue
            canon = canonical.strip().lower()
            alt_set: set[str] = {canon}
            for alt in alts or []:
                if isinstance(alt, str) and alt.strip():
                    alt_set.add(alt.strip().lower())
            by_canonical[canon] = frozenset(alt_set)
            for alt in alt_set:
                by_alias.setdefault(alt, canon)
        return cls(by_canonical=by_canonical, by_alias=by_alias)

    def resolve(self, raw: str) -> str:
        """Return the canonical name for a raw alias, or a sensible
        canonical form of `raw` when no alias matches."""
        if not raw:
            return ""
        key = raw.strip().lower()
        if key in self.by_alias:
            return self.by_alias[key]
        # Normalize: replace underscores / spaces with hyphens.
        return re.sub(r"[\s_]+", "-", key)


@dataclass(frozen=True)
class RepoEntry:
    name: str
    local_path: Path
    config_roots: tuple[Path, ...]
    pipelines: tuple[str, ...]


def load_repos(path: Path | None = None) -> list[RepoEntry]:
    """Read repos.yaml. Paths are expanded; missing ones are kept so
    diagnostics can report them."""
    p = path or REPOS_PATH
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: list[RepoEntry] = []
    for r in data.get("repos", []) or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        local_path = Path(str(r.get("local_path") or f"~/git/{name}")).expanduser().resolve()
        roots_raw = r.get("config_roots") or []
        roots = tuple(
            (local_path / str(root)).resolve() if not str(root).startswith("/") else Path(str(root)).resolve()
            for root in roots_raw
            if isinstance(root, str)
        )
        pipelines = tuple(str(p) for p in (r.get("pipelines") or []) if isinstance(p, str))
        out.append(RepoEntry(name=name, local_path=local_path, config_roots=roots, pipelines=pipelines))
    return out


# ── Canonical-name builders ────────────────────────────────────────────


def node_id(kind: NodeKind, canonical_name: str) -> str:
    """Build the stable node ID."""
    return f"{kind}:{canonical_name.strip().lower()}"


def expand_environment(value: str) -> list[str]:
    """Expand `${environment}` placeholders to each configured env.

    If the value has no placeholder, returns `[value]` unchanged.
    """
    if not value:
        return [""]
    if not _ENV_PLACEHOLDER.search(value):
        return [value]
    return [_ENV_PLACEHOLDER.sub(env, value) for env in ENVIRONMENTS]


def canonical_s3_prefix(bucket: str, prefix: str = "") -> str:
    """Combine bucket + prefix into a canonical S3 path string.

    Strips any leading `s3://`, trims slashes, lowercases the bucket.
    The prefix keeps its case (prefixes can be case-sensitive in S3
    conventions like `market-level/v4`).
    """
    b = bucket.strip().lower()
    if b.startswith("s3://"):
        b = b[len("s3://"):]
    b = b.strip("/")
    p = (prefix or "").strip("/")
    if not p:
        return b
    return f"{b}/{p}"


def canonical_redshift_table(ref: str) -> str:
    """Normalize a Redshift table reference.

    Accepts bare `foo`, two-part `schema.foo`, three-part `db.schema.foo`.
    Output lower-cases the schema/db but leaves the literal table name
    alone (tables are already conventionally lowercase in ATPCO's stack).
    """
    r = (ref or "").strip().lower()
    r = r.strip('"').strip("'").strip("`")
    # drop environment qualifiers the agent sometimes writes like "prod."
    # we keep them — they're part of the identity.
    return r


def canonical_glue_table(database: str, table: str) -> str:
    return f"{(database or '').strip().lower()}.{(table or '').strip().lower()}"


# ── Node + edge dataclasses ────────────────────────────────────────────


@dataclass
class Node:
    kind: NodeKind
    name: str                              # canonical name
    aliases: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict = field(default_factory=dict)
    source: str = ""                       # where this fact came from

    @property
    def id(self) -> str:
        return node_id(self.kind, self.name)


@dataclass
class Edge:
    source_id: str
    target_id: str
    rel: str                               # "reads" | "writes" | "triggers" | "deploys_as" | "part_of" | "depends_on" | "repo"
    weight: float = 1.0
    source: str = ""
    metadata: dict = field(default_factory=dict)


# ── Merge rules ────────────────────────────────────────────────────────


_REL_PRIORITY = {
    "writes": 3,
    "reads": 3,
    "triggers": 2,
    "deploys_as": 2,
    "part_of": 1,
    "depends_on": 1,
    "repo": 1,
}


def merge_nodes(existing: Node, new: Node) -> Node:
    """Merge two Node records with the same ID — union aliases, shallow-merge
    metadata (new wins for concrete keys, existing stays for missing ones),
    concat sources uniquely."""
    assert existing.id == new.id
    merged_aliases = tuple(sorted({*existing.aliases, *new.aliases}))
    merged_meta = {**existing.metadata, **{k: v for k, v in new.metadata.items() if v not in (None, "", [], {})}}
    sources = existing.source.split(", ") if existing.source else []
    if new.source and new.source not in sources:
        sources.append(new.source)
    return Node(
        kind=existing.kind,
        name=existing.name,
        aliases=merged_aliases,
        metadata=merged_meta,
        source=", ".join(sources),
    )


def merge_edges(edges: Iterable[Edge]) -> list[Edge]:
    """Collapse edges with the same (source_id, target_id, rel). Keep the
    highest-priority rel if two passes disagree on it; otherwise union
    sources."""
    keyed: dict[tuple[str, str, str], Edge] = {}
    for e in edges:
        key = (e.source_id, e.target_id, e.rel)
        if key in keyed:
            prior = keyed[key]
            sources = set(filter(None, prior.source.split(", "))) | set(filter(None, e.source.split(", ")))
            keyed[key] = Edge(
                source_id=e.source_id,
                target_id=e.target_id,
                rel=e.rel,
                weight=max(prior.weight, e.weight),
                source=", ".join(sorted(sources)),
                metadata={**prior.metadata, **e.metadata},
            )
        else:
            keyed[key] = e
    return list(keyed.values())


__all__ = [
    "AliasTable",
    "Edge",
    "Node",
    "NodeKind",
    "RepoEntry",
    "ENVIRONMENTS",
    "canonical_glue_table",
    "canonical_redshift_table",
    "canonical_s3_prefix",
    "expand_environment",
    "load_repos",
    "merge_edges",
    "merge_nodes",
    "node_id",
]
