"""Pass 1 — structured-config parser (high confidence, deterministic).

Scans every repo's declared `config_roots` for machine-readable manifests
and emits graph nodes + edges. The common shape we extract (regardless
of format) is:

    stage <name>
        reads  s3://<bucket>/<prefix>  (version prefix if present)
        reads  redshift <schema>.<table>
        writes s3://<bucket>/<prefix>
        writes redshift <schema>.<table>
        writes glue    <db>.<table>
        part_of <pipeline>
        repo   <repo_name>

Formats handled today:
  - Java `.properties`  (the priceeye-analytics / data-collection
    pattern — 12 stages already in `ds-priceeye-analytics/docs/
    config_gold_prod`)
  - AWS SAM / CloudFormation `template.yaml` is stubbed; a future
    commit will add a proper Resources-tree walker.

Each emitted Edge carries `source = "config:<relative_path>:<line>"`
so we can trace any fact back to its origin file.
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
    canonical_glue_table,
    canonical_redshift_table,
    canonical_s3_prefix,
    expand_environment,
    node_id,
)

log = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────


@dataclass
class DiscoveryResult:
    nodes: list[Node]
    edges: list[Edge]
    files_scanned: int
    files_with_signal: int


def discover(
    repos: Iterable[RepoEntry],
    *,
    aliases: AliasTable | None = None,
) -> DiscoveryResult:
    """Run Pass 1 over every repo's declared config_roots. Returns a
    flat DiscoveryResult whose nodes/edges may contain duplicates — the
    caller is expected to merge via `canonicalize.merge_nodes/merge_edges`.
    """
    aliases = aliases or AliasTable.load()
    nodes: list[Node] = []
    edges: list[Edge] = []
    files_scanned = 0
    files_with_signal = 0

    for repo in repos:
        for root in repo.config_roots:
            if not root.exists():
                log.debug("config root missing: %s", root)
                continue
            for path in sorted(root.rglob("*.properties")):
                files_scanned += 1
                got_signal = _parse_properties_file(
                    path=path,
                    repo=repo,
                    aliases=aliases,
                    nodes=nodes,
                    edges=edges,
                )
                if got_signal:
                    files_with_signal += 1
            # SAM templates (future — stub)
            for path in sorted(list(root.rglob("template.yaml")) + list(root.rglob("template.yml"))):
                files_scanned += 1
                # TODO(lineage/sam): walk Resources tree; for now skip.
                _ = path

    return DiscoveryResult(
        nodes=nodes,
        edges=edges,
        files_scanned=files_scanned,
        files_with_signal=files_with_signal,
    )


# ── .properties parser ─────────────────────────────────────────────────


# Matches `key = value` with optional inline comments stripped later.
_PROP_LINE = re.compile(r"^\s*([^#!=:\s][^=:\s]*)\s*[=:]\s*(.*?)\s*$")


def _parse_properties_file(
    *,
    path: Path,
    repo: RepoEntry,
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
) -> bool:
    """Parse one `.properties` file. Returns True if any signal was extracted."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", path, exc)
        return False

    props: dict[str, tuple[str, int]] = {}  # key → (value, lineno)
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line:
            continue
        m = _PROP_LINE.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2).strip().strip('"').strip("'")
        props[key.lower()] = (val, i)

    if not props:
        return False

    # Stage name comes from the file name (the de-facto convention in
    # config_gold_prod/*.properties). `alerts.properties` → stage
    # "alerts", `market-level-generator.properties` → stage
    # "market-level-generator".
    stage_name = aliases.resolve(path.stem)
    stage_id = node_id("stage", stage_name)
    app_id = node_id("app", stage_name)
    try:
        rel_source = str(path.relative_to(repo.local_path))
    except ValueError:
        rel_source = str(path)
    source_tag = f"config:{repo.name}/{rel_source}"

    # Stage / app nodes
    nodes.append(Node(
        kind="stage",
        name=stage_name,
        aliases=tuple(sorted(aliases.by_canonical.get(stage_name, frozenset({stage_name})))),
        metadata={"repo": repo.name, "config_file": rel_source},
        source=source_tag,
    ))
    nodes.append(Node(
        kind="app",
        name=stage_name,
        aliases=tuple(sorted(aliases.by_canonical.get(stage_name, frozenset({stage_name})))),
        metadata={"repo": repo.name},
        source=source_tag,
    ))
    # stage → app link is implicit via shared name; no edge needed.
    # stage → pipelines
    for pipe in repo.pipelines:
        edges.append(Edge(
            source_id=stage_id,
            target_id=node_id("pipeline", pipe),
            rel="part_of",
            source=source_tag,
        ))
        nodes.append(Node(kind="pipeline", name=pipe.strip().lower(), source=source_tag))
    # app → repo
    edges.append(Edge(
        source_id=app_id,
        target_id=node_id("repo", repo.name),
        rel="repo",
        source=source_tag,
    ))
    nodes.append(Node(
        kind="repo",
        name=repo.name.strip().lower(),
        metadata={"local_path": str(repo.local_path)},
        source=source_tag,
    ))

    got_signal = False

    # Collect all input.* and output.* families. Properties files group
    # them by optional sub-key (e.g. `input.bucket`, `input.prefix`,
    # `input.version`, `input_table.dco`).
    def _collect_family(prefix: str) -> dict[str, dict[str, tuple[str, int]]]:
        """Group props by sub-key after `prefix`.

        prefix="input"  returns { "":             { "bucket": ("...", 3), "prefix": ... },
                                  "_table":       { "dco": ("...", 7) } }
        """
        groups: dict[str, dict[str, tuple[str, int]]] = {}
        for k, (v, ln) in props.items():
            # Accept both `input.*` and `input_*.*` (the second form is how
            # the priceeye-analytics configs encode table inputs).
            for group_key in ("", "_table"):
                full_prefix = prefix + group_key
                if k == full_prefix or k.startswith(full_prefix + "."):
                    rest = k[len(full_prefix):].lstrip(".")
                    groups.setdefault(group_key, {})[rest] = (v, ln)
                    break
        return groups

    # ── INPUT side ─────────────────────────────────────────────────
    input_groups = _collect_family("input")
    for group_key, kv in input_groups.items():
        got_signal |= _emit_io_edges(
            kind="reads",
            group_key=group_key,
            kv=kv,
            stage_id=stage_id,
            nodes=nodes,
            edges=edges,
            source_tag=source_tag,
        )

    # ── OUTPUT side ────────────────────────────────────────────────
    output_groups = _collect_family("output")
    for group_key, kv in output_groups.items():
        got_signal |= _emit_io_edges(
            kind="writes",
            group_key=group_key,
            kv=kv,
            stage_id=stage_id,
            nodes=nodes,
            edges=edges,
            source_tag=source_tag,
        )

    # ── Glue target ────────────────────────────────────────────────
    glue_db_val, glue_tbl_val = None, None
    glue_ln = 0
    if "glue.database" in props:
        glue_db_val, glue_ln = props["glue.database"]
    if "glue.table" in props:
        glue_tbl_val, glue_ln = props["glue.table"]
    if glue_db_val and glue_tbl_val:
        for db_x in expand_environment(glue_db_val):
            canon = canonical_glue_table(db_x, glue_tbl_val)
            gt_id = node_id("glue_table", canon)
            nodes.append(Node(
                kind="glue_table",
                name=canon,
                metadata={"database": db_x, "table": glue_tbl_val},
                source=f"{source_tag}:{glue_ln}",
            ))
            edges.append(Edge(
                source_id=stage_id,
                target_id=gt_id,
                rel="writes",
                source=f"{source_tag}:{glue_ln}",
            ))
            got_signal = True

    return got_signal


def _emit_io_edges(
    *,
    kind: str,                # "reads" | "writes"
    group_key: str,           # "" for bucket/prefix, "_table" for redshift
    kv: dict[str, tuple[str, int]],
    stage_id: str,
    nodes: list[Node],
    edges: list[Edge],
    source_tag: str,
) -> bool:
    emitted = False

    if group_key == "":
        # S3 form: bucket [+ prefix] [+ version] [+ customer / carrier]
        bucket_val = kv.get("bucket", (None, 0))[0]
        if bucket_val:
            prefix_val = kv.get("prefix", ("", 0))[0] or ""
            version_val = kv.get("version", ("", 0))[0] or ""
            # Compose the effective prefix. ATPCO conventions see both
            # `input.prefix=v1` and `input.version=v1`; treat either as
            # the first path segment.
            effective_prefix = "/".join(p for p in (prefix_val, version_val) if p and p != prefix_val)
            if not effective_prefix and version_val and not prefix_val:
                effective_prefix = version_val
            if not effective_prefix and prefix_val and not version_val:
                effective_prefix = prefix_val
            # Expand `${environment}`
            bucket_ln = kv.get("bucket", (None, 0))[1]
            for bucket_x in expand_environment(bucket_val):
                canon = canonical_s3_prefix(bucket_x, effective_prefix)
                sp_id = node_id("s3_prefix", canon)
                nodes.append(Node(
                    kind="s3_prefix",
                    name=canon,
                    metadata={
                        "bucket": bucket_x,
                        "prefix": effective_prefix,
                        **({"version": version_val} if version_val else {}),
                        **({k: v for k, (v, _ln) in kv.items() if k in ("customer", "carrier") and v}),
                    },
                    source=f"{source_tag}:{bucket_ln}",
                ))
                edges.append(Edge(
                    source_id=stage_id,
                    target_id=sp_id,
                    rel=kind,
                    source=f"{source_tag}:{bucket_ln}",
                ))
                emitted = True

    elif group_key == "_table":
        # Redshift table form: input_table.<sub>=schema.table
        for sub, (table_val, ln) in kv.items():
            if not table_val:
                continue
            for table_x in expand_environment(table_val):
                canon = canonical_redshift_table(table_x)
                rt_id = node_id("redshift_table", canon)
                nodes.append(Node(
                    kind="redshift_table",
                    name=canon,
                    metadata={"logical_role": sub},
                    source=f"{source_tag}:{ln}",
                ))
                edges.append(Edge(
                    source_id=stage_id,
                    target_id=rt_id,
                    rel=kind,
                    source=f"{source_tag}:{ln}",
                ))
                emitted = True

    return emitted


__all__ = ["discover", "DiscoveryResult"]
