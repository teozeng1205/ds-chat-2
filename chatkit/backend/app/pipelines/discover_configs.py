"""Pass 1 — structured-config parser (high confidence, deterministic).

Scans every repo's declared `config_roots` (+ the whole repo for CFN
templates) for machine-readable manifests and emits graph nodes + edges.
The common shape we extract is:

    stage <name>
        reads  s3://<bucket>/<prefix>  (version prefix if present)
        reads  redshift <schema>.<table>
        writes s3://<bucket>/<prefix>
        writes redshift <schema>.<table>
        writes glue    <db>.<table>
        part_of <pipeline>
        repo   <repo_name>

Formats handled:
  - Java `.properties` manifests (the priceeye-analytics / data-
    collection pattern — 12 stages live in `ds-priceeye-analytics/
    docs/config_gold_prod`).
  - CloudFormation / SAM YAML templates (`**/deploy/**/*.yaml`,
    `**/template.yaml`, `**/serverless.yml`). Walks the Resources tree
    for Lambda / ECS TaskDefinition env vars and extracts any `s3://`
    literal or dotted `schema.table` reference it finds.

Each emitted Edge carries `source = "config:<relative_path>:<line>"`
so we can trace any fact back to its origin file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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

        # CFN / SAM templates — walk the whole repo, not just config_roots,
        # because deploy/ folders are nearly always outside the config
        # roots a caller supplies.
        if repo.local_path.exists():
            for path in _iter_cfn_templates(repo.local_path):
                files_scanned += 1
                got_signal = _parse_cfn_template(
                    path=path,
                    repo=repo,
                    aliases=aliases,
                    nodes=nodes,
                    edges=edges,
                )
                if got_signal:
                    files_with_signal += 1

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


# ── CloudFormation / SAM template parser ──────────────────────────────


# File types that hold CFN / SAM / Serverless templates. We use a prefix
# test against the yaml body to confirm (needs an `AWSTemplateFormatVersion`,
# `Resources`, or `Transform: AWS::Serverless` line) before spending the
# parse cost.
_CFN_GLOBS = (
    "**/deploy/**/*.yaml",
    "**/deploy/**/*.yml",
    "**/template.yaml",
    "**/template.yml",
    "**/serverless.yaml",
    "**/serverless.yml",
    "**/cloudformation/**/*.yaml",
    "**/cloudformation/**/*.yml",
    "**/infra/**/*.yaml",
    "**/infra/**/*.yml",
)

_CFN_SKIP_DIRS = frozenset({
    "node_modules", ".venv", "venv", ".pytest_cache", "__pycache__",
    ".mypy_cache", "dist", "build", "target", ".git",
})

_S3_LITERAL_RE = re.compile(r"s3://([A-Za-z0-9\-.]+)(?:/([^\s\"'<>]+))?")
_ATP_BUCKET_RE = re.compile(
    r"(s3-atp-3victors[A-Za-z0-9$\{\}\-]+?-use1-[A-Za-z0-9\-]+)"
)
_DOTTED_TABLE_RE = re.compile(
    r"\b([a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)\b"
)

# Env-var key hints we flag as input / output. Same vocabulary as Pass 2.
_INPUT_KEY_HINTS = ("INPUT_", "SOURCE_", "READ_", "UPSTREAM_")
_OUTPUT_KEY_HINTS = ("OUTPUT_", "SINK_", "WRITE_", "TARGET_", "DEST_")
_BUCKET_KEY_HINTS = ("BUCKET", "_S3_", "S3_PREFIX", "S3URI")
_TABLE_KEY_HINTS = ("TABLE", "REDSHIFT", "GLUE_TABLE")


def _iter_cfn_templates(repo_root: Path) -> Iterable[Path]:
    """Yield candidate CFN/SAM YAML files, de-duplicated across glob
    patterns and filtered to ones whose content looks like a CFN
    template. Bounded by skip-dirs to avoid walking `.venv` etc.
    """
    seen: set[Path] = set()
    for pattern in _CFN_GLOBS:
        for p in repo_root.glob(pattern):
            try:
                rel = p.relative_to(repo_root)
            except ValueError:
                continue
            if any(seg in _CFN_SKIP_DIRS for seg in rel.parts):
                continue
            if p in seen or not p.is_file():
                continue
            try:
                head = p.read_text(encoding="utf-8", errors="replace")[:4000]
            except Exception:
                continue
            if not _looks_like_cfn(head):
                continue
            seen.add(p)
            yield p


def _looks_like_cfn(head: str) -> bool:
    """Heuristic: file's first 4kB contains a CFN / SAM marker."""
    return (
        "Resources:" in head
        or "AWSTemplateFormatVersion" in head
        or "Transform: AWS::Serverless" in head
        or "service:" in head and "provider:" in head  # serverless.yml
    )


def _parse_cfn_template(
    *,
    path: Path,
    repo: RepoEntry,
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
) -> bool:
    """Parse one CFN / SAM template. Returns True if any signal was extracted.

    We don't do full `!Ref` / `!Sub` resolution — too much code for too
    little gain. Instead we treat the YAML as text + a shallow dict walk:

      1. Pull AppName default from `Parameters` → stage name.
      2. Regex-extract every `s3://bucket/prefix` literal.
      3. Regex-extract every `s3-atp-3victors...-use1-*` bucket literal
         (ATPCO's convention even inside `!Sub`).
      4. For Lambda / ECS TaskDefinition containers, walk
         `Environment` lists and emit reads/writes edges when env var
         names match our INPUT_ / OUTPUT_ / BUCKET / TABLE hints.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", path, exc)
        return False

    try:
        import yaml  # type: ignore[import]
        # CFN uses !Ref / !Sub / !Join / etc. short-tags — ignore them
        # by treating unknown tags as plain mappings.
        class _LenientLoader(yaml.SafeLoader):  # type: ignore[misc]
            pass

        def _ignore_tag(loader, tag_suffix, node):  # type: ignore[no-untyped-def]
            if isinstance(node, yaml.ScalarNode):
                return loader.construct_scalar(node)
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            if isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node)
            return None

        _LenientLoader.add_multi_constructor("!", _ignore_tag)
        parsed = yaml.load(text, Loader=_LenientLoader) or {}
    except Exception as exc:  # noqa: BLE001
        log.debug("yaml parse failed for %s: %s", path, exc)
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    # Stage name
    stage_raw = _stage_name_from_template(parsed, fallback=path.stem)
    stage_name = aliases.resolve(stage_raw)
    if not stage_name:
        return False

    try:
        rel_source = str(path.relative_to(repo.local_path))
    except ValueError:
        rel_source = str(path)
    source_tag = f"config:{repo.name}/{rel_source}"
    stage_id = node_id("stage", stage_name)
    nodes.append(Node(
        kind="stage", name=stage_name,
        aliases=(stage_raw,) if stage_raw != stage_name else (),
        metadata={"repo": repo.name, "template": rel_source},
        source=source_tag,
    ))

    # Link to pipelines
    for pipe in repo.pipelines:
        edges.append(Edge(
            source_id=stage_id, target_id=node_id("pipeline", pipe),
            rel="part_of", source=source_tag,
        ))

    emitted = False

    # (a) Resource-level env-var walk (Lambda + ECS)
    emitted |= _walk_cfn_resources_for_env(
        parsed=parsed, stage_id=stage_id, source_tag=source_tag,
        nodes=nodes, edges=edges,
    )

    # (b) Text-level s3:// + 3victors bucket literals
    for m in _S3_LITERAL_RE.finditer(text):
        bucket, prefix = m.group(1), (m.group(2) or "")
        nodes.append(Node(
            kind="s3_prefix", name=canonical_s3_prefix(bucket, prefix),
            source=source_tag,
        ))
        edges.append(Edge(
            source_id=stage_id,
            target_id=node_id("s3_prefix", canonical_s3_prefix(bucket, prefix)),
            rel="writes",  # default assumption; properties/env-var pass refines
            weight=0.4,     # low confidence: no role annotation from raw literal
            source=source_tag,
        ))
        emitted = True

    for m in _ATP_BUCKET_RE.finditer(text):
        bucket = m.group(1)
        # Skip if already captured via s3:// pattern
        for env_name in expand_environment(bucket):
            name = canonical_s3_prefix(env_name, "")
            nodes.append(Node(kind="s3_prefix", name=name, source=source_tag))
            edges.append(Edge(
                source_id=stage_id,
                target_id=node_id("s3_prefix", name),
                rel="writes",
                weight=0.4,
                source=source_tag,
            ))
            emitted = True

    return emitted


def _stage_name_from_template(parsed: dict, *, fallback: str) -> str:
    """Extract a stage name from a CFN template. Tries:
      1. `Parameters.AppName.Default`
      2. `Parameters.ServiceName.Default`
      3. fallback (file stem)
    """
    params = parsed.get("Parameters") or {}
    if isinstance(params, dict):
        for key in ("AppName", "ServiceName", "FunctionName"):
            p = params.get(key)
            if isinstance(p, dict):
                default = p.get("Default")
                if isinstance(default, str) and default and default != "example-application":
                    return default
    return fallback


def _walk_cfn_resources_for_env(
    *,
    parsed: dict,
    stage_id: str,
    source_tag: str,
    nodes: list[Node],
    edges: list[Edge],
) -> bool:
    """Walk `Resources` looking for env-var definitions on Lambda / ECS
    tasks, extract reads/writes edges. Returns True if any emitted.
    """
    resources = parsed.get("Resources") or {}
    if not isinstance(resources, dict):
        return False
    emitted = False

    for res_body in resources.values():
        if not isinstance(res_body, dict):
            continue
        res_type = res_body.get("Type", "")
        props = res_body.get("Properties") or {}
        if not isinstance(props, dict):
            continue

        env_items: list[tuple[str, Any]] = []  # (key, value) pairs

        if res_type in (
            "AWS::Lambda::Function", "AWS::Serverless::Function",
        ):
            env = (props.get("Environment") or {}).get("Variables") or {}
            if isinstance(env, dict):
                env_items.extend((str(k), v) for k, v in env.items())

        elif res_type == "AWS::ECS::TaskDefinition":
            for container in props.get("ContainerDefinitions") or []:
                if not isinstance(container, dict):
                    continue
                for env in container.get("Environment") or []:
                    if isinstance(env, dict) and "Name" in env and "Value" in env:
                        env_items.append((str(env["Name"]), env["Value"]))

        if not env_items:
            continue

        for key, raw_val in env_items:
            if raw_val is None:
                continue
            val = str(raw_val) if not isinstance(raw_val, (list, dict)) else ""
            if not val:
                continue
            key_upper = key.upper()
            is_input = any(h in key_upper for h in _INPUT_KEY_HINTS)
            is_output = any(h in key_upper for h in _OUTPUT_KEY_HINTS)
            rel = "reads" if is_input else "writes" if is_output else None

            m = _S3_LITERAL_RE.search(val)
            if m:
                bucket, prefix = m.group(1), (m.group(2) or "")
                name = canonical_s3_prefix(bucket, prefix)
                nodes.append(Node(kind="s3_prefix", name=name, source=source_tag))
                edges.append(Edge(
                    source_id=stage_id, target_id=node_id("s3_prefix", name),
                    rel=rel or "writes", weight=0.8 if rel else 0.5,
                    source=source_tag,
                ))
                emitted = True
                continue

            if any(h in key_upper for h in _BUCKET_KEY_HINTS):
                for name_raw in expand_environment(val):
                    name = canonical_s3_prefix(name_raw, "")
                    nodes.append(Node(kind="s3_prefix", name=name, source=source_tag))
                    edges.append(Edge(
                        source_id=stage_id, target_id=node_id("s3_prefix", name),
                        rel=rel or "writes", weight=0.8 if rel else 0.5,
                        source=source_tag,
                    ))
                    emitted = True
                continue

            if any(h in key_upper for h in _TABLE_KEY_HINTS):
                m = _DOTTED_TABLE_RE.search(val)
                if m:
                    table = canonical_redshift_table(m.group(1))
                    nodes.append(Node(
                        kind="redshift_table", name=table, source=source_tag,
                    ))
                    edges.append(Edge(
                        source_id=stage_id,
                        target_id=node_id("redshift_table", table),
                        rel=rel or "writes", weight=0.8 if rel else 0.5,
                        source=source_tag,
                    ))
                    emitted = True

    return emitted


__all__ = ["discover", "DiscoveryResult"]
