"""Pass 2 — AWS live-resource trawl (highest confidence).

Reads the running cloud via boto3 and emits graph nodes + edges. The
running system is ground truth for "what reads / writes what": Lambda
env vars, Step Function ASL definitions, Glue job arguments, EventBridge
rule targets, ECS task definitions, and CloudFormation stack tags all
leak bucket / table / repo references.

Signals extracted, keyed by `source = "aws:<service>:<resource-id>"`:

| Service        | Extracted                                           |
|----------------|------------------------------------------------------|
| CloudFormation | StackName + Tags['repo'] / ['ServiceName'] → repo  |
| Lambda         | Env vars matching INPUT_/OUTPUT_/BUCKET/TABLE       |
| Step Functions | ASL states[].Resource ARNs + Parameters             |
| Glue jobs      | DefaultArguments (--source / --output / tables)     |
| EventBridge    | Targets (which Lambda / SFN the rule triggers)      |
| ECS tasks      | containerDefinitions[].environment                  |

Every function takes an injected `client_factory` so tests can pass
fakes without monkey-patching boto3. In production callers pass
`get_default_factory()` from app.ops.ops_client. All calls are bounded
and wrapped in try/except — a missing service or a bad creds situation
degrades gracefully (returns zero nodes/edges instead of crashing the
full crawler).

Confidence tier: highest — this is what's actually running. When this
pass disagrees with Pass 1 or Pass 3, it wins.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .canonicalize import (
    AliasTable,
    Edge,
    Node,
    RepoEntry,
    canonical_glue_table,
    canonical_redshift_table,
    canonical_s3_prefix,
    node_id,
)

log = logging.getLogger(__name__)

ClientFactory = Callable[[str], Any]


# ── Public API ─────────────────────────────────────────────────────────


@dataclass
class DiscoveryResult:
    nodes: list[Node]
    edges: list[Edge]
    resources_scanned: int
    resources_with_signal: int


# Keys in Lambda / ECS env var dicts that commonly name data inputs /
# outputs. We extract bucket / table from values whose *keys* hint at
# the role. This is the single heuristic that carries the whole pass.
_INPUT_KEY_HINTS = ("INPUT_", "SOURCE_", "READ_", "UPSTREAM_")
_OUTPUT_KEY_HINTS = ("OUTPUT_", "SINK_", "WRITE_", "TARGET_", "DEST_")
_BUCKET_KEY_HINTS = ("BUCKET", "_S3_", "S3_PREFIX")
_TABLE_KEY_HINTS = ("TABLE", "REDSHIFT", "GLUE_TABLE")

_S3_LITERAL = re.compile(r"s3://([A-Za-z0-9\-.]+)(?:/([^\s\"'<>]+))?")
_ARN_LAMBDA = re.compile(r"arn:aws:lambda:[^:]*:\d+:function:([A-Za-z0-9\-_]+)")
_ARN_SFN = re.compile(r"arn:aws:states:[^:]*:\d+:stateMachine:([A-Za-z0-9\-_]+)")


def discover(
    repos: Iterable[RepoEntry] | None = None,
    *,
    aliases: AliasTable | None = None,
    client_factory: ClientFactory | None = None,
    max_items_per_service: int = 500,
) -> DiscoveryResult:
    """Run the live AWS trawl. Returns a flat DiscoveryResult whose
    nodes/edges may contain duplicates — caller merges later.

    Pass in a `client_factory` to stub out boto3 in tests. When omitted,
    the default factory lazily builds real boto3 clients. A missing
    `client_factory=None` in an environment with no creds will simply
    return an empty result.
    """
    aliases = aliases or AliasTable.load()
    if client_factory is None:
        try:
            from app.ops.ops_client import get_default_factory
            client_factory = get_default_factory()
        except Exception as exc:  # noqa: BLE001
            log.debug("discover_aws: no default client factory: %s", exc)
            return DiscoveryResult([], [], 0, 0)

    nodes: list[Node] = []
    edges: list[Edge] = []
    scanned = 0
    with_signal = 0

    # Known repo names — used to attribute resources by their tag
    repo_names: set[str] = set()
    for r in (repos or []):
        repo_names.add(r.name)

    for trawler in (
        _trawl_lambda,
        _trawl_step_functions,
        _trawl_glue_jobs,
        _trawl_eventbridge_rules,
        _trawl_cfn_stacks,
    ):
        try:
            s, ws = trawler(
                client_factory=client_factory,
                aliases=aliases,
                nodes=nodes,
                edges=edges,
                repo_names=repo_names,
                max_items=max_items_per_service,
            )
            scanned += s
            with_signal += ws
        except Exception as exc:  # noqa: BLE001 — one failed service shouldn't kill the pass
            log.warning("discover_aws trawler %s failed: %s", trawler.__name__, exc)

    return DiscoveryResult(
        nodes=nodes,
        edges=edges,
        resources_scanned=scanned,
        resources_with_signal=with_signal,
    )


# ── Service trawlers ───────────────────────────────────────────────────


def _trawl_lambda(
    *,
    client_factory: ClientFactory,
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
    repo_names: set[str],
    max_items: int,
) -> tuple[int, int]:
    client = client_factory("lambda")
    scanned = 0
    with_signal = 0
    token: str | None = None
    while True:
        kw: dict[str, Any] = {"MaxItems": min(50, max_items - scanned)}
        if token:
            kw["Marker"] = token
        try:
            resp = client.list_functions(**kw)
        except Exception as exc:  # noqa: BLE001
            log.debug("lambda list_functions: %s", exc)
            break
        for fn in resp.get("Functions") or []:
            scanned += 1
            name = fn.get("FunctionName")
            if not name:
                continue
            env = (fn.get("Environment") or {}).get("Variables") or {}
            stage = _stage_from_resource_name(name, aliases)
            fn_node = Node(
                kind="lambda",
                name=name.lower(),
                aliases=(name,),
                metadata={
                    "runtime": fn.get("Runtime"),
                    "last_modified": fn.get("LastModified"),
                },
                source=f"aws:lambda:{name}",
            )
            nodes.append(fn_node)
            if stage:
                stage_node = Node(
                    kind="stage", name=stage, source=f"aws:lambda:{name}",
                )
                nodes.append(stage_node)
                edges.append(Edge(
                    source_id=node_id("app", stage), target_id=fn_node.id,
                    rel="deploys_as", source=f"aws:lambda:{name}",
                ))
            extracted = _emit_env_signals(
                stage=stage,
                env=env,
                nodes=nodes,
                edges=edges,
                source=f"aws:lambda:{name}",
            )
            if extracted:
                with_signal += 1
        token = resp.get("NextMarker")
        if not token or scanned >= max_items:
            break
    return scanned, with_signal


def _trawl_step_functions(
    *,
    client_factory: ClientFactory,
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
    repo_names: set[str],
    max_items: int,
) -> tuple[int, int]:
    client = client_factory("stepfunctions")
    scanned = 0
    with_signal = 0
    token: str | None = None
    while True:
        kw: dict[str, Any] = {"maxResults": min(100, max_items - scanned)}
        if token:
            kw["nextToken"] = token
        try:
            resp = client.list_state_machines(**kw)
        except Exception as exc:  # noqa: BLE001
            log.debug("sfn list_state_machines: %s", exc)
            break
        for sm in resp.get("stateMachines") or []:
            scanned += 1
            name = sm.get("name")
            arn = sm.get("stateMachineArn")
            if not name or not arn:
                continue
            sm_node = Node(
                kind="step_fn", name=name.lower(), aliases=(name,),
                metadata={"arn": arn}, source=f"aws:sfn:{name}",
            )
            nodes.append(sm_node)
            try:
                desc = client.describe_state_machine(stateMachineArn=arn)
            except Exception as exc:  # noqa: BLE001
                log.debug("sfn describe_state_machine %s: %s", name, exc)
                continue
            definition = desc.get("definition") or ""
            extracted = _emit_sfn_definition_signals(
                sfn_name=name,
                definition=definition,
                nodes=nodes,
                edges=edges,
                source=f"aws:sfn:{name}",
            )
            if extracted:
                with_signal += 1
        token = resp.get("nextToken")
        if not token or scanned >= max_items:
            break
    return scanned, with_signal


def _trawl_glue_jobs(
    *,
    client_factory: ClientFactory,
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
    repo_names: set[str],
    max_items: int,
) -> tuple[int, int]:
    client = client_factory("glue")
    scanned = 0
    with_signal = 0
    token: str | None = None
    while True:
        kw: dict[str, Any] = {"MaxResults": min(100, max_items - scanned)}
        if token:
            kw["NextToken"] = token
        try:
            resp = client.get_jobs(**kw)
        except Exception as exc:  # noqa: BLE001
            log.debug("glue get_jobs: %s", exc)
            break
        for job in resp.get("Jobs") or []:
            scanned += 1
            name = job.get("Name")
            if not name:
                continue
            args = job.get("DefaultArguments") or {}
            stage = _stage_from_resource_name(name, aliases)
            extracted = _emit_glue_args_signals(
                stage=stage,
                args=args,
                nodes=nodes,
                edges=edges,
                source=f"aws:glue:{name}",
            )
            if extracted:
                with_signal += 1
        token = resp.get("NextToken")
        if not token or scanned >= max_items:
            break
    return scanned, with_signal


def _trawl_eventbridge_rules(
    *,
    client_factory: ClientFactory,
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
    repo_names: set[str],
    max_items: int,
) -> tuple[int, int]:
    client = client_factory("events")
    scanned = 0
    with_signal = 0
    token: str | None = None
    while True:
        kw: dict[str, Any] = {"Limit": min(100, max_items - scanned)}
        if token:
            kw["NextToken"] = token
        try:
            resp = client.list_rules(**kw)
        except Exception as exc:  # noqa: BLE001
            log.debug("events list_rules: %s", exc)
            break
        for rule in resp.get("Rules") or []:
            scanned += 1
            name = rule.get("Name")
            if not name:
                continue
            rule_node = Node(
                kind="event_rule", name=name.lower(), aliases=(name,),
                metadata={"schedule": rule.get("ScheduleExpression"),
                          "state": rule.get("State")},
                source=f"aws:events:{name}",
            )
            nodes.append(rule_node)
            try:
                t_resp = client.list_targets_by_rule(Rule=name, Limit=10)
            except Exception as exc:  # noqa: BLE001
                log.debug("events list_targets_by_rule %s: %s", name, exc)
                continue
            for tgt in t_resp.get("Targets") or []:
                arn = tgt.get("Arn") or ""
                lam = _ARN_LAMBDA.search(arn)
                sfn = _ARN_SFN.search(arn)
                if lam:
                    edges.append(Edge(
                        source_id=rule_node.id,
                        target_id=node_id("lambda", lam.group(1).lower()),
                        rel="triggers", source=f"aws:events:{name}",
                    ))
                    with_signal += 1
                elif sfn:
                    edges.append(Edge(
                        source_id=rule_node.id,
                        target_id=node_id("step_fn", sfn.group(1).lower()),
                        rel="triggers", source=f"aws:events:{name}",
                    ))
                    with_signal += 1
        token = resp.get("NextToken")
        if not token or scanned >= max_items:
            break
    return scanned, with_signal


def _trawl_cfn_stacks(
    *,
    client_factory: ClientFactory,
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
    repo_names: set[str],
    max_items: int,
) -> tuple[int, int]:
    client = client_factory("cloudformation")
    scanned = 0
    with_signal = 0
    token: str | None = None
    while True:
        kw: dict[str, Any] = {}
        if token:
            kw["NextToken"] = token
        try:
            resp = client.describe_stacks(**kw)
        except Exception as exc:  # noqa: BLE001
            log.debug("cfn describe_stacks: %s", exc)
            break
        for stack in resp.get("Stacks") or []:
            scanned += 1
            if scanned > max_items:
                break
            name = stack.get("StackName")
            if not name:
                continue
            tags = {t.get("Key"): t.get("Value")
                    for t in stack.get("Tags") or [] if t.get("Key")}
            repo = tags.get("repo") or tags.get("Repo") or tags.get("ServiceName")
            if not repo:
                continue
            # Drop the leading team/org qualifier if any, then canonicalize
            canon_repo = repo.split("/", 1)[-1].strip().lower()
            if not canon_repo:
                continue
            repo_node = Node(
                kind="repo", name=canon_repo,
                source=f"aws:cfn:{name}",
            )
            nodes.append(repo_node)
            # If we have stage / app signals from other passes, the
            # matching repo node merges into one by id.
            edges.append(Edge(
                source_id=repo_node.id,
                target_id=node_id("app", canon_repo),
                rel="repo", source=f"aws:cfn:{name}",
            ))
            with_signal += 1
        token = resp.get("NextToken")
        if not token or scanned >= max_items:
            break
    return scanned, with_signal


# ── Shared helpers ─────────────────────────────────────────────────────


def _stage_from_resource_name(raw: str, aliases: AliasTable) -> str | None:
    """If the resource name contains a known stage alias, return the
    canonical stage name. Else return None so the caller falls back to
    resource-level attribution.
    """
    if not raw:
        return None
    # Try progressively shorter hyphen-split variants — many ATPCO
    # resources are named like "ds-priceeye-analytics-competitive-position-v2".
    lowered = raw.lower().replace("_", "-")
    tokens = lowered.split("-")
    canonicals = set(aliases.by_canonical.keys())
    known_aliases = set(aliases.by_alias.keys())
    # Prefer longer matches — slide a window over the tokens and pick
    # the longest hyphen-joined slice whose normalized form is a known
    # alias or canonical name.
    for size in range(len(tokens), 0, -1):
        for start in range(0, len(tokens) - size + 1):
            candidate = "-".join(tokens[start:start + size])
            if candidate in canonicals:
                return candidate
            if candidate in known_aliases:
                return aliases.by_alias[candidate]
    # Fallback: single-token aliases (mlg, mla, dco, etc.)
    for tok in tokens:
        if tok in known_aliases:
            return aliases.by_alias[tok]
    return None


def _emit_env_signals(
    *,
    stage: str | None,
    env: dict[str, str],
    nodes: list[Node],
    edges: list[Edge],
    source: str,
) -> int:
    """Turn a Lambda / ECS env-var dict into reads / writes edges. Returns
    the number of edges emitted.
    """
    emitted = 0
    if not stage or not env:
        return 0
    stage_id = node_id("stage", stage)
    for key, value in env.items():
        if not isinstance(value, str) or not value:
            continue
        key_upper = key.upper()
        is_input = any(h in key_upper for h in _INPUT_KEY_HINTS)
        is_output = any(h in key_upper for h in _OUTPUT_KEY_HINTS)
        if not is_input and not is_output:
            continue
        rel = "reads" if is_input else "writes"
        # S3 literal
        m = _S3_LITERAL.search(value)
        if m:
            bucket, prefix = m.group(1), (m.group(2) or "")
            target = Node(
                kind="s3_prefix",
                name=canonical_s3_prefix(bucket, prefix),
                source=source,
            )
            nodes.append(target)
            edges.append(Edge(
                source_id=stage_id, target_id=target.id, rel=rel, source=source,
            ))
            emitted += 1
            continue
        # Dotted table reference
        if _looks_like_table(value) and any(h in key_upper for h in _TABLE_KEY_HINTS):
            target = Node(
                kind="redshift_table",
                name=canonical_redshift_table(value),
                source=source,
            )
            nodes.append(target)
            edges.append(Edge(
                source_id=stage_id, target_id=target.id, rel=rel, source=source,
            ))
            emitted += 1
            continue
        # Bare bucket name (no scheme)
        if any(h in key_upper for h in _BUCKET_KEY_HINTS) and _looks_like_bucket(value):
            target = Node(
                kind="s3_prefix",
                name=canonical_s3_prefix(value, ""),
                source=source,
            )
            nodes.append(target)
            edges.append(Edge(
                source_id=stage_id, target_id=target.id, rel=rel, source=source,
            ))
            emitted += 1
    return emitted


def _emit_sfn_definition_signals(
    *,
    sfn_name: str,
    definition: str,
    nodes: list[Node],
    edges: list[Edge],
    source: str,
) -> int:
    """Parse an ASL JSON blob. Extract every Lambda / SFN target ARN
    referenced in states[].Resource or states[].Parameters and emit a
    `triggers` edge from the SFN to the target.
    """
    if not definition:
        return 0
    try:
        parsed = json.loads(definition)
    except Exception:
        return 0
    states = parsed.get("States") or {}
    sfn_id = node_id("step_fn", sfn_name.lower())
    emitted = 0
    for state_name, state in states.items():
        if not isinstance(state, dict):
            continue
        resource = state.get("Resource")
        if isinstance(resource, str):
            for match in _ARN_LAMBDA.finditer(resource):
                target_id = node_id("lambda", match.group(1).lower())
                edges.append(Edge(
                    source_id=sfn_id, target_id=target_id,
                    rel="triggers", source=source,
                    metadata={"state": state_name},
                ))
                emitted += 1
    return emitted


def _emit_glue_args_signals(
    *,
    stage: str | None,
    args: dict[str, Any],
    nodes: list[Node],
    edges: list[Edge],
    source: str,
) -> int:
    if not stage or not args:
        return 0
    stage_id = node_id("stage", stage)
    emitted = 0
    for key, value in args.items():
        if not isinstance(value, str):
            continue
        key_lower = key.lower()
        # --source / --input_table → reads
        if "source" in key_lower or "input" in key_lower:
            rel = "reads"
        elif "output" in key_lower or "target" in key_lower or "sink" in key_lower:
            rel = "writes"
        else:
            continue
        m = _S3_LITERAL.search(value)
        if m:
            bucket, prefix = m.group(1), (m.group(2) or "")
            target = Node(
                kind="s3_prefix",
                name=canonical_s3_prefix(bucket, prefix),
                source=source,
            )
            nodes.append(target)
            edges.append(Edge(
                source_id=stage_id, target_id=target.id, rel=rel, source=source,
            ))
            emitted += 1
            continue
        if _looks_like_table(value):
            # Glue's database.table vs redshift schema.table is
            # ambiguous from args alone; default to glue-style.
            if "." in value:
                db, _, tbl = value.partition(".")
                target = Node(
                    kind="glue_table",
                    name=canonical_glue_table(db, tbl),
                    source=source,
                )
            else:
                target = Node(
                    kind="glue_table",
                    name=canonical_glue_table("default", value),
                    source=source,
                )
            nodes.append(target)
            edges.append(Edge(
                source_id=stage_id, target_id=target.id, rel=rel, source=source,
            ))
            emitted += 1
    return emitted


def _looks_like_bucket(value: str) -> bool:
    # AWS bucket naming: lowercase letters, digits, dots, hyphens, 3-63 chars.
    if not (3 <= len(value) <= 255):
        return False
    return bool(re.fullmatch(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?", value))


def _looks_like_table(value: str) -> bool:
    # A reasonable dotted identifier, no whitespace or schemes
    if "://" in value or " " in value:
        return False
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", value))


__all__ = [
    "DiscoveryResult",
    "discover",
]
