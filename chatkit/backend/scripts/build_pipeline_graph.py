#!/usr/bin/env python3
"""Build the pipeline lineage graph — Day 1 scope: Pass 1 only.

Reads repos from `app/pipelines/repos.yaml`, runs `discover_configs`
over each one's declared config_roots, merges nodes + edges, and:

  1. UPSERTs everything into the SQLite graph store at
     `app/.data/ds-chat-pipelines.sqlite`.
  2. Writes a human-readable canonical `pipelines.json` under
     `app/investigation/knowledge/pipelines.json` that's checked
     into the repo so PR diffs surface graph changes.

Later passes (AWS live trawl, code patterns, LLM summary, ASCII DAG
mining) will plug into the same script.

Usage:
    .venv/bin/python scripts/build_pipeline_graph.py
    .venv/bin/python scripts/build_pipeline_graph.py --dry-run
    .venv/bin/python scripts/build_pipeline_graph.py --only-repo ds-priceeye-analytics
    .venv/bin/python scripts/build_pipeline_graph.py --clear   (wipe DB before rebuild)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.pipelines.canonicalize import (  # noqa: E402
    AliasTable,
    Edge,
    Node,
    RepoEntry,
    load_repos,
    merge_edges,
    merge_nodes,
)
from app.pipelines.discover_aws import discover as discover_aws  # noqa: E402
from app.pipelines.discover_code import discover as discover_code  # noqa: E402
from app.pipelines.discover_configs import discover as discover_configs  # noqa: E402
from app.pipelines.discover_docs import discover as discover_docs  # noqa: E402
from app.pipelines.graph_store import GraphStore  # noqa: E402

log = logging.getLogger(__name__)


def _dedupe_nodes(nodes: Iterable[Node]) -> list[Node]:
    by_id: dict[str, Node] = {}
    for n in nodes:
        if n.id in by_id:
            by_id[n.id] = merge_nodes(by_id[n.id], n)
        else:
            by_id[n.id] = n
    return list(by_id.values())


def _to_jsonable(node: Node) -> dict:
    return {
        "id": node.id,
        "kind": node.kind,
        "name": node.name,
        "aliases": list(node.aliases),
        "metadata": node.metadata,
        "source": node.source,
    }


def _edge_jsonable(edge: Edge) -> dict:
    return {
        "source": edge.source_id,
        "target": edge.target_id,
        "rel": edge.rel,
        "weight": edge.weight,
        "provenance": edge.source,
        **({"metadata": edge.metadata} if edge.metadata else {}),
    }


def write_canonical_json(path: Path, nodes: list[Node], edges: list[Edge]) -> None:
    """Write a human-readable pipelines.json. Nodes are grouped by kind
    and sorted; edges are sorted by (source, target, rel) so the file
    diffs cleanly across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes_sorted = sorted(nodes, key=lambda n: (n.kind, n.name))
    edges_sorted = sorted(edges, key=lambda e: (e.source_id, e.target_id, e.rel))

    grouped_nodes: dict[str, list] = {}
    for n in nodes_sorted:
        grouped_nodes.setdefault(n.kind, []).append(_to_jsonable(n))

    payload = {
        "_format_version": 1,
        "_stats": {
            "nodes": len(nodes_sorted),
            "edges": len(edges_sorted),
            "by_kind": {k: len(v) for k, v in grouped_nodes.items()},
        },
        "nodes": grouped_nodes,
        "edges": [_edge_jsonable(e) for e in edges_sorted],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build(
    repos: list[RepoEntry],
    *,
    clear: bool,
    dry_run: bool,
    json_out: Path,
    skip_code: bool = False,
    skip_aws: bool = True,
) -> dict:
    aliases = AliasTable.load()

    # Pass 1 — configs
    pass1 = discover_configs(repos, aliases=aliases)
    all_nodes = list(pass1.nodes)
    all_edges = list(pass1.edges)
    passes_run = ["configs"]
    per_pass = {
        "configs": {"files_scanned": pass1.files_scanned,
                    "files_with_signal": pass1.files_with_signal,
                    "nodes": len(pass1.nodes), "edges": len(pass1.edges)},
    }

    # Pass 3 — code patterns
    if not skip_code:
        pass3 = discover_code(repos, aliases=aliases)
        all_nodes.extend(pass3.nodes)
        all_edges.extend(pass3.edges)
        passes_run.append("code")
        per_pass["code"] = {"files_scanned": pass3.files_scanned,
                            "files_with_signal": pass3.files_with_signal,
                            "nodes": len(pass3.nodes), "edges": len(pass3.edges)}

    # Pass 2 — live AWS trawl. Opt-in because it needs credentials and
    # takes longer than the local passes.
    if not skip_aws:
        try:
            pass2 = discover_aws(repos=repos, aliases=aliases)
            all_nodes.extend(pass2.nodes)
            all_edges.extend(pass2.edges)
            passes_run.append("aws")
            per_pass["aws"] = {
                "resources_scanned": pass2.resources_scanned,
                "resources_with_signal": pass2.resources_with_signal,
                "nodes": len(pass2.nodes), "edges": len(pass2.edges),
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("Pass 2 (AWS trawl) failed — continuing without it: %s", exc)

    # Pass 5 — ASCII DAG / prose mining from human-authored docs
    pass5 = discover_docs(repos=repos, aliases=aliases)
    all_nodes.extend(pass5.nodes)
    all_edges.extend(pass5.edges)
    passes_run.append("docs")
    per_pass["docs"] = {"files_scanned": pass5.files_scanned,
                        "files_with_signal": pass5.files_with_signal,
                        "nodes": len(pass5.nodes), "edges": len(pass5.edges)}

    nodes = _dedupe_nodes(all_nodes)
    edges = merge_edges(all_edges)

    summary = {
        "passes_run": passes_run,
        "per_pass": per_pass,
        "nodes": len(nodes),
        "edges": len(edges),
        "by_kind": Counter(n.kind for n in nodes),
        "by_rel":  Counter(e.rel  for e in edges),
    }

    if dry_run:
        log.info("dry-run: not writing pipelines.json or graph store")
        return summary

    # Persist to SQLite
    store = GraphStore()
    if clear:
        store.clear()
    store.upsert(nodes, edges)
    store.close()

    # Persist canonical JSON
    write_canonical_json(json_out, nodes, edges)
    summary["json_path"] = str(json_out)
    summary["db_path"] = "app/.data/ds-chat-pipelines.sqlite"
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="don't write JSON or SQLite")
    parser.add_argument("--clear", action="store_true", help="wipe graph DB before rebuild")
    parser.add_argument("--only-repo", action="append", default=None,
                        help="scan only the named repo(s) from repos.yaml")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="override output path (default: app/investigation/knowledge/pipelines.json)")
    parser.add_argument("--skip-code", action="store_true",
                        help="skip the slow Pass 3 (code pattern scan)")
    parser.add_argument("--with-aws", action="store_true",
                        help="run Pass 2 (live AWS trawl; needs credentials)")
    args = parser.parse_args(argv)

    repos = load_repos()
    if args.only_repo:
        want = set(args.only_repo)
        repos = [r for r in repos if r.name in want]
    if not repos:
        log.error("no repos matched; nothing to do")
        return 1

    json_out = args.json_out or (
        BACKEND_ROOT / "app" / "investigation" / "knowledge" / "pipelines.json"
    )

    summary = build(
        repos, clear=args.clear, dry_run=args.dry_run, json_out=json_out,
        skip_code=args.skip_code, skip_aws=not args.with_aws,
    )
    print(json.dumps({k: (dict(v) if isinstance(v, Counter) else v) for k, v in summary.items()},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
