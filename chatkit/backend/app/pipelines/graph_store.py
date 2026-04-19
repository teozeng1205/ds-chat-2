"""Persistent graph store — SQLite-backed.

Adjacency-list tables live in their own SQLite file under
`app/.data/ds-chat-pipelines.sqlite`. Keeping them separate from the
existing `knowledge.sqlite` lets the graph be rebuilt / wiped without
disturbing the lexical KB or the semantic index.

Public API:
  - GraphStore(db_path) — open / create schema
  - store.upsert(nodes, edges) — merge-persist
  - store.clear() — wipe all rows (used by full rebuilds)
  - store.neighbors(node_id, direction, depth) — BFS traversal
  - store.resolve(raw) — free-text → node_id using alias table
  - store.get_node(node_id) / get_edges(node_id)
  - store.stats() — for drift reports

The traversal API (`neighbors`) is what `trace_pipeline` uses.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .canonicalize import AliasTable, Edge, Node, node_id

log = logging.getLogger(__name__)


DEFAULT_GRAPH_DB_ENV = "DS_CHAT_PIPELINE_GRAPH_DB"
DEFAULT_GRAPH_DB_FILENAME = "ds-chat-pipelines.sqlite"


def default_graph_db_path() -> Path:
    env = os.environ.get(DEFAULT_GRAPH_DB_ENV)
    if env:
        return Path(env).expanduser().resolve()
    backend_root = Path(__file__).resolve().parents[2]
    data_dir = backend_root / "app" / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (data_dir / DEFAULT_GRAPH_DB_FILENAME).resolve()


Direction = Literal["upstream", "downstream", "both"]


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS kb_graph_nodes (
        id           TEXT PRIMARY KEY,
        kind         TEXT NOT NULL,
        name         TEXT NOT NULL,
        aliases      TEXT,
        metadata     TEXT,
        source       TEXT NOT NULL,
        updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kb_graph_edges (
        source_id    TEXT NOT NULL,
        target_id    TEXT NOT NULL,
        rel          TEXT NOT NULL,
        weight       REAL NOT NULL DEFAULT 1.0,
        source       TEXT NOT NULL,
        metadata     TEXT,
        updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        PRIMARY KEY (source_id, target_id, rel)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_graph_nodes_kind ON kb_graph_nodes(kind)",
    "CREATE INDEX IF NOT EXISTS idx_graph_nodes_name ON kb_graph_nodes(name)",
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON kb_graph_edges(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_tgt ON kb_graph_edges(target_id)",
]


_UPSTREAM_RELS = frozenset({"reads"})      # follow edge from target → source
_DOWNSTREAM_RELS = frozenset({"writes"})   # follow edge from source → target


@dataclass
class GraphHit:
    id: str
    kind: str
    name: str
    aliases: list[str]
    metadata: dict
    source: str


@dataclass
class EdgeHit:
    source_id: str
    target_id: str
    rel: str
    weight: float
    source: str
    metadata: dict


class GraphStore:
    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or default_graph_db_path()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kb_graph_edges")
            self._conn.execute("DELETE FROM kb_graph_nodes")
            self._conn.commit()

    # ── Writes ────────────────────────────────────────────────────────

    def upsert(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> tuple[int, int]:
        n_count, e_count = 0, 0
        now = time.time()
        _ = now
        with self._lock:
            for node in nodes:
                self._conn.execute(
                    """
                    INSERT INTO kb_graph_nodes(id, kind, name, aliases, metadata, source)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        aliases    = excluded.aliases,
                        metadata   = excluded.metadata,
                        source     = excluded.source,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    """,
                    (
                        node.id,
                        node.kind,
                        node.name,
                        json.dumps(list(node.aliases), ensure_ascii=True),
                        json.dumps(node.metadata or {}, ensure_ascii=True, default=str),
                        node.source or "",
                    ),
                )
                n_count += 1
            for edge in edges:
                self._conn.execute(
                    """
                    INSERT INTO kb_graph_edges(source_id, target_id, rel, weight, source, metadata)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, rel) DO UPDATE SET
                        weight     = MAX(kb_graph_edges.weight, excluded.weight),
                        source     = excluded.source,
                        metadata   = excluded.metadata,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    """,
                    (
                        edge.source_id,
                        edge.target_id,
                        edge.rel,
                        float(edge.weight or 1.0),
                        edge.source or "",
                        json.dumps(edge.metadata or {}, ensure_ascii=True, default=str),
                    ),
                )
                e_count += 1
            self._conn.commit()
        return n_count, e_count

    # ── Reads ─────────────────────────────────────────────────────────

    def get_node(self, nid: str) -> GraphHit | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, kind, name, aliases, metadata, source FROM kb_graph_nodes WHERE id = ?",
                (nid,),
            ).fetchone()
        if not row:
            return None
        return _hit_from_row(row)

    def get_edges(self, nid: str, *, direction: Direction = "both") -> list[EdgeHit]:
        with self._lock:
            sql, params = _edges_query(nid, direction)
            rows = self._conn.execute(sql, params).fetchall()
        return [
            EdgeHit(
                source_id=r[0], target_id=r[1], rel=r[2], weight=r[3],
                source=r[4] or "", metadata=_try_json(r[5]),
            )
            for r in rows
        ]

    def stats(self) -> dict[str, int]:
        with self._lock:
            n_total = self._conn.execute("SELECT COUNT(*) FROM kb_graph_nodes").fetchone()[0]
            e_total = self._conn.execute("SELECT COUNT(*) FROM kb_graph_edges").fetchone()[0]
            by_kind = dict(
                self._conn.execute(
                    "SELECT kind, COUNT(*) FROM kb_graph_nodes GROUP BY kind"
                ).fetchall()
            )
            by_rel = dict(
                self._conn.execute(
                    "SELECT rel, COUNT(*) FROM kb_graph_edges GROUP BY rel"
                ).fetchall()
            )
        return {"total_nodes": int(n_total), "total_edges": int(e_total),
                "by_kind": {k: int(v) for k, v in by_kind.items()},
                "by_rel":  {k: int(v) for k, v in by_rel.items()}}

    # ── Traversal ─────────────────────────────────────────────────────

    def neighbors(
        self,
        nid: str,
        *,
        direction: Direction = "both",
        depth: int = 3,
        rels: set[str] | None = None,
    ) -> dict:
        """BFS from `nid`. Returns a dict with:

        - origin: GraphHit | None
        - nodes: list[GraphHit] (everything visited, incl. origin)
        - edges: list[EdgeHit]  (every edge traversed, with direction intact)
        - reached_by_depth: {1: [...], 2: [...], …}
        """
        origin = self.get_node(nid)
        if origin is None:
            return {"origin": None, "nodes": [], "edges": [], "reached_by_depth": {}}

        visited: set[str] = {nid}
        seen_edges: set[tuple[str, str, str]] = set()
        nodes_out: dict[str, GraphHit] = {nid: origin}
        edges_out: list[EdgeHit] = []
        reached_by_depth: dict[int, list[str]] = {}

        frontier: deque[tuple[str, int]] = deque([(nid, 0)])
        while frontier:
            cur, d = frontier.popleft()
            if d >= depth:
                continue
            for e in self.get_edges(cur, direction=direction):
                if rels and e.rel not in rels:
                    continue
                key = (e.source_id, e.target_id, e.rel)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                # Which end did we just travel to?
                other = e.target_id if cur == e.source_id else e.source_id
                edges_out.append(e)
                if other not in visited:
                    visited.add(other)
                    hit = self.get_node(other)
                    if hit is not None:
                        nodes_out[other] = hit
                    reached_by_depth.setdefault(d + 1, []).append(other)
                    frontier.append((other, d + 1))

        return {
            "origin": origin,
            "nodes": list(nodes_out.values()),
            "edges": edges_out,
            "reached_by_depth": reached_by_depth,
        }

    # ── Alias resolution ──────────────────────────────────────────────

    def resolve(self, raw: str, *, aliases: AliasTable | None = None) -> str | None:
        """Turn free text → node_id. Tries:

          1. exact match on id / name
          2. aliases table lookup then name match
          3. contains-match on name / aliases
        """
        if not raw:
            return None
        needle = raw.strip().lower()
        with self._lock:
            # Exact id
            row = self._conn.execute(
                "SELECT id FROM kb_graph_nodes WHERE id = ? LIMIT 1", (needle,)
            ).fetchone()
            if row:
                return row[0]
            # Exact name
            row = self._conn.execute(
                "SELECT id FROM kb_graph_nodes WHERE name = ? ORDER BY kind LIMIT 1", (needle,)
            ).fetchone()
            if row:
                return row[0]

            # Alias resolution
            alias = (aliases or AliasTable.load()).resolve(needle)
            row = self._conn.execute(
                "SELECT id FROM kb_graph_nodes WHERE name = ? ORDER BY kind LIMIT 1", (alias,)
            ).fetchone()
            if row:
                return row[0]

            # Contains-match on name (e.g. "market_level_anomalies_v4" should
            # hit the redshift_table node whose name ends with that table name).
            row = self._conn.execute(
                "SELECT id FROM kb_graph_nodes WHERE name LIKE ? ORDER BY length(name) LIMIT 1",
                (f"%{needle}%",),
            ).fetchone()
            if row:
                return row[0]

            # Contains-match on aliases JSON
            row = self._conn.execute(
                "SELECT id FROM kb_graph_nodes WHERE aliases LIKE ? ORDER BY length(aliases) LIMIT 1",
                (f'%"{needle}"%',),
            ).fetchone()
            return row[0] if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── Helpers ────────────────────────────────────────────────────────────


def _edges_query(nid: str, direction: Direction) -> tuple[str, tuple]:
    base = "SELECT source_id, target_id, rel, weight, source, metadata FROM kb_graph_edges WHERE "
    if direction == "upstream":
        # Upstream = this node is the READER / CONSUMER: traverse edges pointing
        # INTO this node (target_id = nid for "writes") OR outgoing "reads".
        return (base + "(target_id = ? AND rel = 'writes') OR (source_id = ? AND rel = 'reads')",
                (nid, nid))
    if direction == "downstream":
        return (base + "(source_id = ? AND rel = 'writes') OR (target_id = ? AND rel = 'reads')",
                (nid, nid))
    return (base + "source_id = ? OR target_id = ?", (nid, nid))


def _try_json(s: str | None) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _hit_from_row(row) -> GraphHit:
    return GraphHit(
        id=row[0], kind=row[1], name=row[2],
        aliases=_try_json(row[3]).get("__list__") or (json.loads(row[3]) if row[3] else []),
        metadata=_try_json(row[4]),
        source=row[5] or "",
    )


# Patch _hit_from_row to deal with aliases encoded as a bare JSON list
def _hit_from_row(row) -> GraphHit:  # noqa: F811 — override
    aliases_raw = row[3] or "[]"
    try:
        aliases = json.loads(aliases_raw)
        if not isinstance(aliases, list):
            aliases = []
    except Exception:
        aliases = []
    return GraphHit(
        id=row[0], kind=row[1], name=row[2],
        aliases=list(aliases),
        metadata=_try_json(row[4]),
        source=row[5] or "",
    )


__all__ = [
    "Direction",
    "EdgeHit",
    "GraphHit",
    "GraphStore",
    "default_graph_db_path",
]
