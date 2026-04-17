"""Semantic index over embedded text chunks.

SQLite-backed store of (id, text, embedding, metadata) tuples with
cosine-similarity search. Deliberately small and dependency-light —
numpy only, no DuckDB-VSS or FAISS — because the KB is at most a few
thousand chunks, well within numpy's comfort zone for a linear scan
over dense vectors.

API:
  - SemanticIndex(db_path, dim=None)
  - upsert(id, text, embedding, metadata=None, kind=None)
  - search(query_embedding, top_k=8, kind=None, where=None) -> list[Hit]
  - hybrid_search(query_embedding, lexical_terms, top_k=8, ...) — blends
    cosine with a simple lexical overlap score
  - count() / stats() / delete(id) / clear()

Embeddings are serialized as little-endian float32 bytes; numpy loads
them zero-copy for search.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id           TEXT PRIMARY KEY,
        kind         TEXT,
        text         TEXT NOT NULL,
        embedding    BLOB NOT NULL,
        dim          INTEGER NOT NULL,
        metadata     TEXT,
        updated_at   REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind)",
]


@dataclass
class Hit:
    id: str
    kind: Optional[str]
    text: str
    score: float
    cosine: float
    lexical: float
    metadata: dict[str, Any] = field(default_factory=dict)


_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _pack(vec: Iterable[float]) -> bytes:
    vals = list(vec)
    return struct.pack(f"<{len(vals)}f", *vals)


def _unpack(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec)) or 1.0


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class SemanticIndex:
    """SQLite-backed semantic index with cosine + hybrid search.

    `dim` is enforced on upsert. If omitted in the constructor, it is
    pinned by the first upsert; subsequent mismatches raise ValueError.
    """

    def __init__(self, db_path: Path, dim: int | None = None) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._dim = dim
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()

    # ── Writes ──

    def upsert(
        self,
        id: str,
        text: str,
        embedding: Iterable[float],
        *,
        kind: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec = list(embedding)
        if not vec:
            raise ValueError("embedding is empty")
        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            raise ValueError(f"embedding dim mismatch: got {len(vec)}, expected {self._dim}")
        blob = _pack(vec)
        meta = json.dumps(metadata or {}, ensure_ascii=True, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO chunks(id, kind, text, embedding, dim, metadata, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    text=excluded.text,
                    embedding=excluded.embedding,
                    dim=excluded.dim,
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at
                """,
                (id, kind, text, blob, len(vec), meta, time.time()),
            )
            self._conn.commit()

    def delete(self, id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM chunks WHERE id = ?", (id,))
            self._conn.commit()
        return cur.rowcount > 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.commit()

    # ── Reads ──

    def count(self, kind: str | None = None) -> int:
        with self._lock:
            if kind is None:
                row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM chunks WHERE kind = ?", (kind,)).fetchone()
        return int(row[0])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT COALESCE(kind,'?') AS kind, COUNT(*) FROM chunks GROUP BY kind ORDER BY COUNT(*) DESC"
            ).fetchall()
        return {"total": sum(r[1] for r in rows), "by_kind": {r[0]: int(r[1]) for r in rows}, "dim": self._dim}

    def search(
        self,
        query_embedding: Iterable[float],
        *,
        top_k: int = 8,
        kind: str | None = None,
        min_score: float | None = None,
    ) -> list[Hit]:
        q = list(query_embedding)
        if not q:
            return []
        return self._search_blended(q, lexical_terms=None, top_k=top_k, kind=kind,
                                    w_cos=1.0, w_lex=0.0, min_score=min_score)

    def hybrid_search(
        self,
        query_embedding: Iterable[float],
        lexical_terms: Iterable[str],
        *,
        top_k: int = 8,
        kind: str | None = None,
        w_cos: float = 0.7,
        w_lex: float = 0.3,
        min_score: float | None = None,
    ) -> list[Hit]:
        q = list(query_embedding)
        return self._search_blended(q, lexical_terms=list(lexical_terms), top_k=top_k,
                                    kind=kind, w_cos=w_cos, w_lex=w_lex, min_score=min_score)

    def _search_blended(
        self,
        q: list[float],
        *,
        lexical_terms: list[str] | None,
        top_k: int,
        kind: str | None,
        w_cos: float,
        w_lex: float,
        min_score: float | None,
    ) -> list[Hit]:
        q_norm = _norm(q)
        lex_terms = [t.lower() for t in lexical_terms] if lexical_terms else []

        with self._lock:
            if kind is None:
                rows = self._conn.execute(
                    "SELECT id, kind, text, embedding, dim, metadata FROM chunks"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, kind, text, embedding, dim, metadata FROM chunks WHERE kind = ?",
                    (kind,),
                ).fetchall()

        scored: list[Hit] = []
        for id_, rkind, text, blob, dim, meta in rows:
            if dim != len(q):
                continue
            vec = _unpack(blob, dim)
            denom = q_norm * _norm(vec)
            cosine = _dot(q, vec) / denom if denom else 0.0
            lexical = _lex_overlap(text, lex_terms) if lex_terms else 0.0
            score = w_cos * cosine + w_lex * lexical
            if min_score is not None and score < min_score:
                continue
            metadata = {}
            if meta:
                try:
                    metadata = json.loads(meta)
                except Exception:
                    metadata = {}
            scored.append(Hit(id=id_, kind=rkind, text=text, score=score,
                              cosine=cosine, lexical=lexical, metadata=metadata))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _lex_overlap(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    tokens = set(tokenize(text))
    if not tokens:
        return 0.0
    matches = sum(1 for t in terms if t in tokens)
    return matches / len(terms)
