"""SQLite store for KB V2 typed items, chunks, edges, and tasks."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .models import KnowledgeChunk, KnowledgeEdge, KnowledgeItem, TaskRecipe

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")
_SOURCE_WEIGHT = {
    "structured_snapshot": 1.35,
    "code_verified": 1.25,
    "live_verified": 1.5,
    "doc_hint": 0.45,
    "task_hint": 0.6,
}


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=True, sort_keys=True, default=str)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _source_type(item_metadata: dict[str, Any], chunk_metadata: dict[str, Any]) -> str:
    return str(
        chunk_metadata.get("source_type")
        or item_metadata.get("source_type")
        or chunk_metadata.get("authority")
        or item_metadata.get("authority")
        or "unknown"
    )


class KnowledgeStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kb_v2_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_v2_items (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_path TEXT,
                    metadata TEXT,
                    confidence REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_v2_chunks (
                    id TEXT PRIMARY KEY,
                    item_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_path TEXT,
                    heading TEXT,
                    citation TEXT,
                    metadata TEXT,
                    confidence REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_v2_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    rel TEXT NOT NULL,
                    source_path TEXT,
                    metadata TEXT,
                    confidence REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id, rel)
                );
                CREATE TABLE IF NOT EXISTS kb_v2_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    triggers TEXT NOT NULL,
                    tool_plan TEXT NOT NULL,
                    source_path TEXT,
                    metadata TEXT,
                    confidence REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_kb_v2_items_type ON kb_v2_items(type);
                CREATE INDEX IF NOT EXISTS idx_kb_v2_chunks_item ON kb_v2_chunks(item_id);
                CREATE INDEX IF NOT EXISTS idx_kb_v2_chunks_kind ON kb_v2_chunks(kind);
                CREATE INDEX IF NOT EXISTS idx_kb_v2_edges_source ON kb_v2_edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_kb_v2_edges_target ON kb_v2_edges(target_id);
                """
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            for table in ("kb_v2_edges", "kb_v2_chunks", "kb_v2_items", "kb_v2_tasks"):
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kb_v2_meta(key, value) VALUES(?, ?)",
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kb_v2_meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else None

    def upsert_items(self, items: Iterable[KnowledgeItem]) -> int:
        now = time.time()
        count = 0
        with self._lock:
            for item in items:
                self._conn.execute(
                    """
                    INSERT INTO kb_v2_items(id, type, name, title, summary, source_path, metadata, confidence, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        type=excluded.type,
                        name=excluded.name,
                        title=excluded.title,
                        summary=excluded.summary,
                        source_path=excluded.source_path,
                        metadata=excluded.metadata,
                        confidence=excluded.confidence,
                        updated_at=excluded.updated_at
                    """,
                    (
                        item.id,
                        item.type,
                        item.name,
                        item.title,
                        item.summary,
                        item.source_path,
                        _json_dumps(item.metadata),
                        float(item.confidence),
                        now,
                    ),
                )
                count += 1
            self._conn.commit()
        return count

    def upsert_chunks(self, chunks: Iterable[KnowledgeChunk]) -> int:
        now = time.time()
        count = 0
        with self._lock:
            for chunk in chunks:
                self._conn.execute(
                    """
                    INSERT INTO kb_v2_chunks(id, item_id, kind, text, source_path, heading, citation, metadata, confidence, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        item_id=excluded.item_id,
                        kind=excluded.kind,
                        text=excluded.text,
                        source_path=excluded.source_path,
                        heading=excluded.heading,
                        citation=excluded.citation,
                        metadata=excluded.metadata,
                        confidence=excluded.confidence,
                        updated_at=excluded.updated_at
                    """,
                    (
                        chunk.id,
                        chunk.item_id,
                        chunk.kind,
                        chunk.text,
                        chunk.source_path,
                        chunk.heading,
                        chunk.citation,
                        _json_dumps(chunk.metadata),
                        float(chunk.confidence),
                        now,
                    ),
                )
                count += 1
            self._conn.commit()
        return count

    def upsert_edges(self, edges: Iterable[KnowledgeEdge]) -> int:
        now = time.time()
        count = 0
        with self._lock:
            for edge in edges:
                self._conn.execute(
                    """
                    INSERT INTO kb_v2_edges(source_id, target_id, rel, source_path, metadata, confidence, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, rel) DO UPDATE SET
                        source_path=excluded.source_path,
                        metadata=excluded.metadata,
                        confidence=excluded.confidence,
                        updated_at=excluded.updated_at
                    """,
                    (
                        edge.source_id,
                        edge.target_id,
                        edge.rel,
                        edge.source_path,
                        _json_dumps(edge.metadata),
                        float(edge.confidence),
                        now,
                    ),
                )
                count += 1
            self._conn.commit()
        return count

    def upsert_tasks(self, tasks: Iterable[TaskRecipe]) -> int:
        now = time.time()
        count = 0
        with self._lock:
            for task in tasks:
                self._conn.execute(
                    """
                    INSERT INTO kb_v2_tasks(id, name, description, triggers, tool_plan, source_path, metadata, confidence, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        description=excluded.description,
                        triggers=excluded.triggers,
                        tool_plan=excluded.tool_plan,
                        source_path=excluded.source_path,
                        metadata=excluded.metadata,
                        confidence=excluded.confidence,
                        updated_at=excluded.updated_at
                    """,
                    (
                        task.id,
                        task.name,
                        task.description,
                        json.dumps(list(task.triggers), ensure_ascii=True),
                        json.dumps(list(task.tool_plan), ensure_ascii=True),
                        task.source_path,
                        _json_dumps(task.metadata),
                        float(task.confidence),
                        now,
                    ),
                )
                count += 1
            self._conn.commit()
        return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = {
                "items": self._conn.execute("SELECT COUNT(*) FROM kb_v2_items").fetchone()[0],
                "chunks": self._conn.execute("SELECT COUNT(*) FROM kb_v2_chunks").fetchone()[0],
                "edges": self._conn.execute("SELECT COUNT(*) FROM kb_v2_edges").fetchone()[0],
                "tasks": self._conn.execute("SELECT COUNT(*) FROM kb_v2_tasks").fetchone()[0],
            }
            by_type = dict(self._conn.execute("SELECT type, COUNT(*) FROM kb_v2_items GROUP BY type").fetchall())
        return {**{k: int(v) for k, v in rows.items()}, "by_type": {k: int(v) for k, v in by_type.items()}}

    def search_chunks(self, query: str, *, top_k: int = 12) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.id, c.item_id, c.kind, c.text, c.source_path, c.heading, c.citation,
                       c.metadata, c.confidence,
                       i.type, i.name, i.title, i.summary, i.source_path, i.metadata, i.confidence
                FROM kb_v2_chunks c
                JOIN kb_v2_items i ON i.id = c.item_id
                """
            ).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            text = " ".join(str(v or "") for v in (row[2], row[3], row[4], row[5], row[10], row[11], row[12]))
            hay = text.lower()
            tokens = set(tokenize(text))
            score = 0.0
            for tok in q_tokens:
                if tok in tokens:
                    score += 2.0
                elif tok in hay:
                    score += 0.7
                if tok in str(row[10]).lower():
                    score += 2.0
            if row[9] == "table" and "prod." in str(row[10]):
                score += 0.35
            if row[9] == "table" and str(row[10]).startswith("local."):
                score -= 0.75
            if row[1] == "doc_overview:priceeye" and {"priceeye", "work"}.issubset(q_set):
                score += 6.0
            if not q_set.intersection(tokens) and score <= 0:
                continue
            if score <= 0:
                continue
            item_metadata = _json_loads(row[14])
            chunk_metadata = _json_loads(row[7])
            source_type = _source_type(item_metadata, chunk_metadata)
            source_weight = _SOURCE_WEIGHT.get(source_type, 1.0)
            scored.append(
                (
                    score * float(row[15] or 1.0) * float(row[8] or 1.0) * source_weight,
                    {
                        "chunk": {
                            "id": row[0],
                            "item_id": row[1],
                            "kind": row[2],
                            "text": row[3],
                            "source_path": row[4],
                            "heading": row[5],
                            "citation": row[6],
                            "metadata": chunk_metadata,
                            "confidence": row[8],
                            "source_type": source_type,
                            "requires_verification": bool(
                                chunk_metadata.get("requires_verification")
                                or item_metadata.get("requires_verification")
                            ),
                        },
                        "item": {
                            "id": row[1],
                            "type": row[9],
                            "name": row[10],
                            "title": row[11],
                            "summary": row[12],
                            "source_path": row[13] or row[4],
                            "metadata": item_metadata,
                            "confidence": row[15],
                            "source_type": source_type,
                            "requires_verification": bool(item_metadata.get("requires_verification")),
                        },
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        out = []
        for score, payload in scored[:top_k]:
            payload["score"] = round(float(score), 4)
            out.append(payload)
        return out

    def match_tasks(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        q_tokens = set(tokenize(query))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, description, triggers, tool_plan, source_path, metadata, confidence FROM kb_v2_tasks"
            ).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            try:
                triggers = json.loads(row[3] or "[]")
            except Exception:
                triggers = []
            try:
                tool_plan = json.loads(row[4] or "[]")
            except Exception:
                tool_plan = []
            text = " ".join([row[1], row[2], " ".join(triggers)]).lower()
            tokens = set(tokenize(text))
            score = len(q_tokens.intersection(tokens)) * 1.5
            name_tokens = set(tokenize(str(row[1])))
            if name_tokens and name_tokens.issubset(q_tokens):
                score += 8.0
            for trigger in triggers:
                trigger_l = str(trigger).lower()
                if trigger_l and trigger_l in (query or "").lower():
                    score += 5.0
            if not str(row[0]).startswith("task:e2e:"):
                score += 2.0
            if score <= 0:
                continue
            task = {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "triggers": triggers,
                "tool_plan": tool_plan,
                "source_path": row[5],
                "metadata": _json_loads(row[6]),
                "confidence": row[7],
            }
            source_type = str(task["metadata"].get("source_type") or "unknown")
            scored.append((score * float(row[7] or 1.0) * _SOURCE_WEIGHT.get(source_type, 1.0), task))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**task, "score": round(float(score), 4)} for score, task in scored[:top_k]]

    def edges_for_items(self, item_ids: list[str], *, limit: int = 80) -> list[dict[str, Any]]:
        if not item_ids:
            return []
        placeholders = ",".join("?" for _ in item_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT e.source_id, e.target_id, e.rel, e.source_path, e.metadata, e.confidence,
                       si.type, si.name, ti.type, ti.name
                FROM kb_v2_edges e
                LEFT JOIN kb_v2_items si ON si.id = e.source_id
                LEFT JOIN kb_v2_items ti ON ti.id = e.target_id
                WHERE e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders})
                LIMIT ?
                """,
                (*item_ids, *item_ids, int(limit)),
            ).fetchall()
        return [
            {
                "source_id": r[0],
                "target_id": r[1],
                "rel": r[2],
                "source_path": r[3],
                "metadata": _json_loads(r[4]),
                "confidence": r[5],
                "source_type": _source_type({}, _json_loads(r[4])),
                "source": {"type": r[6], "name": r[7]},
                "target": {"type": r[8], "name": r[9]},
            }
            for r in rows
        ]

    def items_by_names(self, names: list[str]) -> list[dict[str, Any]]:
        if not names:
            return []
        placeholders = ",".join("?" for _ in names)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, type, name, title, summary, source_path, metadata, confidence
                FROM kb_v2_items
                WHERE name IN ({placeholders})
                """,
                tuple(names),
            ).fetchall()
        return [
            {
                "id": row[0],
                "type": row[1],
                "name": row[2],
                "title": row[3],
                "summary": row[4],
                "source_path": row[5],
                "metadata": _json_loads(row[6]),
                "confidence": row[7],
                "source_type": str(_json_loads(row[6]).get("source_type") or "unknown"),
                "requires_verification": bool(_json_loads(row[6]).get("requires_verification")),
                "score": 4.0,
                "matched_chunk_id": None,
                "matched_text": row[4],
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
