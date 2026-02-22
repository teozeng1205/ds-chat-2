"""Local knowledge base loader with lexical + vector indexes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .nextgen_types import KBSearchHit, RelationshipSpec, TableSpec

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    yaml = None


DEFAULT_KB_ROOT = Path(__file__).resolve().parent.parent / "knowledgebase"


def _read_yaml(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for knowledgebase *.yaml files. Install dependency `pyyaml`."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


class KnowledgeBaseService:
    """Loads local KB files and serves table/docs/playbook lookup APIs."""

    def __init__(self, root_dir: Path | str | None = None):
        self.root_dir = (Path(root_dir) if root_dir else DEFAULT_KB_ROOT).resolve()
        self.index_dir = (self.root_dir / ".index").resolve()
        self.vector_dir = (self.index_dir / "vector").resolve()
        self.fts_path = (self.index_dir / "fts.sqlite").resolve()
        self.state_path = (self.index_dir / "state.json").resolve()
        self._lock = threading.Lock()

        self.tables: dict[str, TableSpec] = {}
        self.relationships: list[RelationshipSpec] = []
        self.playbooks: dict[str, dict[str, Any]] = {}
        self.common_codes: dict[str, Any] = {}

        self._doc_records: list[dict[str, Any]] = []
        self._common_code_index: dict[str, tuple[str, str]] = {}
        self._loaded_hash: str | None = None

    def _source_files(self) -> list[Path]:
        files: list[Path] = []
        for rel in ["tables", "relationships", "entities", "playbooks", "docs", "schemas"]:
            folder = self.root_dir / rel
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".md", ".json", ".txt"}:
                    files.append(path)
        return files

    def _source_hash(self) -> str:
        digest = hashlib.sha256()
        for path in self._source_files():
            digest.update(str(path.relative_to(self.root_dir)).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def refresh_if_needed(self, force: bool = False) -> None:
        with self._lock:
            current_hash = self._source_hash()
            if not force and self._loaded_hash == current_hash:
                return
            self._load_all()
            self._build_indexes()
            self._loaded_hash = current_hash
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({"source_hash": current_hash}, indent=2),
                encoding="utf-8",
            )

    def _load_all(self) -> None:
        self.tables = self._load_tables()
        self.relationships = self._load_relationships()
        self.playbooks = self._load_playbooks()
        self.common_codes = self._load_common_codes()
        self._common_code_index = self._build_common_code_index(self.common_codes)
        self._doc_records = self._build_doc_records()

    def _load_tables(self) -> dict[str, TableSpec]:
        tables_dir = self.root_dir / "tables"
        loaded: dict[str, TableSpec] = {}
        for path in sorted(tables_dir.glob("*.yaml")):
            payload = _read_yaml(path) or {}
            entries = payload.get("tables", []) if isinstance(payload, dict) else []
            if isinstance(payload, dict) and "table_id" in payload:
                entries = [payload]
            for entry in entries:
                spec = TableSpec.model_validate(entry)
                loaded[spec.table_id] = spec
        return loaded

    def _load_relationships(self) -> list[RelationshipSpec]:
        graph_path = self.root_dir / "relationships" / "graph.yaml"
        if not graph_path.exists():
            return []
        payload = _read_yaml(graph_path) or {}
        rows = payload.get("relationships", []) if isinstance(payload, dict) else []
        out: list[RelationshipSpec] = []
        for row in rows:
            out.append(RelationshipSpec.model_validate(row))
        return out

    def _load_playbooks(self) -> dict[str, dict[str, Any]]:
        playbook_dir = self.root_dir / "playbooks"
        out: dict[str, dict[str, Any]] = {}
        for path in sorted(playbook_dir.glob("*.yaml")):
            payload = _read_yaml(path) or {}
            playbook_id = str(payload.get("playbook_id") or path.stem)
            payload["playbook_id"] = playbook_id
            out[playbook_id] = payload
        return out

    def _load_common_codes(self) -> dict[str, Any]:
        path = self.root_dir / "entities" / "common_codes.yaml"
        if not path.exists():
            return {}
        payload = _read_yaml(path) or {}
        return payload if isinstance(payload, dict) else {}

    def _build_common_code_index(self, payload: dict[str, Any]) -> dict[str, tuple[str, str]]:
        index: dict[str, tuple[str, str]] = {}
        buckets: list[tuple[str, str]] = [
            ("providers", "provider"),
            ("sites", "site"),
            ("customers", "customer"),
        ]
        for key, entity_type in buckets:
            rows = payload.get(key, []) if isinstance(payload, dict) else []
            for row in rows:
                if isinstance(row, str):
                    canonical = row.strip().upper()
                    tokens = [canonical]
                else:
                    canonical = str(row.get("code", "")).strip().upper()
                    aliases = row.get("aliases", []) if isinstance(row, dict) else []
                    tokens = [canonical, *[str(alias).strip().upper() for alias in aliases]]
                for token in tokens:
                    norm = _normalize_token(token)
                    if norm:
                        index[norm] = (entity_type, canonical)
        return index

    def _build_doc_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for spec in self.tables.values():
            title = f"Table {spec.table_id}"
            body = json.dumps(spec.model_dump(mode="json"), indent=2)
            records.append(
                {
                    "doc_id": f"table::{spec.table_id}",
                    "scope": "table",
                    "path": f"tables/{spec.table_id}",
                    "title": title,
                    "body": body,
                }
            )

        for relation in self.relationships:
            records.append(
                {
                    "doc_id": f"relationship::{relation.left_table_id}::{relation.right_table_id}::{relation.left_key}",
                    "scope": "relationship",
                    "path": "relationships/graph.yaml",
                    "title": f"Relation {relation.left_table_id} -> {relation.right_table_id}",
                    "body": relation.model_dump_json(indent=2),
                }
            )

        for playbook_id, playbook in self.playbooks.items():
            records.append(
                {
                    "doc_id": f"playbook::{playbook_id}",
                    "scope": "playbook",
                    "path": f"playbooks/{playbook_id}.yaml",
                    "title": str(playbook.get("title") or playbook_id),
                    "body": json.dumps(playbook, indent=2),
                }
            )

        docs_dir = self.root_dir / "docs"
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            body = path.read_text(encoding="utf-8", errors="replace")
            rel_no_suffix = str(path.relative_to(self.root_dir).with_suffix(""))
            doc_id = f"doc::{rel_no_suffix.replace('/', '::')}"
            records.append(
                {
                    "doc_id": doc_id,
                    "scope": "doc",
                    "path": str(path.relative_to(self.root_dir)),
                    "title": path.stem,
                    "body": body,
                }
            )

        return records

    def _build_indexes(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self._rebuild_fts_index()
        self._rebuild_vector_index()

    def _rebuild_fts_index(self) -> None:
        conn = sqlite3.connect(self.fts_path)
        try:
            conn.execute("DROP TABLE IF EXISTS docs_fts")
            conn.execute(
                """
                CREATE VIRTUAL TABLE docs_fts
                USING fts5(doc_id UNINDEXED, scope UNINDEXED, path UNINDEXED, title, body)
                """
            )
            conn.executemany(
                "INSERT INTO docs_fts (doc_id, scope, path, title, body) VALUES (?, ?, ?, ?, ?)",
                [
                    (r["doc_id"], r["scope"], r["path"], r["title"], r["body"])
                    for r in self._doc_records
                ],
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for token in text.lower().split():
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dim
            sign = -1.0 if (digest[4] % 2) else 1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _atomic_save_numpy(path: Path, values: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=path.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, values, allow_pickle=False)
            temp_path.replace(path)
        except Exception:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=path.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            temp_path.replace(path)
        except Exception:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _rebuild_vector_index(self) -> None:
        meta_rows = [
            {
                "doc_id": rec["doc_id"],
                "scope": rec["scope"],
                "path": rec["path"],
                "title": rec["title"],
                "body": rec["body"],
            }
            for rec in self._doc_records
        ]
        vectors = np.vstack(
            [self._hash_embed(f"{row['title']}\n{row['body']}") for row in meta_rows]
        ) if meta_rows else np.zeros((0, 384), dtype=np.float32)
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

        self._atomic_save_numpy(self.vector_dir / "doc_vectors.npy", vectors)
        self._atomic_write_json(self.vector_dir / "doc_meta.json", meta_rows)

    def get_table(self, table_id: str) -> TableSpec:
        self.refresh_if_needed()
        try:
            return self.tables[table_id]
        except KeyError as exc:
            valid = ", ".join(sorted(self.tables.keys()))
            raise ValueError(f"Unknown table_id '{table_id}'. Known table IDs: {valid}") from exc

    def list_tables(self) -> list[dict[str, Any]]:
        self.refresh_if_needed()
        return [spec.model_dump(mode="json") for spec in self.tables.values()]

    def list_relationships(self) -> list[dict[str, Any]]:
        self.refresh_if_needed()
        return [rel.model_dump(mode="json") for rel in self.relationships]

    def resolve_common_code(self, code: str) -> tuple[str, str] | None:
        self.refresh_if_needed()
        return self._common_code_index.get(_normalize_token(code))

    def list_playbooks(self) -> list[dict[str, Any]]:
        self.refresh_if_needed()
        return list(self.playbooks.values())

    def match_playbook(self, question: str) -> dict[str, Any] | None:
        self.refresh_if_needed()
        lowered = question.lower()

        best: tuple[int, dict[str, Any]] | None = None
        for payload in self.playbooks.values():
            keywords = [str(k).lower() for k in payload.get("keywords", [])]
            score = sum(1 for keyword in keywords if keyword and keyword in lowered)
            if best is None or score > best[0]:
                best = (score, payload)

        if best and best[0] > 0:
            return best[1]

        if "top site" in lowered and "issue" in lowered:
            return self.playbooks.get("top_site_issues")
        return None

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        self.refresh_if_needed()
        if not query.strip():
            return []

        merged: dict[str, KBSearchHit] = {}
        for hit in self._search_fts(query, top_k=top_k):
            merged[hit.doc_id] = hit

        for hit in self._search_vector(query, top_k=top_k):
            existing = merged.get(hit.doc_id)
            if existing is None:
                merged[hit.doc_id] = hit
                continue
            merged[hit.doc_id] = KBSearchHit(
                doc_id=hit.doc_id,
                scope=hit.scope,
                path=hit.path,
                title=hit.title,
                snippet=existing.snippet,
                score=max(existing.score, hit.score),
                method="merged",
            )

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return [item.model_dump(mode="json") for item in ranked[:top_k]]

    def _search_fts(self, query: str, top_k: int) -> list[KBSearchHit]:
        if not self.fts_path.exists():
            return []
        conn = sqlite3.connect(self.fts_path)
        try:
            rows = conn.execute(
                """
                SELECT doc_id, scope, path, title, substr(body, 1, 240), bm25(docs_fts) AS rank
                FROM docs_fts
                WHERE docs_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()

        hits: list[KBSearchHit] = []
        for doc_id, scope, path, title, snippet, rank in rows:
            rank_val = float(rank) if rank is not None else 1.0
            score = 1.0 / (1.0 + abs(rank_val))
            hits.append(
                KBSearchHit(
                    doc_id=str(doc_id),
                    scope=str(scope),
                    path=str(path),
                    title=str(title),
                    snippet=str(snippet),
                    score=score,
                    method="fts",
                )
            )
        return hits

    def _search_vector(self, query: str, top_k: int) -> list[KBSearchHit]:
        vec_path = self.vector_dir / "doc_vectors.npy"
        meta_path = self.vector_dir / "doc_meta.json"
        if not vec_path.exists() or not meta_path.exists():
            return []

        try:
            matrix = np.load(vec_path, allow_pickle=False)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        if not isinstance(meta, list):
            return []
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return []
        if len(meta) < matrix.shape[0]:
            return []
        if len(meta) > matrix.shape[0]:
            meta = meta[: matrix.shape[0]]

        matrix_norm = np.asarray(matrix, dtype=np.float64)
        matrix_norm = np.nan_to_num(matrix_norm, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            row_norms = np.linalg.norm(matrix_norm, axis=1)
        row_norms = np.nan_to_num(row_norms, nan=0.0, posinf=0.0, neginf=0.0)
        nonzero_rows = row_norms > 0
        if np.any(nonzero_rows):
            matrix_norm[nonzero_rows] = matrix_norm[nonzero_rows] / row_norms[nonzero_rows, None]

        query_vec = np.asarray(self._hash_embed(query, dim=matrix_norm.shape[1]), dtype=np.float64)
        query_vec = np.nan_to_num(query_vec, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            query_norm = float(np.linalg.norm(query_vec))
        if query_norm > 0:
            query_vec = query_vec / query_norm

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            scores = matrix_norm @ query_vec
        scores = np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)
        scores = np.clip(scores, -1.0, 1.0)

        max_hits = max(1, min(top_k, int(scores.shape[0])))
        top_idx = np.argsort(scores)[::-1][:max_hits]

        hits: list[KBSearchHit] = []
        for idx in top_idx:
            row = meta[int(idx)]
            sim = float(scores[int(idx)])
            normalized = (sim + 1.0) / 2.0
            snippet = str(row.get("body", ""))[:240]
            hits.append(
                KBSearchHit(
                    doc_id=str(row.get("doc_id")),
                    scope=str(row.get("scope")),
                    path=str(row.get("path")),
                    title=str(row.get("title")),
                    snippet=snippet,
                    score=normalized,
                    method="vector",
                )
            )
        return hits
