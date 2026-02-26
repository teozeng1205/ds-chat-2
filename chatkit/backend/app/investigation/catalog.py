"""Local file-backed knowledge catalog with sqlite index."""

from __future__ import annotations

import glob
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TableKnowledge:
    table_name: str
    datasource: str
    tier: str
    notes: str
    partitions: list[dict[str, Any]]
    columns: list[dict[str, Any]]
    sample_row: dict[str, Any] | None
    query_example: str | None
    analysis_example: str | None


class LocalCodeCatalog:
    """Local provider/site/customer code catalog from common_codes.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, Any] | None = None
        self._mtime: float = 0.0

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"providers": [], "sites": [], "customers": [], "customer_sites": []}
        mtime = self.path.stat().st_mtime
        if self._cache is not None and mtime <= self._mtime:
            return self._cache
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        self._cache = payload
        self._mtime = mtime
        return payload

    @staticmethod
    def _normalize(raw: str) -> str:
        return "".join(ch for ch in str(raw).upper() if ch.isalnum())

    def resolve(self, token: str) -> tuple[str, str] | None:
        payload = self._load()
        buckets = [
            ("providers", "provider"),
            ("sites", "site"),
            ("customers", "customer"),
            ("customer_sites", "customer_site"),
        ]
        norm = self._normalize(token)
        if not norm:
            return None

        for bucket, entity in buckets:
            for row in payload.get(bucket, []) or []:
                if isinstance(row, str):
                    canonical = row.strip().upper()
                    aliases: list[str] = []
                else:
                    canonical = str(row.get("code", "")).strip().upper()
                    aliases = [str(v).strip().upper() for v in (row.get("aliases", []) or [])]
                for candidate in [canonical, *aliases]:
                    if self._normalize(candidate) == norm:
                        return entity, canonical
        return None

    def rows(self) -> list[tuple[str, str, str]]:
        payload = self._load()
        out: list[tuple[str, str, str]] = []
        for key, entity in [("providers", "provider"), ("sites", "site"), ("customers", "customer"), ("customer_sites", "customer_site")]:
            for row in payload.get(key, []) or []:
                if isinstance(row, str):
                    code = row.strip().upper()
                else:
                    code = str(row.get("code", "")).strip().upper()
                if code:
                    out.append((code, entity, "common_codes"))
        return out


class KnowledgeBase:
    """Local sqlite-backed KB index with source files in investigation/knowledge."""

    def __init__(self, *, root: Path, db_path: Path) -> None:
        self.root = root
        self.db_path = db_path
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kb_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_tables (
                    table_name TEXT PRIMARY KEY,
                    datasource TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    notes TEXT,
                    query_example TEXT,
                    analysis_example TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_partitions (
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    role TEXT,
                    inferred_type TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_columns (
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    data_type TEXT,
                    nullable INTEGER,
                    is_key INTEGER,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_example_rows (
                    table_name TEXT PRIMARY KEY,
                    example_json_masked TEXT,
                    captured_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_codes (
                    code TEXT NOT NULL,
                    code_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _source_files(self) -> list[Path]:
        out: list[Path] = []
        for pattern in ["*.md", "*.json", "docs/*.md", "docs/*.txt"]:
            out.extend(sorted(self.root.glob(pattern)))
        return [p for p in out if p.is_file()]

    def _source_hash(self) -> str:
        digest = hashlib.sha256()
        for path in self._source_files():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _parse_tables_markdown(path: Path) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return out
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for raw in lines:
            line = raw.strip()
            if not line.startswith("|"):
                continue
            cells = [part.strip() for part in line.strip("|").split("|")]
            if len(cells) < 1:
                continue
            candidate = cells[0].strip("`")
            if "." not in candidate or " " in candidate:
                continue
            if candidate.lower().startswith("table"):
                continue
            datasource = "redshift_analytics"
            if candidate.startswith("prod.monitoring") or candidate.startswith("local.monitoring"):
                datasource = "redshift_core"
            if candidate.startswith("priceeye."):
                datasource = "mysql_priceeye"
            out.setdefault(
                candidate,
                {
                    "table_name": candidate,
                    "datasource": datasource,
                    "tier": "common",
                    "notes": cells[1] if len(cells) > 1 else "",
                    "partitions": [],
                    "columns": [],
                    "sample_row": None,
                    "query_example": f"SELECT * FROM {candidate} LIMIT 200",
                    "analysis_example": "Run profile summary and missingness analysis",
                },
            )
        return out

    def refresh(self, *, force: bool, catalog: LocalCodeCatalog) -> dict[str, Any]:
        now = time.time()
        source_hash = self._source_hash()
        conn = sqlite3.connect(self.db_path)
        try:
            old_row = conn.execute("SELECT value FROM kb_meta WHERE key='source_hash'").fetchone()
            old_hash = str(old_row[0]) if old_row else ""
            if not force and old_hash == source_hash:
                return {"ok": True, "refreshed": False, "source_hash": source_hash}

            conn.execute("DELETE FROM kb_tables")
            conn.execute("DELETE FROM kb_partitions")
            conn.execute("DELETE FROM kb_columns")
            conn.execute("DELETE FROM kb_example_rows")
            conn.execute("DELETE FROM kb_codes")
            conn.execute("DELETE FROM kb_documents")

            tables_doc = self.root / "tables.md"
            table_rows = self._parse_tables_markdown(tables_doc)
            for table_name, row in table_rows.items():
                conn.execute(
                    """
                    INSERT INTO kb_tables (table_name, datasource, tier, notes, query_example, analysis_example, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        table_name,
                        row.get("datasource", "redshift_analytics"),
                        row.get("tier", "common"),
                        row.get("notes", ""),
                        row.get("query_example"),
                        row.get("analysis_example"),
                        now,
                    ),
                )

            for code, code_type, source in catalog.rows():
                conn.execute(
                    "INSERT INTO kb_codes (code, code_type, source, updated_at) VALUES (?, ?, ?, ?)",
                    (code, code_type, source, now),
                )

            for file_path in self._source_files():
                content = file_path.read_text(encoding="utf-8", errors="replace")
                doc_id = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:24]
                conn.execute(
                    "INSERT INTO kb_documents (id, source_path, content, updated_at) VALUES (?, ?, ?, ?)",
                    (doc_id, str(file_path), content, now),
                )

            conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('source_hash', ?)", (source_hash,))
            conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('last_refresh', ?)", (str(now),))
            conn.commit()
            return {
                "ok": True,
                "refreshed": True,
                "source_hash": source_hash,
                "tables_indexed": len(table_rows),
                "codes_indexed": len(catalog.rows()),
            }
        finally:
            conn.close()

    def browse_files(self, path_or_glob: str) -> dict[str, Any]:
        pattern = path_or_glob.strip() or "*"
        base = self.root
        if any(ch in pattern for ch in "*?[]"):
            files = [Path(path) for path in glob.glob(str(base / pattern), recursive=True)]
        else:
            target = (base / pattern).resolve()
            files = [target] if target.exists() else []
        entries: list[dict[str, Any]] = []
        for path in files:
            if not path.is_file() or base not in path.parents and path != base:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            entries.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "content": text[:8000],
                    "truncated": len(text) > 8000,
                }
            )
        return {"count": len(entries), "files": entries}

    def retrieve(self, *, question: str, entities: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
        tokens = [tok.lower() for tok in question.split() if tok]
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("SELECT table_name, datasource, tier, notes, query_example, analysis_example FROM kb_tables").fetchall()
            scored: list[tuple[float, dict[str, Any]]] = []
            for table_name, datasource, tier, notes, query_example, analysis_example in rows:
                score = 0.0
                text = f"{table_name} {notes}".lower()
                for tok in tokens:
                    if tok in text:
                        score += 1.0
                if entities.get("providers") and "provider" in text:
                    score += 1.5
                if entities.get("sites") and "site" in text:
                    score += 1.5
                if entities.get("customers") and "customer" in text:
                    score += 1.5
                if score <= 0:
                    continue
                scored.append(
                    (
                        score,
                        {
                            "table_name": table_name,
                            "datasource": datasource,
                            "tier": tier,
                            "notes": notes,
                            "query_example": query_example,
                            "analysis_example": analysis_example,
                        },
                    )
                )
            scored.sort(key=lambda item: item[0], reverse=True)
            candidate_tables = [item[1]["table_name"] for item in scored[:top_k]]
            hints = [item[1] for item in scored[:top_k]]
            return {
                "candidate_tables": candidate_tables,
                "table_hints": hints,
            }
        finally:
            conn.close()

    def upsert_table_metadata(
        self,
        *,
        table_name: str,
        datasource: str,
        columns: list[dict[str, Any]],
        partitions: list[dict[str, Any]],
        sample_row_masked: dict[str, Any] | None,
        tier: str,
        notes: str,
    ) -> None:
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO kb_tables (table_name, datasource, tier, notes, query_example, analysis_example, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table_name,
                    datasource,
                    tier,
                    notes,
                    f"SELECT * FROM {table_name} LIMIT 200",
                    "Profile table and compute missingness/cardinality/numeric summary",
                    now,
                ),
            )
            conn.execute("DELETE FROM kb_partitions WHERE table_name = ?", (table_name,))
            conn.execute("DELETE FROM kb_columns WHERE table_name = ?", (table_name,))
            for part in partitions:
                conn.execute(
                    "INSERT INTO kb_partitions (table_name, column_name, role, inferred_type, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        table_name,
                        str(part.get("column", "")),
                        str(part.get("role", "recommended")),
                        str(part.get("inferred_type", "unknown")),
                        now,
                    ),
                )
            for col in columns:
                conn.execute(
                    "INSERT INTO kb_columns (table_name, column_name, data_type, nullable, is_key, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        table_name,
                        str(col.get("column_name", "")),
                        str(col.get("data_type", "")),
                        1 if bool(col.get("nullable", True)) else 0,
                        1 if bool(col.get("is_key", False)) else 0,
                        now,
                    ),
                )
            if sample_row_masked is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO kb_example_rows (table_name, example_json_masked, captured_at) VALUES (?, ?, ?)",
                    (table_name, json.dumps(sample_row_masked, ensure_ascii=True), now),
                )
            conn.commit()
        finally:
            conn.close()


__all__ = [
    "KnowledgeBase",
    "LocalCodeCatalog",
    "TableKnowledge",
]
