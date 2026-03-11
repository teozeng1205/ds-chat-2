"""Local file-backed knowledge catalog with sqlite index and task cards."""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import re
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
    s3_location: str | None = None
    git_repo: str | None = None
    git_path: str | None = None


class LocalCodeCatalog:
    """Local provider/site/customer code catalog from common_codes.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, Any] | None = None
        self._mtime: float = 0.0

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"providers": [], "sites": [], "customers": []}
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
        for key, entity in [
            ("providers", "provider"),
            ("sites", "site"),
            ("customers", "customer"),
        ]:
            for row in payload.get(key, []) or []:
                if isinstance(row, str):
                    code = row.strip().upper()
                else:
                    code = str(row.get("code", "")).strip().upper()
                if code:
                    out.append((code, entity, "common_codes"))
        return out


class KnowledgeBase:
    """Local sqlite-backed KB index with natural-language task cards and docs."""

    _TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")

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
                    max_sales_date INTEGER,
                    s3_location TEXT,
                    git_repo TEXT,
                    git_path TEXT,
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
            # Schema migration: add new columns if not present in existing DBs
            col_names = [row[1] for row in conn.execute("PRAGMA table_info(kb_tables)").fetchall()]
            if "max_sales_date" not in col_names:
                conn.execute("ALTER TABLE kb_tables ADD COLUMN max_sales_date INTEGER")
                conn.commit()
            if "s3_location" not in col_names:
                conn.execute("ALTER TABLE kb_tables ADD COLUMN s3_location TEXT")
                conn.commit()
            if "git_repo" not in col_names:
                conn.execute("ALTER TABLE kb_tables ADD COLUMN git_repo TEXT")
                conn.commit()
            if "git_path" not in col_names:
                conn.execute("ALTER TABLE kb_tables ADD COLUMN git_path TEXT")
                conn.commit()
        finally:
            conn.close()

    def _source_files(self) -> list[Path]:
        patterns = [
            "*.md",
            "*.json",
            "docs/**/*.md",
            "docs/**/*.txt",
        ]
        out: list[Path] = []
        for pattern in patterns:
            out.extend(sorted(self.root.glob(pattern)))
        return [p for p in out if p.is_file()]

    def _load_live_table_metadata(self) -> dict[str, dict[str, Any]]:
        path = self.root / "common_table_live_metadata.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        rows = payload.get("tables", []) if isinstance(payload, dict) else []
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            table_name = str(row.get("table_name", "")).strip()
            if not table_name:
                continue
            out[table_name] = row
        return out

    def _source_hash(self) -> str:
        digest = hashlib.sha256()
        for path in self._source_files():
            digest.update(str(path.relative_to(self.root)).encode("utf-8"))
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
            if not cells:
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
                    "query_example": f"SELECT * FROM {candidate} LIMIT 200",
                    "analysis_example": "Run generic profile + targeted python analysis",
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
            # Drop legacy task_cards table if it exists from prior versions
            conn.execute("DROP TABLE IF EXISTS kb_task_cards")

            table_rows = self._parse_tables_markdown(self.root / "tables.md")
            live_meta = self._load_live_table_metadata()
            for table_name, row in table_rows.items():
                live = live_meta.get(table_name, {})
                status = str(live.get("status", "")).lower()
                status_note = ""
                if status:
                    status_note = f" | live_status={status}"
                notes = str(row.get("notes", "") or "") + status_note
                max_sales_date = live.get("max_sales_date")  # int YYYYMMDD or None
                tier = live.get("tier") or row.get("tier", "common")
                s3_location = live.get("s3_location")
                git_repo = live.get("git_repo")
                git_path = live.get("git_path")
                conn.execute(
                    """
                    INSERT INTO kb_tables (table_name, datasource, tier, notes, query_example, analysis_example, max_sales_date, s3_location, git_repo, git_path, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        table_name,
                        row.get("datasource", "redshift_analytics"),
                        tier,
                        notes,
                        row.get("query_example"),
                        row.get("analysis_example"),
                        max_sales_date,
                        s3_location,
                        git_repo,
                        git_path,
                        now,
                    ),
                )
                live_columns = live.get("columns", [])
                live_partitions = live.get("partitions", [])
                sample_row_masked = live.get("sample_row_masked")
                if isinstance(live_columns, list):
                    for col in live_columns:
                        if not isinstance(col, dict):
                            continue
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
                if isinstance(live_partitions, list):
                    for part in live_partitions:
                        if not isinstance(part, dict):
                            continue
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
                if isinstance(sample_row_masked, dict) and sample_row_masked:
                    conn.execute(
                        "INSERT OR REPLACE INTO kb_example_rows (table_name, example_json_masked, captured_at) VALUES (?, ?, ?)",
                        (table_name, json.dumps(sample_row_masked, ensure_ascii=True), now),
                    )

            for code, code_type, source in catalog.rows():
                conn.execute(
                    "INSERT INTO kb_codes (code, code_type, source, updated_at) VALUES (?, ?, ?, ?)",
                    (code, code_type, source, now),
                )

            source_files = self._source_files()
            for file_path in source_files:
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
        pattern = path_or_glob.strip() or "**/*"
        base = self.root
        if any(ch in pattern for ch in "*?[]"):
            files = [Path(path) for path in glob.glob(str(base / pattern), recursive=True)]
        else:
            target = (base / pattern).resolve()
            files = [target] if target.exists() else []
        entries: list[dict[str, Any]] = []
        for path in files:
            if not path.is_file():
                continue
            if base not in path.parents and path != base:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            entries.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "content": text[:12000],
                    "truncated": len(text) > 12000,
                }
            )
        return {"count": len(entries), "files": entries}

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return [m.group(0).lower() for m in cls._TOKEN_RE.finditer(text or "")]

    def _load_partition_info(self, conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
        """Load partition info keyed by table name."""
        rows = conn.execute("SELECT table_name, column_name, role, inferred_type FROM kb_partitions").fetchall()
        out: dict[str, list[dict[str, Any]]] = {}
        for table_name, column_name, role, inferred_type in rows:
            out.setdefault(table_name, []).append(
                {"column": column_name, "role": role, "inferred_type": inferred_type}
            )
        return out

    def retrieve(self, *, question: str, entities: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
        tokens = self._tokens(question)
        conn = sqlite3.connect(self.db_path)
        try:
            partition_info = self._load_partition_info(conn)

            table_rows = conn.execute(
                "SELECT table_name, datasource, tier, notes, query_example, analysis_example, max_sales_date, s3_location, git_repo, git_path FROM kb_tables"
            ).fetchall()
            yesterday_int = int(
                (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y%m%d")
            )
            table_scored: list[tuple[float, dict[str, Any]]] = []
            for table_name, datasource, tier, notes, query_example, analysis_example, max_sales_date, s3_location, git_repo, git_path in table_rows:
                score = 0.0
                text = f"{table_name} {notes}".lower()
                for tok in tokens:
                    if tok in text:
                        score += 1.0
                if entities.get("providers") and any(v in text for v in ["provider", "providercode"]):
                    score += 1.0
                if entities.get("sites") and "site" in text:
                    score += 1.0
                if entities.get("customers") and "customer" in text:
                    score += 1.0
                if "live_status=error" in text:
                    score -= 5.0
                if table_name.startswith("local."):
                    score -= 2.5
                if "{" in table_name or "}" in table_name:
                    score -= 10.0
                if max_sales_date and int(max_sales_date) >= yesterday_int:
                    score += 1.5
                if score <= 0:
                    continue
                table_scored.append(
                    (
                        score,
                        {
                            "table_name": table_name,
                            "datasource": datasource,
                            "tier": tier,
                            "notes": notes,
                            "query_example": query_example,
                            "analysis_example": analysis_example,
                            "max_sales_date": max_sales_date,
                            "s3_location": s3_location,
                            "git_repo": git_repo,
                            "git_path": git_path,
                            "partitions": partition_info.get(table_name, []),
                        },
                    )
                )
            table_scored.sort(key=lambda item: item[0], reverse=True)

            candidate_tables = [item[1]["table_name"] for item in table_scored[:top_k]]
            table_hints = [item[1] for item in table_scored[:top_k]]

            # Search documents
            doc_rows = conn.execute(
                "SELECT id, source_path, content FROM kb_documents"
            ).fetchall()
            doc_scored: list[tuple[float, dict]] = []
            for doc_id, source_path, content in doc_rows:
                score = sum(1.0 for tok in tokens if tok in content.lower())
                if score > 0:
                    lines = content.splitlines()
                    pivot = next(
                        (i for i, ln in enumerate(lines) if any(tok in ln.lower() for tok in tokens)),
                        0,
                    )
                    start = max(0, pivot - 5)
                    snippet = "\n".join(lines[start : start + 60])[:3000]
                    doc_scored.append((score, {
                        "source": Path(source_path).name,
                        "snippet": snippet,
                    }))
            doc_scored.sort(key=lambda x: x[0], reverse=True)
            document_hints = [item[1] for item in doc_scored[:3]]

            return {
                "candidate_tables": candidate_tables,
                "table_hints": table_hints,
                "document_hints": document_hints,
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
        s3_location: str | None = None,
        git_repo: str | None = None,
        git_path: str | None = None,
    ) -> None:
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO kb_tables (table_name, datasource, tier, notes, query_example, analysis_example, s3_location, git_repo, git_path, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table_name,
                    datasource,
                    tier,
                    notes,
                    f"SELECT * FROM {table_name} LIMIT 200",
                    "Profile dataset and, if needed, run python analysis",
                    s3_location,
                    git_repo,
                    git_path,
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
