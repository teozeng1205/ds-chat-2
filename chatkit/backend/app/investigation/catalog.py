"""Local file-backed knowledge catalog with sqlite index and task cards."""

from __future__ import annotations

import glob
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


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
        for key, entity in [
            ("providers", "provider"),
            ("sites", "site"),
            ("customers", "customer"),
            ("customer_sites", "customer_site"),
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
                CREATE TABLE IF NOT EXISTS kb_task_cards (
                    card_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    signals_json TEXT NOT NULL,
                    required_entities_json TEXT NOT NULL,
                    candidate_tables_json TEXT NOT NULL,
                    actions_json TEXT NOT NULL,
                    analysis_mode TEXT NOT NULL,
                    analysis_instructions TEXT NOT NULL,
                    python_template TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _source_files(self) -> list[Path]:
        patterns = [
            "*.md",
            "*.json",
            "docs/**/*.md",
            "docs/**/*.txt",
            "task_cards/**/*.md",
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

    @staticmethod
    def _parse_markdown_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        if not text.startswith("---\n"):
            return {}, text
        end = text.find("\n---\n", 4)
        if end <= 0:
            return {}, text
        frontmatter_raw = text[4:end]
        body = text[end + 5 :]
        if yaml is None:
            return {}, body
        data = yaml.safe_load(frontmatter_raw) or {}
        if not isinstance(data, dict):
            data = {}
        return data, body

    def _load_task_cards(self) -> list[dict[str, Any]]:
        cards_dir = self.root / "task_cards"
        if not cards_dir.exists():
            return []

        cards: list[dict[str, Any]] = []
        for path in sorted(cards_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            fm, body = self._parse_markdown_frontmatter(text)
            card_id = str(fm.get("id") or path.stem)
            card = {
                "card_id": card_id,
                "title": str(fm.get("title") or card_id),
                "signals": [str(item).strip().lower() for item in (fm.get("signals") or []) if str(item).strip()],
                "required_entities": [str(item).strip().lower() for item in (fm.get("required_entities") or []) if str(item).strip()],
                "candidate_tables": [str(item).strip() for item in (fm.get("candidate_tables") or []) if str(item).strip()],
                "actions": list(fm.get("actions") or []),
                "analysis_mode": str(fm.get("analysis_mode") or "profile_dataset"),
                "analysis_instructions": str(fm.get("analysis_instructions") or ""),
                "python_template": str(fm.get("python_template") or ""),
                "body": body.strip(),
                "source_path": str(path),
            }
            cards.append(card)
        return cards

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
            conn.execute("DELETE FROM kb_task_cards")

            table_rows = self._parse_tables_markdown(self.root / "tables.md")
            live_meta = self._load_live_table_metadata()
            for table_name, row in table_rows.items():
                live = live_meta.get(table_name, {})
                status = str(live.get("status", "")).lower()
                status_note = ""
                if status:
                    status_note = f" | live_status={status}"
                notes = str(row.get("notes", "") or "") + status_note
                conn.execute(
                    """
                    INSERT INTO kb_tables (table_name, datasource, tier, notes, query_example, analysis_example, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        table_name,
                        row.get("datasource", "redshift_analytics"),
                        row.get("tier", "common"),
                        notes,
                        row.get("query_example"),
                        row.get("analysis_example"),
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

            task_cards = self._load_task_cards()
            for card in task_cards:
                conn.execute(
                    """
                    INSERT INTO kb_task_cards (
                        card_id, title, signals_json, required_entities_json, candidate_tables_json,
                        actions_json, analysis_mode, analysis_instructions, python_template,
                        body, source_path, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card["card_id"],
                        card["title"],
                        json.dumps(card.get("signals", []), ensure_ascii=True),
                        json.dumps(card.get("required_entities", []), ensure_ascii=True),
                        json.dumps(card.get("candidate_tables", []), ensure_ascii=True),
                        json.dumps(card.get("actions", []), ensure_ascii=True),
                        card.get("analysis_mode", "profile_dataset"),
                        card.get("analysis_instructions", ""),
                        card.get("python_template", ""),
                        card.get("body", ""),
                        card.get("source_path", ""),
                        now,
                    ),
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
                "task_cards_indexed": len(task_cards),
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

    def retrieve(self, *, question: str, entities: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
        tokens = self._tokens(question)
        conn = sqlite3.connect(self.db_path)
        try:
            table_rows = conn.execute(
                "SELECT table_name, datasource, tier, notes, query_example, analysis_example FROM kb_tables"
            ).fetchall()
            table_scored: list[tuple[float, dict[str, Any]]] = []
            for table_name, datasource, tier, notes, query_example, analysis_example in table_rows:
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
                        },
                    )
                )
            table_scored.sort(key=lambda item: item[0], reverse=True)

            card_rows = conn.execute(
                """
                SELECT card_id, title, signals_json, required_entities_json, candidate_tables_json,
                       actions_json, analysis_mode, analysis_instructions, python_template, body, source_path
                FROM kb_task_cards
                """
            ).fetchall()

            card_scored: list[tuple[float, dict[str, Any]]] = []
            for (
                card_id,
                title,
                signals_json,
                required_entities_json,
                candidate_tables_json,
                actions_json,
                analysis_mode,
                analysis_instructions,
                python_template,
                body,
                source_path,
            ) in card_rows:
                signals = json.loads(signals_json or "[]")
                required_entities = json.loads(required_entities_json or "[]")
                candidate_tables = json.loads(candidate_tables_json or "[]")
                actions = json.loads(actions_json or "[]")

                score = 0.0
                title_body = f"{title} {body}".lower()
                for tok in tokens:
                    if tok in title_body:
                        score += 0.7
                for signal in signals:
                    signal_text = str(signal).strip().lower()
                    if signal_text and signal_text in question.lower():
                        score += 2.0
                for ent in required_entities:
                    if ent == "provider" and entities.get("providers"):
                        score += 1.0
                    if ent == "site" and entities.get("sites"):
                        score += 1.0
                    if ent == "customer" and entities.get("customers"):
                        score += 1.0
                for table_name, _datasource, _tier, _notes, _q, _a in table_rows:
                    if table_name in candidate_tables and any(tok in table_name.lower() for tok in tokens):
                        score += 0.5
                if score <= 0:
                    continue

                card_scored.append(
                    (
                        score,
                        {
                            "card_id": card_id,
                            "title": title,
                            "signals": signals,
                            "required_entities": required_entities,
                            "candidate_tables": candidate_tables,
                            "actions": actions,
                            "analysis_mode": analysis_mode,
                            "analysis_instructions": analysis_instructions,
                            "python_template": python_template,
                            "body": body,
                            "source_path": source_path,
                        },
                    )
                )

            card_scored.sort(key=lambda item: item[0], reverse=True)

            # If top task-card provides candidate tables, prioritize them.
            preferred_card_tables: list[str] = []
            if card_scored:
                top_card = card_scored[0][1]
                preferred_card_tables = [str(item) for item in (top_card.get("candidate_tables") or []) if str(item).strip()]
            if preferred_card_tables:
                boost: dict[str, float] = {name: float(100 - idx) for idx, name in enumerate(preferred_card_tables)}
                table_scored = sorted(
                    table_scored,
                    key=lambda item: (boost.get(item[1]["table_name"], 0.0), item[0]),
                    reverse=True,
                )

            candidate_tables = [item[1]["table_name"] for item in table_scored[:top_k]]
            table_hints = [item[1] for item in table_scored[:top_k]]
            task_cards = [item[1] for item in card_scored[:top_k]]

            return {
                "candidate_tables": candidate_tables,
                "table_hints": table_hints,
                "task_cards": task_cards,
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
                    "Profile dataset and, if needed, run python analysis",
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
