"""SQLite-backed ChatKit store — persists threads, items, and attachments.

Drop-in replacement for InMemoryStore that survives restarts, so the ChatKit
history panel can list and reopen past conversations. Behaviour (pagination,
ordering, NotFound semantics) mirrors InMemoryStore; only the backing changes
from a process-local dict to a SQLite file under app/.data/.

ChatKit's ThreadItem is an annotated discriminated union, so we (de)serialize
with pydantic TypeAdapters rather than per-class model_validate.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata
from pydantic import TypeAdapter

_THREAD_ADAPTER: TypeAdapter[ThreadMetadata] = TypeAdapter(ThreadMetadata)
_ITEM_ADAPTER: TypeAdapter[ThreadItem] = TypeAdapter(ThreadItem)
_ATTACHMENT_ADAPTER: TypeAdapter[Attachment] = TypeAdapter(Attachment)


def default_thread_db_path() -> Path:
    data_dir = Path(__file__).resolve().parent / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "ds-chat-threads.sqlite"


class SqliteThreadStore(Store[dict]):
    """Persistent ChatKit store backed by SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or default_thread_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id   TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS thread_items (
                    thread_id TEXT NOT NULL,
                    id        TEXT NOT NULL,
                    data      TEXT NOT NULL,
                    PRIMARY KEY (thread_id, id)
                );
                CREATE INDEX IF NOT EXISTS idx_thread_items_thread ON thread_items(thread_id);
                CREATE TABLE IF NOT EXISTS attachments (
                    id   TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # ── (de)serialization ──
    @staticmethod
    def _dump(adapter: TypeAdapter, obj: Any) -> str:
        return adapter.dump_json(obj).decode("utf-8")

    # ── threads ──
    async def load_thread(self, thread_id: str, context: dict) -> ThreadMetadata:
        with self._lock:
            row = self._conn.execute("SELECT data FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Thread {thread_id} not found")
        return _THREAD_ADAPTER.validate_json(row[0])

    async def save_thread(self, thread: ThreadMetadata, context: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO threads(id, data) VALUES(?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (thread.id, self._dump(_THREAD_ADAPTER, thread)),
            )
            self._conn.commit()

    async def load_threads(self, limit: int, after: str | None, order: str, context: dict) -> Page[ThreadMetadata]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM threads").fetchall()
        threads = [_THREAD_ADAPTER.validate_json(r[0]) for r in rows]
        return self._paginate(
            threads, after=after, limit=limit, order=order,
            sort_key=lambda t: t.created_at, cursor_key=lambda t: t.id,
        )

    async def delete_thread(self, thread_id: str, context: dict) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM thread_items WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            self._conn.commit()

    # ── items ──
    async def load_thread_items(self, thread_id: str, after: str | None, limit: int, order: str, context: dict) -> Page[ThreadItem]:
        with self._lock:
            exists = self._conn.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone()
            if exists is None:
                raise NotFoundError(f"Thread {thread_id} not found")
            rows = self._conn.execute("SELECT data FROM thread_items WHERE thread_id = ?", (thread_id,)).fetchall()
        items = [_ITEM_ADAPTER.validate_json(r[0]) for r in rows]
        return self._paginate(
            items, after=after, limit=limit, order=order,
            sort_key=lambda i: i.created_at, cursor_key=lambda i: i.id,
        )

    async def add_thread_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        with self._lock:
            exists = self._conn.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone()
            if exists is None:
                raise NotFoundError(f"Thread {thread_id} not found")
            self._conn.execute(
                "INSERT INTO thread_items(thread_id, id, data) VALUES(?, ?, ?) "
                "ON CONFLICT(thread_id, id) DO UPDATE SET data = excluded.data",
                (thread_id, item.id, self._dump(_ITEM_ADAPTER, item)),
            )
            self._conn.commit()

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        await self.add_thread_item(thread_id, item, context)

    async def load_item(self, thread_id: str, item_id: str, context: dict) -> ThreadItem:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM thread_items WHERE thread_id = ? AND id = ?", (thread_id, item_id)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
        return _ITEM_ADAPTER.validate_json(row[0])

    async def delete_thread_item(self, thread_id: str, item_id: str, context: dict) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM thread_items WHERE thread_id = ? AND id = ?", (thread_id, item_id)
            )
            self._conn.commit()

    # ── attachments ──
    async def save_attachment(self, attachment: Attachment, context: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO attachments(id, data) VALUES(?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (attachment.id, self._dump(_ATTACHMENT_ADAPTER, attachment)),
            )
            self._conn.commit()

    async def load_attachment(self, attachment_id: str, context: dict) -> Attachment:
        with self._lock:
            row = self._conn.execute("SELECT data FROM attachments WHERE id = ?", (attachment_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return _ATTACHMENT_ADAPTER.validate_json(row[0])

    async def delete_attachment(self, attachment_id: str, context: dict) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM attachments WHERE id = ?", (attachment_id,))
            self._conn.commit()

    # ── pagination (mirrors InMemoryStore) ──
    def _paginate(
        self,
        rows: list[Any],
        *,
        after: str | None,
        limit: int,
        order: str,
        sort_key: Callable[[Any], Any],
        cursor_key: Callable[[Any], str],
    ) -> Page[Any]:
        sorted_rows = sorted(rows, key=sort_key, reverse=order == "desc")
        start = 0
        if after:
            for idx, row in enumerate(sorted_rows):
                if cursor_key(row) == after:
                    start = idx + 1
                    break
        data = sorted_rows[start : start + limit]
        has_more = start + limit < len(sorted_rows)
        next_after = cursor_key(data[-1]) if has_more and data else None
        return Page(data=data, has_more=has_more, after=next_after)
