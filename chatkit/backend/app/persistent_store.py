"""
SQLite-backed store compatible with the ChatKit Store interface.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata


def _format_dt(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _serialize(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        payload = obj.model_dump()
    elif hasattr(obj, "dict"):
        payload = obj.dict()
    elif hasattr(obj, "__dict__"):
        payload = obj.__dict__
    else:
        payload = obj
    return json.dumps(payload, default=str)


def _deserialize(payload: str, model_cls: type[Any]) -> Any:
    data = json.loads(payload)
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(data)
    return model_cls(**data)


class SQLiteStore(Store[dict]):
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    data TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    created_at TEXT,
                    data TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_thread_id ON items(thread_id);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_created_at ON items(created_at);"
            )

    async def load_thread(self, thread_id: str, context: dict) -> ThreadMetadata:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT data FROM threads WHERE id = ?;", (thread_id,)
            ).fetchone()
        if not row:
            raise NotFoundError(f"Thread {thread_id} not found")
        return _deserialize(row["data"], ThreadMetadata)

    async def save_thread(self, thread: ThreadMetadata, context: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO threads (id, created_at, data)
                VALUES (?, ?, ?);
                """,
                (thread.id, _format_dt(getattr(thread, "created_at", None)), _serialize(thread)),
            )

    async def load_threads(
        self, limit: int, after: str | None, order: str, context: dict
    ) -> Page[ThreadMetadata]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT data FROM threads;"
            ).fetchall()
        threads = [_deserialize(row["data"], ThreadMetadata) for row in rows]
        return self._paginate(
            threads,
            after,
            limit,
            order,
            sort_key=lambda t: t.created_at,
            cursor_key=lambda t: t.id,
        )

    async def load_thread_items(
        self, thread_id: str, after: str | None, limit: int, order: str, context: dict
    ) -> Page[ThreadItem]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT data FROM items WHERE thread_id = ?;",
                (thread_id,),
            ).fetchall()
        items = [_deserialize(row["data"], ThreadItem) for row in rows]
        return self._paginate(
            items,
            after,
            limit,
            order,
            sort_key=lambda i: i.created_at,
            cursor_key=lambda i: i.id,
        )

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: dict
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO items (id, thread_id, created_at, data)
                VALUES (?, ?, ?, ?);
                """,
                (
                    item.id,
                    thread_id,
                    _format_dt(getattr(item, "created_at", None)),
                    _serialize(item),
                ),
            )

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        await self.add_thread_item(thread_id, item, context)

    async def load_item(
        self, thread_id: str, item_id: str, context: dict
    ) -> ThreadItem:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT data FROM items WHERE thread_id = ? AND id = ?;",
                (thread_id, item_id),
            ).fetchone()
        if not row:
            raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
        return _deserialize(row["data"], ThreadItem)

    async def delete_thread(self, thread_id: str, context: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM items WHERE thread_id = ?;", (thread_id,))
            self._conn.execute("DELETE FROM threads WHERE id = ?;", (thread_id,))

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: dict
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM items WHERE thread_id = ? AND id = ?;",
                (thread_id, item_id),
            )

    def _paginate(
        self,
        rows: list,
        after: str | None,
        limit: int,
        order: str,
        sort_key,
        cursor_key,
    ) -> Page:
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

    async def save_attachment(self, attachment: Attachment, context: dict) -> None:
        raise NotImplementedError()

    async def load_attachment(self, attachment_id: str, context: dict) -> Attachment:
        raise NotImplementedError()

    async def delete_attachment(self, attachment_id: str, context: dict) -> None:
        raise NotImplementedError()


def default_sqlite_path() -> str:
    env_path = os.getenv("CHATKIT_SQLITE_PATH")
    if env_path:
        return env_path
    return str(Path(__file__).resolve().parent / "chatkit.sqlite")
