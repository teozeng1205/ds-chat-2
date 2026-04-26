"""In-memory ChatKit store for active server sessions."""

from __future__ import annotations

from typing import Any, Callable

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata


class InMemoryStore(Store[dict]):
    """Minimal process-local store compatible with ChatKit's Store interface.

    The app only needs active conversation state while the backend process is
    running. Keeping this in memory avoids accumulating thread history records
    or SQLite state on disk.
    """

    def __init__(self) -> None:
        self._threads: dict[str, ThreadMetadata] = {}
        self._items: dict[str, dict[str, ThreadItem]] = {}
        self._attachments: dict[str, Attachment] = {}

    async def load_thread(self, thread_id: str, context: dict) -> ThreadMetadata:
        try:
            return self._threads[thread_id]
        except KeyError as exc:
            raise NotFoundError(f"Thread {thread_id} not found") from exc

    async def save_thread(self, thread: ThreadMetadata, context: dict) -> None:
        self._threads[thread.id] = thread
        self._items.setdefault(thread.id, {})

    async def load_threads(
        self, limit: int, after: str | None, order: str, context: dict
    ) -> Page[ThreadMetadata]:
        return self._paginate(
            list(self._threads.values()),
            after=after,
            limit=limit,
            order=order,
            sort_key=lambda thread: thread.created_at,
            cursor_key=lambda thread: thread.id,
        )

    async def load_thread_items(
        self, thread_id: str, after: str | None, limit: int, order: str, context: dict
    ) -> Page[ThreadItem]:
        if thread_id not in self._threads:
            raise NotFoundError(f"Thread {thread_id} not found")
        return self._paginate(
            list(self._items.get(thread_id, {}).values()),
            after=after,
            limit=limit,
            order=order,
            sort_key=lambda item: item.created_at,
            cursor_key=lambda item: item.id,
        )

    async def add_thread_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        if thread_id not in self._threads:
            raise NotFoundError(f"Thread {thread_id} not found")
        self._items.setdefault(thread_id, {})[item.id] = item

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        await self.add_thread_item(thread_id, item, context)

    async def load_item(self, thread_id: str, item_id: str, context: dict) -> ThreadItem:
        try:
            return self._items[thread_id][item_id]
        except KeyError as exc:
            raise NotFoundError(f"Item {item_id} not found in thread {thread_id}") from exc

    async def delete_thread(self, thread_id: str, context: dict) -> None:
        self._threads.pop(thread_id, None)
        self._items.pop(thread_id, None)

    async def delete_thread_item(self, thread_id: str, item_id: str, context: dict) -> None:
        self._items.get(thread_id, {}).pop(item_id, None)

    async def save_attachment(self, attachment: Attachment, context: dict) -> None:
        self._attachments[attachment.id] = attachment

    async def load_attachment(self, attachment_id: str, context: dict) -> Attachment:
        try:
            return self._attachments[attachment_id]
        except KeyError as exc:
            raise NotFoundError(f"Attachment {attachment_id} not found") from exc

    async def delete_attachment(self, attachment_id: str, context: dict) -> None:
        self._attachments.pop(attachment_id, None)

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
