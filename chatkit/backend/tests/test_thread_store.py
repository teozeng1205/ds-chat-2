from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from chatkit.types import InferenceOptions, ThreadMetadata, UserMessageItem

from app.sqlite_thread_store import SqliteThreadStore
from app.thread_store import InMemoryStore


def _thread() -> ThreadMetadata:
    return ThreadMetadata(id="thread-1", created_at=datetime.now(timezone.utc))


def _message(idx: int) -> UserMessageItem:
    return UserMessageItem(
        id=f"item-{idx}",
        thread_id="thread-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=idx),
        content=[{"type": "input_text", "text": f"message {idx}"}],
        inference_options=InferenceOptions(model="gpt-5.4"),
    )


def test_store_trims_old_thread_items_when_bounded() -> None:
    async def run_case() -> None:
        store = InMemoryStore(max_items_per_thread=3)
        context = {}

        await store.save_thread(_thread(), context=context)
        for idx in range(5):
            await store.add_thread_item("thread-1", _message(idx), context=context)

        page = await store.load_thread_items(
            "thread-1",
            after=None,
            limit=10,
            order="asc",
            context=context,
        )

        assert [item.id for item in page.data] == ["item-2", "item-3", "item-4"]

    asyncio.run(run_case())


def test_sqlite_store_persists_across_instances(tmp_path) -> None:
    """Threads + items survive a 'restart' (a fresh store on the same file)."""

    async def run_case() -> None:
        db = tmp_path / "threads.sqlite"
        context = {}

        store = SqliteThreadStore(db_path=db)
        await store.save_thread(_thread(), context=context)
        for idx in range(5):
            await store.add_thread_item("thread-1", _message(idx), context=context)

        # New instance on the same file == process restart.
        reopened = SqliteThreadStore(db_path=db)

        threads = await reopened.load_threads(limit=10, after=None, order="desc", context=context)
        assert [t.id for t in threads.data] == ["thread-1"]

        page = await reopened.load_thread_items(
            "thread-1", after=None, limit=10, order="asc", context=context
        )
        assert [item.id for item in page.data] == [f"item-{i}" for i in range(5)]
        # Round-tripped content + typed fields survive serialization.
        first = page.data[0]
        assert first.content[0].text == "message 0"
        assert first.inference_options.model == "gpt-5.4"

        # Pagination cursor behaves like the in-memory store.
        first_two = await reopened.load_thread_items(
            "thread-1", after=None, limit=2, order="asc", context=context
        )
        assert [i.id for i in first_two.data] == ["item-0", "item-1"]
        assert first_two.has_more is True

    asyncio.run(run_case())
