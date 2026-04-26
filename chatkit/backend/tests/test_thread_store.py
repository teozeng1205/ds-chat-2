from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from chatkit.types import InferenceOptions, ThreadMetadata, UserMessageItem

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
