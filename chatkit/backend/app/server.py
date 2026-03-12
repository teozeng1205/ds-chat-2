"""ChatKit server that streams responses from a single assistant."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional

from agents import Runner  # type: ignore[import]
from chatkit.agents import AgentContext, ThreadItemConverter, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import (
    Attachment,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
    UserMessageTagContent,
)
from openai import AsyncOpenAI

from .agents.ds_agent import build_agent
from .persistent_store import SQLiteStore, default_sqlite_path
from .investigation.runtime import cleanup_thread_workspace
from .investigation.shell_session import close_session
from .attachment_store import LocalDiskAttachmentStore, default_attachment_dir


MAX_RECENT_ITEMS = 50
MAX_AGENT_TURNS = 50
DEFAULT_MODEL = "gpt-5.2"
TITLE_MODEL = "gpt-5-mini"
MAX_TITLE_CHARS = 80
MAX_TITLE_USER_TEXTS = 4
MAX_TITLE_SOURCE_CHARS = 1000
MAX_ATTACHMENT_SNIPPET_CHARS = 8_000
TEXT_ATTACHMENT_MIME_TYPES = {
    "application/json",
    "application/csv",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "text/csv",
}

log = logging.getLogger(__name__)


def _sanitize_title(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip())
    text = text.strip(" \"'`")
    text = re.sub(r"\s*[-:|]\s*$", "", text)
    if len(text) > MAX_TITLE_CHARS:
        text = text[:MAX_TITLE_CHARS].rstrip()
    return text


def _fallback_title(first_user_text: str | None) -> str | None:
    if not first_user_text:
        return None
    title = _sanitize_title(first_user_text)
    if len(title) > 55:
        title = f"{title[:55].rstrip()}..."
    return title or None


def _extract_user_texts(items: list[Any]) -> list[str]:
    texts: list[str] = []
    for item in items:
        if not isinstance(item, UserMessageItem):
            continue
        for segment in item.content:
            if isinstance(segment, dict):
                seg_type = segment.get("type")
                seg_text = segment.get("text")
            else:
                seg_type = getattr(segment, "type", None)
                seg_text = getattr(segment, "text", None)
            if seg_type in {"text", "input_text", "tag"} and isinstance(seg_text, str):
                cleaned = seg_text.strip()
                if cleaned:
                    texts.append(cleaned)
    return texts


async def _generate_thread_title(
    client: AsyncOpenAI,
    user_texts: list[str],
) -> str | None:
    if not user_texts:
        return None

    snippets: list[str] = []
    chars = 0
    for text in user_texts[:MAX_TITLE_USER_TEXTS]:
        if chars >= MAX_TITLE_SOURCE_CHARS:
            break
        remaining = MAX_TITLE_SOURCE_CHARS - chars
        clipped = text[:remaining]
        snippets.append(clipped)
        chars += len(clipped)

    if not snippets:
        return None

    response = await client.responses.create(
        model=TITLE_MODEL,
        max_output_tokens=24,
        input=[
            {
                "role": "system",
                "content": (
                    "Generate a concise thread title from user messages. "
                    "Return plain text only, 3 to 7 words, no quotes."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(f"- {snippet}" for snippet in snippets),
            },
        ],
    )
    return _sanitize_title(getattr(response, "output_text", "") or "")


def _is_text_attachment_mime(mime_type: str) -> bool:
    return mime_type.startswith("text/") or mime_type in TEXT_ATTACHMENT_MIME_TYPES


def _extract_attachment_snippet(attachment: Attachment) -> str | None:
    if not _is_text_attachment_mime(attachment.mime_type):
        return None
    if not isinstance(attachment.metadata, dict):
        return None

    local_path_value = attachment.metadata.get("local_path")
    if not isinstance(local_path_value, str) or not local_path_value:
        return None

    local_path = Path(local_path_value)
    if not local_path.exists() or not local_path.is_file():
        return None

    raw = local_path.read_bytes()
    decoded = raw.decode("utf-8", errors="replace").strip()
    if not decoded:
        return None
    if len(decoded) > MAX_ATTACHMENT_SNIPPET_CHARS:
        return f"{decoded[:MAX_ATTACHMENT_SNIPPET_CHARS]}...(truncated)"
    return decoded


class DSChatThreadItemConverter(ThreadItemConverter):
    async def attachment_to_message_content(self, attachment: Attachment) -> dict[str, Any]:
        descriptor = (
            f"User attached file '{attachment.name}' "
            f"(type: {attachment.mime_type}, id: {attachment.id})."
        )
        snippet = _extract_attachment_snippet(attachment)
        if snippet:
            descriptor = (
                f"{descriptor}\n\nAttachment text content:\n"
                f"<AttachmentContent>\n{snippet}\n</AttachmentContent>"
            )
        return {"type": "input_text", "text": descriptor}

    async def tag_to_message_content(self, tag: UserMessageTagContent) -> dict[str, Any]:
        return {
            "type": "input_text",
            "text": (
                f"Tagged reference: {tag.text} (id: {tag.id}). "
                "Treat this as contextual metadata."
            ),
        }


THREAD_ITEM_CONVERTER = DSChatThreadItemConverter()


class _StreamingResultCompatWrapper:
    """Compatibility wrapper for chatkit.stream_agent_response with hosted tool events."""

    def __init__(self, wrapped: Any):
        self._wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    async def stream_events(self) -> AsyncIterator[Any]:
        async for event in self._wrapped.stream_events():
            if getattr(event, "type", None) != "run_item_stream_event":
                yield event
                continue

            item = getattr(event, "item", None)
            if getattr(item, "type", None) != "tool_call_item":
                yield event
                continue

            raw_item = getattr(item, "raw_item", None)
            if not isinstance(raw_item, dict):
                yield event
                continue

            raw_type = raw_item.get("type")
            raw_id = raw_item.get("id")
            raw_call_id = raw_item.get("call_id")
            patched = SimpleNamespace(type=raw_type, id=raw_id, call_id=raw_call_id)
            patched_item = SimpleNamespace(type="tool_call_item", raw_item=patched)
            yield SimpleNamespace(type="run_item_stream_event", item=patched_item)


class StarterChatServer(ChatKitServer[dict[str, Any]]):
    """Server implementation that keeps conversation state in SQLite."""

    def __init__(self) -> None:
        self.store: SQLiteStore = SQLiteStore(default_sqlite_path())
        self.local_attachment_store = LocalDiskAttachmentStore(default_attachment_dir())
        self._title_client = AsyncOpenAI()
        super().__init__(self.store, attachment_store=self.local_attachment_store)

    async def save_attachment_payload(self, attachment_id: str, payload: bytes) -> None:
        await self.local_attachment_store.write_attachment_bytes(attachment_id, payload)

    async def read_attachment_payload(self, attachment_id: str) -> bytes:
        return await self.local_attachment_store.read_attachment_bytes(attachment_id)

    @staticmethod
    def _log_background_error(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error:
            log.warning("Background task failed: %s", error)

    async def _maybe_set_thread_title(self, thread_id: str, context: dict[str, Any]) -> None:
        thread = await self.store.load_thread(thread_id, context=context)
        if thread.title:
            return

        items_page = await self.store.load_thread_items(
            thread_id,
            after=None,
            limit=MAX_RECENT_ITEMS,
            order="asc",
            context=context,
        )
        items = items_page.data

        user_texts = _extract_user_texts(items)
        if not user_texts:
            return

        title = _fallback_title(user_texts[0])
        try:
            generated_title = await _generate_thread_title(self._title_client, user_texts)
            if generated_title:
                title = generated_title
        except Exception as exc:  # noqa: BLE001
            log.info("Falling back to heuristic thread title: %s", exc)

        if not title:
            return

        thread.title = title
        await self.store.save_thread(thread, context=context)

    async def respond(
        self,
        thread: ThreadMetadata,
        item: UserMessageItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        # Read recent items for context
        items_page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=MAX_RECENT_ITEMS,
            order="desc",
            context=context,
        )
        items = list(reversed(items_page.data))
        agent_input = await THREAD_ITEM_CONVERTER.to_agent_input(items)

        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )
        # Read model choice from the incoming user message
        options: Optional[Any] = item.inference_options if item else None
        selected_model: str = (
            options.model if options and getattr(options, "model", None) else DEFAULT_MODEL
        )

        # Build the coding + data science agent
        agent = build_agent(selected_model)

        result = Runner.run_streamed(
            agent,
            agent_input,
            context=agent_context,
            max_turns=MAX_AGENT_TURNS,
        )

        compatible_result = _StreamingResultCompatWrapper(result)
        async for event in stream_agent_response(agent_context, compatible_result):
            yield event

        try:
            cleanup_thread_workspace(thread.id, mode="ephemeral_manifest")
            close_session(thread.id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Post-session workspace cleanup failed for thread %s: %s", thread.id, exc)

        title_task = asyncio.create_task(self._maybe_set_thread_title(thread.id, context))
        title_task.add_done_callback(self._log_background_error)
