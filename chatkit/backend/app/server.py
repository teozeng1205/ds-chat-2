"""ChatKit server that streams responses from a single assistant."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional

from agents import Runner  # type: ignore[import]
from chatkit.agents import AgentContext, ThreadItemConverter, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import (
    Attachment,
    AudioInput,
    ThreadMetadata,
    ThreadStreamEvent,
    TranscriptionResult,
    UserMessageItem,
    UserMessageTagContent,
)

from .attachment_store import LocalDiskAttachmentStore, default_attachment_dir
from .agents.ds_agent import build_agent
from .investigation.runtime import cleanup_thread_workspace
from .investigation.shell_session import close_session
from .sqlite_thread_store import SqliteThreadStore


MAX_RECENT_ITEMS = 50
MAX_AGENT_TURNS = 600
DEFAULT_MODEL = "gpt-5.5"
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
    """Server implementation for active in-process conversations."""

    def __init__(self) -> None:
        # Persistent store so the history panel can list/reopen past threads.
        # Full thread history is retained; the agent context is bounded by the
        # MAX_RECENT_ITEMS limit on the load_thread_items query, not by trimming.
        self.store = SqliteThreadStore()
        self.local_attachment_store = LocalDiskAttachmentStore(default_attachment_dir())
        super().__init__(self.store, attachment_store=self.local_attachment_store)

    async def transcribe(self, audio_input: AudioInput, context: dict[str, Any]) -> TranscriptionResult:
        import io
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        result = await client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.webm", io.BytesIO(audio_input.data), audio_input.mime_type),
        )
        return TranscriptionResult(text=result.text)

    async def save_attachment_payload(self, attachment_id: str, payload: bytes) -> None:
        await self.local_attachment_store.write_attachment_bytes(attachment_id, payload)

    async def read_attachment_payload(self, attachment_id: str) -> bytes:
        return await self.local_attachment_store.read_attachment_bytes(attachment_id)

    async def get_session_meta(self, thread_id: str, context: dict[str, Any]) -> dict[str, Any]:
        """Return lightweight session metadata for the active thread."""
        items_page = await self.store.load_thread_items(
            thread_id,
            after=None,
            limit=MAX_RECENT_ITEMS,
            order="desc",
            context=context,
        )
        items = items_page.data
        turn_count = sum(1 for item in items if not isinstance(item, UserMessageItem))
        model: str | None = None
        for item in items:  # desc order — most recent first
            if isinstance(item, UserMessageItem):
                opts = getattr(item, "inference_options", None)
                if opts:
                    m = getattr(opts, "model", None)
                    if m:
                        model = m
                        break

        return {"model": model, "turn_count": turn_count}

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
