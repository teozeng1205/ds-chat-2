"""ChatKit server that streams responses from a single assistant."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from agents import Runner  # type: ignore[import]
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions  # type: ignore[import]
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

from .persistent_store import SQLiteStore, default_sqlite_path
from .anomalies_tools import anomalies_instructions, anomalies_tools
from .internal_monitoring_tools import (
    internal_monitoring_instructions,
    internal_monitoring_tools,
)
from .codebase_tools import (
    codebase_explainer_instructions,
    codebase_explainer_tools,
)
from .attachment_store import LocalDiskAttachmentStore, default_attachment_dir
from agents import Agent  # type: ignore[import]


MAX_RECENT_ITEMS = 50
MAX_AGENT_TURNS = 50
DEFAULT_MODEL = "gpt-4.1-mini"
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


def _build_analytics_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Analytics Agent",
        handoff_description=(
            "Handles market/customer analytics anomaly analysis (e.g., AA/B6) and market-level anomaly summaries."
        ),
        instructions=anomalies_instructions(),
        tools=anomalies_tools(),
    )


def _build_internal_monitoring_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Internal Monitoring Agent",
        handoff_description=(
            "Handles internal monitoring via S3 anomaly partitions and SQL issue-analysis tools."
        ),
        instructions=internal_monitoring_instructions(),
        tools=internal_monitoring_tools(),
    )


def _build_codebase_explainer_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Codebase Explanation Agent",
        handoff_description=(
            "Handles codebase architecture/explanation and Codex-like sandboxed tooling under ~/git."
        ),
        instructions=codebase_explainer_instructions(),
        tools=codebase_explainer_tools(),
    )


def _build_orchestrator_agent(
    model: str,
    analytics_agent: Agent[AgentContext[dict[str, Any]]],
    internal_agent: Agent[AgentContext[dict[str, Any]]],
    codebase_agent: Agent[AgentContext[dict[str, Any]]],
) -> Agent[AgentContext[dict[str, Any]]]:
    orchestrator_instructions = prompt_with_handoff_instructions(
        (
            "You are the routing orchestrator for this chat.\n"
            "Choose the best specialist agent for the user's request.\n"
            "Route market/customer anomaly requests to Analytics Agent.\n"
            "Route provider/site/customer/late-request monitoring anomaly requests to Internal Monitoring Agent.\n"
            "Route codebase understanding, architecture walkthrough, repository exploration, local shell, and python sandbox requests to Codebase Explanation Agent.\n"
            "If the request is ambiguous, ask one short clarification question before handing off."
        )
    )
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Multi-Agent Orchestrator",
        instructions=orchestrator_instructions,
        handoffs=[analytics_agent, internal_agent, codebase_agent],
    )


def build_agent(tool_choice: Optional[str], model: str) -> Agent[AgentContext[dict[str, Any]]]:
    """Construct an agent for the selected mode; default to orchestrator handoffs."""
    chosen_model = model or DEFAULT_MODEL
    analytics_agent = _build_analytics_agent(chosen_model)
    internal_agent = _build_internal_monitoring_agent(chosen_model)
    codebase_agent = _build_codebase_explainer_agent(chosen_model)

    # Optional direct routing for compatibility if caller explicitly sets tool_choice.
    if tool_choice == "market_anomalies":
        return analytics_agent
    if tool_choice == "internal_monitoring":
        return internal_agent
    if tool_choice == "codebase_explainer":
        return codebase_agent

    return _build_orchestrator_agent(
        model=chosen_model,
        analytics_agent=analytics_agent,
        internal_agent=internal_agent,
        codebase_agent=codebase_agent,
    )


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
        # Read tool and model choices from the incoming user message
        # Inference options may be absent; treat as an untyped payload to avoid tight coupling
        options: Optional[Any] = item.inference_options if item else None
        selected_model: str = (
            options.model if options and getattr(options, "model", None) else DEFAULT_MODEL
        )
        tool_choice_id: Optional[str] = (
            options.tool_choice.id
            if options and getattr(options, "tool_choice", None)
            else None
        )

        # Build the appropriate agent based on user selections
        agent = build_agent(tool_choice_id, selected_model)

        result = Runner.run_streamed(
            agent,
            agent_input,
            context=agent_context,
            max_turns=MAX_AGENT_TURNS,
        )

        async for event in stream_agent_response(agent_context, result):
            yield event

        title_task = asyncio.create_task(self._maybe_set_thread_title(thread.id, context))
        title_task.add_done_callback(self._log_background_error)
