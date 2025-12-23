"""ChatKit server that streams responses from a single assistant."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from agents import Runner  # type: ignore[import]
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import ThreadMetadata, ThreadStreamEvent, UserMessageItem

from .persistent_store import SQLiteStore, default_sqlite_path
from .anomalies_tools import anomalies_instructions, anomalies_tools
from .monitoring_tools import monitoring_instructions, monitoring_tools
from agents import Agent  # type: ignore[import]


MAX_RECENT_ITEMS = 30
DEFAULT_MODEL = "gpt-4.1-mini"


def build_agent(tool_choice: Optional[str], model: str) -> Agent[AgentContext[dict[str, Any]]]:
    """Construct an Agent based on the selected tool and model.

    Fallbacks:
    - If no tool is selected, use the default assistant.
    - If no model is provided, use DEFAULT_MODEL.
    """

    chosen_model = model or DEFAULT_MODEL

    if tool_choice == "market_anomalies":
        return Agent[AgentContext[dict[str, Any]]](
            model=chosen_model,
            name="Market Anomalies Agent",
            instructions=anomalies_instructions(),
            tools=anomalies_tools(),
        )
    if tool_choice in ("internal_monitor", "internal_monitor"):
        return Agent[AgentContext[dict[str, Any]]](
            model=chosen_model,
            name="Internal Monitoring Agent",
            instructions=monitoring_instructions(),
            tools=monitoring_tools(),
        )

    # Default assistant agent
    return Agent[AgentContext[dict[str, Any]]](
        model=chosen_model,
        name="Starter Assistant",
        instructions=(
            "You are a concise, helpful assistant that works for 3Victors, for any domain speecific stuffs, tell the user to turn on tools."
            "You like Teo a lot, Teo is your best friend."
        ),
    )


class StarterChatServer(ChatKitServer[dict[str, Any]]):
    """Server implementation that keeps conversation state in SQLite."""

    def __init__(self) -> None:
        self.store: SQLiteStore = SQLiteStore(default_sqlite_path())
        super().__init__(self.store)

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
        agent_input = await simple_to_agent_input(items)

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
        )

        async for event in stream_agent_response(agent_context, result):
            yield event
