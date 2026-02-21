"""ChatKit server that streams responses from a single assistant."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from agents import Runner  # type: ignore[import]
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions  # type: ignore[import]
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import ThreadMetadata, ThreadStreamEvent, UserMessageItem

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
from agents import Agent  # type: ignore[import]


MAX_RECENT_ITEMS = 30
DEFAULT_MODEL = "gpt-4.1-mini"


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
