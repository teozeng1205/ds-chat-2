"""Top-level orchestrator that routes to investigation or codebase agents."""

from __future__ import annotations

from typing import Any, Optional

from agents import Agent
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from chatkit.agents import AgentContext

from ..investigation.runtime import is_investigation_engine_enabled
from .codebase_agent import build_codebase_agent
from .investigation_agent import build_investigation_agent

DEFAULT_MODEL = "gpt-4.1-mini"


def _build_orchestrator(
    model: str,
    investigation_agent: Agent[AgentContext[dict[str, Any]]],
    codebase_agent: Agent[AgentContext[dict[str, Any]]],
) -> Agent[AgentContext[dict[str, Any]]]:
    instructions = prompt_with_handoff_instructions(
        (
            "You are the routing orchestrator for this chat.\n"
            "Choose the best specialist agent for the user's request.\n"
            "Route monitoring, anomaly, provider/site/customer impact, SQL/S3 dataframe investigations, and data issue analysis to Investigation Operator Agent.\n"
            "Route codebase understanding, architecture walkthrough, repository exploration, local shell, python execution, plotting, and ad-hoc analysis requests to Codebase Explanation Agent.\n"
            "If the request is ambiguous, ask one short clarification question before handing off."
        )
    )
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Multi-Agent Orchestrator",
        instructions=instructions,
        handoffs=[investigation_agent, codebase_agent],
    )


def build_agent(tool_choice: Optional[str], model: str) -> Agent[AgentContext[dict[str, Any]]]:
    chosen_model = model or DEFAULT_MODEL
    codebase_agent = build_codebase_agent(chosen_model)

    # During migration, keep investigation disabled switch support, but default enabled.
    if not is_investigation_engine_enabled():
        return codebase_agent

    investigation_agent = build_investigation_agent(chosen_model)

    # Compatibility mapping for historical tool choice IDs.
    if tool_choice in {
        "internal_investigation",
        "internal_monitoring",
        "market_anomalies",
        "knowledge_planner",
        "data_access",
        "analysis",
        "synthesis",
    }:
        return investigation_agent
    if tool_choice == "codebase_explainer":
        return codebase_agent

    return _build_orchestrator(
        model=chosen_model,
        investigation_agent=investigation_agent,
        codebase_agent=codebase_agent,
    )
