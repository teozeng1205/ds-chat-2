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
            "Route SQL/S3/dataframe/internal data investigations to Investigation Operator Agent.\n"
            "Route repository/code explanation and local code analysis requests to Codebase Explanation Agent.\n"
            "Ask one concise clarification only if routing is ambiguous."
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

    if not is_investigation_engine_enabled():
        return codebase_agent

    investigation_agent = build_investigation_agent(chosen_model)

    if tool_choice == "codebase_explainer":
        return codebase_agent
    if tool_choice == "investigation":
        return investigation_agent

    return _build_orchestrator(
        model=chosen_model,
        investigation_agent=investigation_agent,
        codebase_agent=codebase_agent,
    )
