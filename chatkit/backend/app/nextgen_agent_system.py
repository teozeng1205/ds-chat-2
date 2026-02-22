"""Agent graph assembly for DS Chat Next-Gen."""

from __future__ import annotations

from typing import Any

from agents import Agent
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from chatkit.agents import AgentContext

from .codebase_tools import codebase_explainer_instructions, codebase_explainer_tools
from .nextgen_tools import (
    analysis_instructions,
    analysis_tools,
    data_access_instructions,
    data_access_tools,
    knowledge_planner_instructions,
    planner_tools,
    synthesis_instructions,
    synthesis_tools,
)


def _supports_local_shell(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("codex-mini")


def _build_knowledge_planner_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Knowledge Planner Agent",
        handoff_description=(
            "Builds KB-driven investigation plans, resolves entities, and enforces partition-required clarifications."
        ),
        instructions=knowledge_planner_instructions(),
        tools=planner_tools(),
    )


def _build_data_access_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Data Access Agent",
        handoff_description=(
            "Executes partition-safe SQL/S3 extraction via ds-threevictors and materializes datasets to local workspace."
        ),
        instructions=data_access_instructions(),
        tools=data_access_tools(),
    )


def _build_analysis_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Analysis Agent",
        handoff_description=(
            "Runs offline dataframe analysis and joins local dataset artifacts for issue investigation."
        ),
        instructions=analysis_instructions(),
        tools=analysis_tools(),
    )


def _build_synthesis_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Synthesis Agent",
        handoff_description=(
            "Composes evidence-backed conclusions and caveats from local analysis outputs and manifests."
        ),
        instructions=synthesis_instructions(),
        tools=synthesis_tools(),
    )


def _build_codebase_explainer_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    include_shell = _supports_local_shell(model)
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Codebase Explanation Agent",
        handoff_description=(
            "Handles codebase architecture/explanation and Codex-like sandboxed tooling under /git."
        ),
        instructions=codebase_explainer_instructions(include_shell=include_shell),
        tools=codebase_explainer_tools(include_shell=include_shell),
    )


def _build_orchestrator_agent(
    *,
    model: str,
    planner_agent: Agent[AgentContext[dict[str, Any]]],
    data_access_agent: Agent[AgentContext[dict[str, Any]]],
    analysis_agent: Agent[AgentContext[dict[str, Any]]],
    synthesis_agent: Agent[AgentContext[dict[str, Any]]],
    codebase_agent: Agent[AgentContext[dict[str, Any]]],
) -> Agent[AgentContext[dict[str, Any]]]:
    instructions = prompt_with_handoff_instructions(
        (
            "You are the DS Chat Next-Gen Orchestrator. "
            "Use multi-agent handoffs to investigate issues across DB and S3 data. "
            "Route planning/KB/table selection/entity resolution to Knowledge Planner Agent. "
            "Route extraction to Data Access Agent. "
            "Route dataframe processing and statistics to Analysis Agent. "
            "Route final conclusions to Synthesis Agent. "
            "Route repository/codebase questions to Codebase Explanation Agent. "
            "If required partition values are missing, ask one concise clarification before extraction. "
            "Never bypass partition-safe workflows. Environment is fixed to 3VDEV."
        )
    )
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="DS Chat Next-Gen Orchestrator",
        instructions=instructions,
        handoffs=[planner_agent, data_access_agent, analysis_agent, synthesis_agent, codebase_agent],
    )


def build_agent(tool_choice: str | None, model: str) -> Agent[AgentContext[dict[str, Any]]]:
    chosen_model = model or "gpt-4.1-mini"

    planner_agent = _build_knowledge_planner_agent(chosen_model)
    data_access_agent = _build_data_access_agent(chosen_model)
    analysis_agent = _build_analysis_agent(chosen_model)
    synthesis_agent = _build_synthesis_agent(chosen_model)
    codebase_agent = _build_codebase_explainer_agent(chosen_model)

    if tool_choice in {"knowledge_planner", "market_anomalies", "internal_monitoring"}:
        return planner_agent
    if tool_choice == "data_access":
        return data_access_agent
    if tool_choice == "analysis":
        return analysis_agent
    if tool_choice == "synthesis":
        return synthesis_agent
    if tool_choice == "codebase_explainer":
        return codebase_agent

    return _build_orchestrator_agent(
        model=chosen_model,
        planner_agent=planner_agent,
        data_access_agent=data_access_agent,
        analysis_agent=analysis_agent,
        synthesis_agent=synthesis_agent,
        codebase_agent=codebase_agent,
    )
