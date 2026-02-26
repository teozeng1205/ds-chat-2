"""Codebase explainer agent builder."""

from __future__ import annotations

from typing import Any

from agents import Agent
from chatkit.agents import AgentContext

from ..codebase_tools import codebase_explainer_instructions, codebase_explainer_tools


def _supports_local_shell(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("codex-mini")


def build_codebase_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
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
