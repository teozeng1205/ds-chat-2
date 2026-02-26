"""Investigation operator agent builder."""

from __future__ import annotations

from typing import Any

from agents import Agent
from chatkit.agents import AgentContext

from ..tools.investigation_tools import investigation_instructions, investigation_tools


def build_investigation_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="Investigation Operator Agent",
        handoff_description=(
            "Handles generalized internal investigations across SQL, S3, local knowledge browsing, and pandas analysis."
        ),
        instructions=investigation_instructions(),
        tools=investigation_tools(),
    )
