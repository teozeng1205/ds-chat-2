"""Planner sub-agent with read-only exploration tools.

Tools given:
  - read_file, list_dir        (filesystem probing)
  - search_kb                   (knowledge base retrieval)
  - inspect_table               (schema / partition info)
  - resolve_codes               (provider / site / customer code lookup)

Deliberately NOT given:
  - execute_sql / fetch_s3      (side effects: workspace datasets)
  - bash / edit_file / git      (side effects on the machine)
  - publish_image / download    (UI side effects)
  - apply_patch                 (mutates files; planner is read-only)

Output contract: a JSON-ish structured plan the main agent can execute
deterministically. Prompt is tight and discourages chain-of-thought
leakage.

The main agent invokes this planner through ``as_agent_tool``.
"""

from __future__ import annotations

from typing import Any

from agents import Agent
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

from ..tools.investigation_tools import (
    inspect_table,
    resolve_codes,
    search_kb,
)
from ..tools.shell_tools import list_dir, read_file

DEFAULT_PLANNER_MODEL = "gpt-5.4"
DEFAULT_PLANNER_MAX_TURNS = 5
PLANNER_TOOL_NAME = "plan_task"


PLANNER_INSTRUCTIONS = """You are the DS Chat planner. Your job is to turn a user's
ambiguous request into a concrete, executable plan the main agent can follow.

You have READ-ONLY exploration tools: read_file, list_dir, search_kb, inspect_table,
resolve_codes. Use them to ground the plan in what the environment actually contains
— never guess at table names, column names, file paths, or code references.

Before emitting the plan:
- If the question mentions a provider / site / customer, call resolve_codes.
- If it names a table (even implicitly, like "the anomalies table"), call
  inspect_table on a likely match to confirm columns and partitions.
- If relevant docs might exist, call search_kb with the domain terms.
- If a file path is mentioned, use read_file / list_dir to check it exists.

Emit a single JSON object on stdout with this shape, and nothing else:

{
  "plan": [
    {"step": 1, "tool": "<tool_name>", "inputs": {...}, "expected_outcome": "..."},
    ...
  ],
  "assumptions": ["..."],
  "risks":       ["..."]
}

Rules:
- Max 10 steps. Prefer the fewest that reliably solve the task.
- Each step must name a real tool the main agent has: execute_sql, fetch_s3,
  inspect_table, search_kb, resolve_codes, run_python (via bash), publish_image,
  read_file, edit_file, list_dir, bash, git, fetch_url, render_image, download_file.
- Inputs must be concrete (table names, bucket names, customer codes), not placeholders.
- If the request is trivial (single tool call), return a one-step plan — don't pad.
- If it is impossible or out of scope, return an empty plan and explain in "risks".
- Never include chain-of-thought outside the JSON envelope.
"""


def build_planner_agent(model: str = DEFAULT_PLANNER_MODEL) -> Agent[Any]:
    """Construct the planner sub-agent.

    The planner has no direct access to side-effectful tools; it can
    only read the environment (files, KB, schema, codes). The main
    agent executes the plan afterwards.
    """
    return Agent(
        model=OpenAIResponsesModel(model=model, openai_client=AsyncOpenAI()),
        name="DS Chat Planner",
        instructions=PLANNER_INSTRUCTIONS,
        tools=[
            read_file,
            list_dir,
            search_kb,
            inspect_table,
            resolve_codes,
        ],
    )


def planner_tool_description() -> str:
    return (
        "Generate a concrete, grounded execution plan for complex multi-step tasks "
        "(5+ steps, or when the approach is non-obvious). The planner probes the "
        "environment (files, KB, schema, codes) with read-only tools before emitting "
        "a numbered plan with tool, inputs, and expected outcome per step. Returns "
        "a JSON envelope: plan[], assumptions[], risks[]."
    )


def as_agent_tool(
    planner: Agent[Any],
    *,
    tool_name: str = PLANNER_TOOL_NAME,
    max_turns: int = DEFAULT_PLANNER_MAX_TURNS,
) -> Any:
    """Wrap the planner as a tool for the main agent to invoke.

    Returns whatever `Agent.as_tool(...)` returns in the SDK version
    that's installed (signature has been stable across 0.9.x → 0.13.x).
    """
    return planner.as_tool(
        tool_name=tool_name,
        tool_description=planner_tool_description(),
        max_turns=max_turns,
    )


__all__ = [
    "PLANNER_INSTRUCTIONS",
    "PLANNER_TOOL_NAME",
    "DEFAULT_PLANNER_MAX_TURNS",
    "DEFAULT_PLANNER_MODEL",
    "build_planner_agent",
    "planner_tool_description",
    "as_agent_tool",
]
