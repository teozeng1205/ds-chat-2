"""Reviewer sub-agent.

Fast, cheap (gpt-5.4-mini) grounded-verdict agent. Given the user
question, the final answer, the tool-call log, and any dataset
manifests, it returns a structured verdict the main agent must pass
before finalizing responses that cite numeric SQL results.

Tools: none. The reviewer reasons over the inputs alone; it is not
allowed to run new SQL or fetch fresh data — its job is to assess
whether the answer is grounded in what the main agent already pulled.

Output contract (JSON):
{
  "verdict":    "pass" | "soft_fail" | "hard_fail",
  "confidence": 0.0 .. 1.0,
  "concerns": [
    {"kind": "unsupported_claim" | "contradiction" | "stale_data" |
             "missing_partition" | "wrong_table" | "math_error" |
             "other",
     "text": "..."}
  ]
}
"""

from __future__ import annotations

from typing import Any

from agents import Agent
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

DEFAULT_REVIEWER_MODEL = "gpt-5.4-mini"
DEFAULT_REVIEWER_MAX_TURNS = 2
REVIEWER_TOOL_NAME = "review_answer"


REVIEWER_INSTRUCTIONS = """You are the DS Chat reviewer. You do not have tools.
You receive a user question, the main agent's final answer, and the tool-call log
(which tools were called with what arguments and what they returned).

Your job: decide whether the answer is grounded in what the tools actually produced.

Check each numeric claim in the answer against values visible in the tool results.
Flag:
  - unsupported_claim  — the answer asserts a number or fact the tools did not return.
  - contradiction      — the answer contradicts a tool result.
  - stale_data         — the tools returned dates older than the user implies.
  - missing_partition  — the SQL ignored a partition filter on a known partitioned table.
  - wrong_table        — the tools queried a table that doesn't fit the question.
  - math_error         — arithmetic in the answer doesn't match the tool result.
  - other              — other grounding problems.

Return exactly this JSON and nothing else:

{
  "verdict":    "pass" | "soft_fail" | "hard_fail",
  "confidence": 0.0 to 1.0,
  "concerns":   [{"kind": "...", "text": "..."}]
}

Rules:
- pass: the answer is grounded, no material concerns.
- soft_fail: the answer is mostly right but has minor issues (the main agent may
  revise or caveat).
- hard_fail: the answer is wrong or unsupported enough that it must be redone.
- confidence reflects how sure you are of your verdict.
- Never speculate about data the tools didn't return.
- Never write outside the JSON envelope.
"""


def build_reviewer_agent(model: str = DEFAULT_REVIEWER_MODEL) -> Agent[Any]:
    """Construct the reviewer sub-agent (no tools — pure reasoning)."""
    return Agent(
        model=OpenAIResponsesModel(model=model, openai_client=AsyncOpenAI()),
        name="DS Chat Reviewer",
        instructions=REVIEWER_INSTRUCTIONS,
        tools=[],
    )


def reviewer_tool_description() -> str:
    return (
        "Review the final answer against the tool-call log and return a structured "
        "verdict. Call before finalizing any response that cites numeric SQL results "
        "or specific data points. Returns JSON: {verdict, confidence, concerns[]}."
    )


def as_agent_tool(
    reviewer: Agent[Any],
    *,
    tool_name: str = REVIEWER_TOOL_NAME,
    max_turns: int = DEFAULT_REVIEWER_MAX_TURNS,
) -> Any:
    return reviewer.as_tool(
        tool_name=tool_name,
        tool_description=reviewer_tool_description(),
        max_turns=max_turns,
    )


# ── Verdict parsing helper (so the main agent / server can act on the result) ──

def parse_verdict(raw: str) -> dict[str, Any]:
    """Best-effort parse of a reviewer response into a typed dict.

    Accepts either a bare JSON object or JSON embedded in surrounding
    text. Returns a safe default shape on any parse failure so callers
    don't need to branch on exceptions.
    """
    import json as _json
    import re as _re

    text = (raw or "").strip()
    match = _re.search(r"\{.*\}", text, flags=_re.S)
    if not match:
        return {"verdict": "soft_fail", "confidence": 0.0, "concerns": [
            {"kind": "other", "text": "reviewer returned no JSON"}
        ]}
    try:
        parsed = _json.loads(match.group(0))
    except Exception:
        return {"verdict": "soft_fail", "confidence": 0.0, "concerns": [
            {"kind": "other", "text": "reviewer returned invalid JSON"}
        ]}

    verdict = str(parsed.get("verdict") or "soft_fail").lower()
    if verdict not in {"pass", "soft_fail", "hard_fail"}:
        verdict = "soft_fail"
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    concerns_raw = parsed.get("concerns") or []
    concerns: list[dict[str, str]] = []
    for c in concerns_raw if isinstance(concerns_raw, list) else []:
        if not isinstance(c, dict):
            continue
        concerns.append({
            "kind": str(c.get("kind") or "other"),
            "text": str(c.get("text") or "").strip(),
        })
    return {"verdict": verdict, "confidence": confidence, "concerns": concerns}


__all__ = [
    "REVIEWER_INSTRUCTIONS",
    "REVIEWER_TOOL_NAME",
    "DEFAULT_REVIEWER_MAX_TURNS",
    "DEFAULT_REVIEWER_MODEL",
    "build_reviewer_agent",
    "reviewer_tool_description",
    "as_agent_tool",
    "parse_verdict",
]
