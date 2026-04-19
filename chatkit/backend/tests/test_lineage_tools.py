"""Smoke tests for lineage_tools registration.

End-to-end `trace_pipeline` behavior is covered by the graph_store
tests (neighbors / resolve / stats). Here we confirm the @function_tool
wrapper is registered correctly in the agent's tool list.
"""

from __future__ import annotations


def test_lineage_tools_factory_registers_trace_pipeline() -> None:
    from app.tools.lineage_tools import lineage_tools
    tools = lineage_tools()
    names = {getattr(t, "name", None) for t in tools}
    assert "trace_pipeline" in names


def test_tool_description_covers_usage() -> None:
    from app.tools.lineage_tools import lineage_tools
    [t] = lineage_tools()
    desc = (getattr(t, "description", None) or "").lower()
    # Short description should hit at least these concepts
    assert "pipeline" in desc or "lineage" in desc
    assert "upstream" in desc
    assert "table" in desc


def test_agent_registers_trace_pipeline() -> None:
    """The agent build_agent() wires lineage_tools in. Fail loudly if a
    refactor accidentally drops it."""
    from app.agents.ds_agent import build_agent
    agent = build_agent("gpt-5.4-mini")
    names = {getattr(t, "name", None) for t in agent.tools}
    assert "trace_pipeline" in names


def test_skill_file_shipped() -> None:
    from pathlib import Path
    backend_root = Path(__file__).resolve().parents[1]
    skill = backend_root / "skills" / "pipeline_lineage.md"
    assert skill.exists()
    body = skill.read_text(encoding="utf-8")
    assert "trace_pipeline" in body
    assert "upstream" in body
