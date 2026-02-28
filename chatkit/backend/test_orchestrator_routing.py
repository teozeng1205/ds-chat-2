from __future__ import annotations

from app.agents.investigation_agent import build_investigation_agent


def test_build_investigation_agent() -> None:
    agent = build_investigation_agent("gpt-4.1-mini")
    assert agent.name == "DS Chat Investigation Agent"
    tool_names = {str(getattr(tool, "name", "")) for tool in agent.tools}
    assert "execute_sql" in tool_names
    assert "resolve_codes" in tool_names
    assert len(tool_names) == 6


def test_investigation_agent_has_rich_instructions() -> None:
    agent = build_investigation_agent("gpt-4.1-mini")
    instructions = agent.instructions
    assert "partition" in instructions.lower()
    assert "redshift_analytics" in instructions
    assert "redshift_core" in instructions
    assert "mysql_priceeye" in instructions
