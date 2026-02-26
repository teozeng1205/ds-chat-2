from __future__ import annotations

from app.agents.orchestrator import build_agent


def _handoff_names(agent) -> set[str]:
    return {str(getattr(handoff, "name", "")) for handoff in getattr(agent, "handoffs", [])}


def test_orchestrator_routes_to_investigation_and_codebase() -> None:
    orchestrator = build_agent(None, "gpt-4.1-mini")
    assert orchestrator.name == "Multi-Agent Orchestrator"
    names = _handoff_names(orchestrator)
    assert "Investigation Operator Agent" in names
    assert "Codebase Explanation Agent" in names


def test_tool_choice_compatibility_maps_legacy_ids_to_investigation() -> None:
    investigation = build_agent("internal_monitoring", "gpt-4.1-mini")
    assert investigation.name == "Investigation Operator Agent"


def test_tool_choice_codebase_routes_to_codebase_agent() -> None:
    codebase = build_agent("codebase_explainer", "gpt-4.1-mini")
    assert codebase.name == "Codebase Explanation Agent"
