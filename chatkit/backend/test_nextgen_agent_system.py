from __future__ import annotations

from agents import Agent

from app.nextgen_agent_system import build_agent


def _handoff_names(agent: Agent) -> set[str]:
    return {str(getattr(handoff, "name", "")) for handoff in agent.handoffs}


def test_orchestrator_wires_full_pipeline_handoffs() -> None:
    orchestrator = build_agent(None, "gpt-4.1-mini")
    assert orchestrator.name == "DS Chat Next-Gen Orchestrator"

    specialist_names = _handoff_names(orchestrator)
    assert "Knowledge Planner Agent" in specialist_names
    assert "Data Access Agent" in specialist_names
    assert "Analysis Agent" in specialist_names
    assert "Synthesis Agent" in specialist_names

    planner = next(h for h in orchestrator.handoffs if getattr(h, "name", "") == "Knowledge Planner Agent")
    data_access = next(h for h in orchestrator.handoffs if getattr(h, "name", "") == "Data Access Agent")
    analysis = next(h for h in orchestrator.handoffs if getattr(h, "name", "") == "Analysis Agent")
    synthesis = next(h for h in orchestrator.handoffs if getattr(h, "name", "") == "Synthesis Agent")

    assert {"Data Access Agent", "Synthesis Agent"}.issubset(_handoff_names(planner))
    assert {"Analysis Agent", "Knowledge Planner Agent"}.issubset(_handoff_names(data_access))
    assert {"Data Access Agent", "Synthesis Agent"}.issubset(_handoff_names(analysis))
    assert {"Analysis Agent"}.issubset(_handoff_names(synthesis))


def test_tool_choice_planner_keeps_execution_handoffs() -> None:
    planner = build_agent("knowledge_planner", "gpt-4.1-mini")
    assert planner.name == "Knowledge Planner Agent"
    assert {"Data Access Agent", "Synthesis Agent"}.issubset(_handoff_names(planner))
