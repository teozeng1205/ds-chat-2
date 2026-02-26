from __future__ import annotations

from app.tools.investigation_tools import investigation_instructions, investigation_tools


def _tool_names() -> set[str]:
    return {str(getattr(tool, "name", "")) for tool in investigation_tools()}


def test_investigation_tools_include_autonomous_eda_surface() -> None:
    names = _tool_names()
    assert "investigate_issue" in names
    assert "run_table_eda" in names
    assert "extract_sql_to_dataset" in names
    assert "operator_run_python" in names


def test_investigation_instructions_reference_table_eda() -> None:
    text = investigation_instructions()
    assert "run_table_eda" in text
    assert "lineage" in text.lower()
