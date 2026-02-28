from __future__ import annotations

from app.tools.investigation_tools import investigation_tools


def _tool_names() -> set[str]:
    return {str(getattr(tool, "name", "")) for tool in investigation_tools()}


def test_investigation_tools_six_atomic_tools() -> None:
    names = _tool_names()
    assert "execute_sql" in names
    assert "fetch_s3" in names
    assert "run_python" in names
    assert "inspect_table" in names
    assert "search_kb" in names
    assert "resolve_codes" in names
    assert len(names) == 6
