"""Smoke tests for the planner sub-agent factory.

We don't invoke the planner end-to-end (that requires OpenAI + real
tool context). We verify structural properties: it builds, has the
expected read-only tools, and can be wrapped as an agent-tool.
"""

from __future__ import annotations

from app.agents.planner import (
    DEFAULT_PLANNER_MAX_TURNS,
    DEFAULT_PLANNER_MODEL,
    PLANNER_INSTRUCTIONS,
    PLANNER_TOOL_NAME,
    build_planner_agent,
    planner_tool_description,
)


def test_planner_builds() -> None:
    planner = build_planner_agent()
    assert planner is not None
    assert planner.name == "DS Chat Planner"
    assert PLANNER_INSTRUCTIONS.strip() in (planner.instructions or "")


def test_planner_has_readonly_tools_only() -> None:
    planner = build_planner_agent()
    tool_names = {getattr(t, "name", None) for t in planner.tools}
    assert "read_file" in tool_names
    assert "list_dir" in tool_names
    assert "search_kb" in tool_names
    assert "inspect_table" in tool_names
    assert "resolve_codes" in tool_names
    # Explicitly excluded
    for forbidden in ("execute_sql", "fetch_s3", "bash", "edit_file", "git",
                      "publish_image", "render_image", "download_file"):
        assert forbidden not in tool_names, f"planner should not have {forbidden}"


def test_planner_tool_description_mentions_json_envelope() -> None:
    desc = planner_tool_description()
    assert "JSON envelope" in desc
    assert "plan" in desc


def test_planner_model_defaults_to_strong() -> None:
    # gpt-5.4 (not mini) — planners need reasoning power
    assert DEFAULT_PLANNER_MODEL == "gpt-5.4"
    assert DEFAULT_PLANNER_MAX_TURNS == 5
    assert PLANNER_TOOL_NAME == "plan_task"
