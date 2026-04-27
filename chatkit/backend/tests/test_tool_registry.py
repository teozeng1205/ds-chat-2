from app.agent_harness import build_default_tool_registry


def _tool_names(model: str = "gpt-5.4-mini") -> set[str]:
    registry = build_default_tool_registry(
        model=model,
        include_apply_patch=False,
    )
    return {getattr(tool, "name", getattr(tool, "tool_name", None)) for tool in registry.build_tools()}


def test_default_registry_excludes_orchestration_tools() -> None:
    names = _tool_names()
    assert "bash" in names
    assert "execute_sql" in names
    assert "search_kb" in names
    assert "trace_pipeline" in names
    assert "plan_task" not in names
    assert "review_answer" not in names
    assert "sfn_list_executions" in names


def test_registry_can_enable_orchestration_for_experiments() -> None:
    registry = build_default_tool_registry(
        model="gpt-5.4-mini",
        include_apply_patch=False,
        include_orchestration=True,
    )
    names = {getattr(tool, "name", getattr(tool, "tool_name", None)) for tool in registry.build_tools()}
    assert "plan_task" in names
    assert "review_answer" in names


def test_registry_can_disable_aws_ops() -> None:
    registry = build_default_tool_registry(
        model="gpt-5.4-mini",
        include_apply_patch=False,
        include_aws_ops=False,
    )
    names = {getattr(tool, "name", getattr(tool, "tool_name", None)) for tool in registry.build_tools()}
    assert "bash" in names
    assert "execute_sql" in names
    assert "plan_task" not in names
    assert "review_answer" not in names
    assert "sfn_list_executions" not in names
