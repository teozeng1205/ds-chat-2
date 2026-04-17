"""Smoke tests for ops_tools factory.

We verify the @function_tool wrappers are correctly registered and
exposed via the ops_tools() factory. Semantic behavior is covered by
test_ops_client.py against the underlying pure functions.
"""

from __future__ import annotations

from app.tools.ops_tools import ops_tools


def test_ops_tools_factory_returns_nine_tools() -> None:
    tools = ops_tools()
    names = [getattr(t, "name", None) for t in tools]
    assert len(tools) == 9
    expected = {
        "sfn_list_executions",
        "sfn_describe_execution",
        "sfn_get_execution_history",
        "lambda_get_last_errors",
        "logs_insights_query",
        "ecs_describe_tasks",
        "ecs_list_stopped_reasons",
        "cloudwatch_alarms",
        "eventbridge_describe_rule",
    }
    assert set(names) == expected


def test_tools_are_function_tool_instances() -> None:
    tools = ops_tools()
    for t in tools:
        # FunctionTool objects expose a `.name` attr and a coroutine `.on_invoke_tool` /
        # something similar. We just check they are not raw functions.
        assert hasattr(t, "name")
        assert not callable(getattr(t, "__code__", None) and getattr(t, "__call__", None))


def test_tool_descriptions_are_non_empty() -> None:
    for t in ops_tools():
        desc = getattr(t, "description", None) or ""
        assert len(desc) > 20, f"tool {t.name} has a suspiciously short description"
