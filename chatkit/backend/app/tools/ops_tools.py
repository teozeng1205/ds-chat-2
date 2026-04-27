"""@function_tool wrappers around the read-only AWS ops client.

Registers agent-facing tools for Step Functions, Lambda logs,
CloudWatch Logs Insights, ECS, CloudWatch alarms, and EventBridge.
Every wrapper streams progress updates and returns a plain dict the
agent can reason about.

All tools are read-only. No mutate-the-cloud operation appears here.
"""

from __future__ import annotations

import logging
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent

from ..ops import ops_client as oc
from ._common import TIMEOUT_AWS, tool_error

log = logging.getLogger(__name__)


async def _stream(ctx: RunContextWrapper[AgentContext], icon: str, text: str) -> None:  # type: ignore[type-arg]
    try:
        await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))
    except Exception:
        pass  # progress is best-effort


def _err(exc: Exception) -> dict[str, Any]:
    log.exception("ops tool failed: %s", exc)
    return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


# ── SFN ──

@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def sfn_list_executions(
    ctx: RunContextWrapper[AgentContext],
    state_machine_arn: str,
    status_filter: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """List Step Functions executions for one known state machine ARN (read-only).

    Do not use this tool to discover or list state machines. It requires a
    non-empty state_machine_arn; use AWS CLI `aws stepfunctions list-state-machines`
    via bash when the user asks for state machine names.

    Args:
        state_machine_arn: ARN of the state machine (e.g. arn:aws:states:us-east-1:...:stateMachine/My-SFN).
        status_filter: Optional RUNNING | SUCCEEDED | FAILED | TIMED_OUT | ABORTED.
        max_results: Cap (default 50).

    Returns: {ok, executions: [{executionArn, name, status, startDate, stopDate}]}.
    """
    try:
        if not state_machine_arn.strip():
            return {
                "ok": False,
                "error": "state_machine_arn is required; use bash with aws stepfunctions list-state-machines to list state machines",
                "error_type": "MissingStateMachineArn",
            }
        await _stream(ctx, "clock", f"Listing SFN executions for {state_machine_arn.rsplit(':', 1)[-1]}.")
        executions = oc.sfn_list_executions(
            state_machine_arn, status_filter=status_filter, max_results=max_results
        )
        await _stream(ctx, "check-circle", f"Found {len(executions)} executions.")
        return {"ok": True, "executions": executions}
    except Exception as exc:
        return _err(exc)


@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def sfn_describe_execution(
    ctx: RunContextWrapper[AgentContext],
    execution_arn: str,
) -> dict[str, Any]:
    """Describe one SFN execution: status, input, output, error, cause.

    Use this after sfn_list_executions finds a FAILED run to see what broke.
    """
    try:
        await _stream(ctx, "clock", "Describing SFN execution.")
        detail = oc.sfn_describe_execution(execution_arn)
        if detail is None:
            return {"ok": False, "error": "execution not found", "error_type": "NotFound"}
        await _stream(ctx, "check-circle", f"Execution {detail.get('status')}.")
        return {"ok": True, **detail}
    except Exception as exc:
        return _err(exc)


@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def sfn_get_execution_history(
    ctx: RunContextWrapper[AgentContext],
    execution_arn: str,
    max_results: int = 200,
) -> dict[str, Any]:
    """Return a flattened SFN execution history (up to max_results events).

    Useful for identifying which task/state failed. Each event carries
    its details_kind + details flattened from the AWS event payload.
    """
    try:
        await _stream(ctx, "clock", "Fetching SFN execution history.")
        events = oc.sfn_get_execution_history(execution_arn, max_results=max_results)
        failed = [e for e in events if "Failed" in (e.get("type") or "")]
        await _stream(ctx, "check-circle", f"{len(events)} events, {len(failed)} failure events.")
        return {"ok": True, "events": events, "failure_count": len(failed)}
    except Exception as exc:
        return _err(exc)


# ── Lambda via Logs ──

@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def lambda_get_last_errors(
    ctx: RunContextWrapper[AgentContext],
    function_name: str,
    lookback_hours: int = 6,
    max_events: int = 50,
) -> dict[str, Any]:
    """Fetch recent error-shaped log events for a Lambda function.

    Looks back `lookback_hours` hours in /aws/lambda/{function_name}.
    Returns the matching events (ERROR / Exception / Traceback /
    "Task timed out") so the agent can quote specific messages.
    """
    try:
        await _stream(ctx, "clock", f"Scanning last {lookback_hours}h of errors for Lambda {function_name}.")
        lookback = max(60, lookback_hours * 3600)
        events = oc.lambda_get_last_errors(function_name, lookback_seconds=lookback, max_events=max_events)
        await _stream(ctx, "check-circle", f"Found {len(events)} error events.")
        return {"ok": True, "function": function_name, "events": events, "lookback_hours": lookback_hours}
    except Exception as exc:
        return _err(exc)


# ── Logs Insights ──

@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def logs_insights_query(
    ctx: RunContextWrapper[AgentContext],
    log_group: str,
    query: str,
    since_seconds: int = 3600,
    max_results: int = 200,
    timeout_s: int = 60,
) -> dict[str, Any]:
    """Run a CloudWatch Logs Insights query against a single log group.

    Example query: `fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc`.
    Polls to completion (up to timeout_s). Returns rows, status, and
    AWS-reported statistics.
    """
    try:
        await _stream(ctx, "clock", f"Running Logs Insights on {log_group}.")
        result = oc.logs_insights_query(
            log_group, query,
            since_seconds=since_seconds, max_results=max_results, timeout_s=timeout_s,
        )
        status = result.get("status")
        rows = result.get("rows") or []
        await _stream(ctx, "check-circle", f"Query {status}: {len(rows)} rows.")
        return {"ok": True, **result}
    except Exception as exc:
        return _err(exc)


# ── ECS ──

@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def ecs_describe_tasks(
    ctx: RunContextWrapper[AgentContext],
    cluster: str,
    max_tasks: int = 100,
) -> dict[str, Any]:
    """List RUNNING ECS tasks in a cluster with their status + health.

    Useful as a first step before ecs_list_stopped_reasons.
    """
    try:
        await _stream(ctx, "clock", f"Listing ECS tasks on {cluster}.")
        tasks = oc.ecs_describe_tasks(cluster, max_tasks=max_tasks)
        await _stream(ctx, "check-circle", f"{len(tasks)} running tasks.")
        return {"ok": True, "cluster": cluster, "tasks": tasks}
    except Exception as exc:
        return _err(exc)


@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def ecs_list_stopped_reasons(
    ctx: RunContextWrapper[AgentContext],
    cluster: str,
    service: str | None = None,
    max_tasks: int = 100,
) -> dict[str, Any]:
    """List recently STOPPED ECS tasks with their stoppedReason + container exit codes.

    Use to diagnose crashes: exit code 137 = OOMKilled, 139 = segfault, etc.
    """
    try:
        await _stream(ctx, "clock", f"Scanning stopped ECS tasks on {cluster}{'/' + service if service else ''}.")
        stopped = oc.ecs_list_stopped_reasons(cluster, service=service, max_tasks=max_tasks)
        await _stream(ctx, "check-circle", f"{len(stopped)} stopped tasks.")
        return {"ok": True, "cluster": cluster, "service": service, "stopped": stopped}
    except Exception as exc:
        return _err(exc)


# ── CloudWatch alarms ──

@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def cloudwatch_alarms(
    ctx: RunContextWrapper[AgentContext],
    state_value: str | None = None,
    name_prefix: str | None = None,
    max_results: int = 100,
) -> dict[str, Any]:
    """List CloudWatch alarms, optionally filtered by state and name prefix.

    state_value: ALARM | OK | INSUFFICIENT_DATA. Pass 'ALARM' to list
    only currently-firing alarms.
    """
    try:
        state_label = state_value or "ANY"
        await _stream(ctx, "clock", f"Listing CloudWatch alarms (state={state_label}).")
        alarms = oc.cloudwatch_alarms(
            state_value=state_value, name_prefix=name_prefix, max_results=max_results,
        )
        await _stream(ctx, "check-circle", f"{len(alarms)} alarms.")
        return {"ok": True, "alarms": alarms, "state_value": state_value}
    except Exception as exc:
        return _err(exc)


# ── EventBridge ──

@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def eventbridge_describe_rule(
    ctx: RunContextWrapper[AgentContext],
    name: str,
    event_bus_name: str | None = None,
) -> dict[str, Any]:
    """Describe a single EventBridge rule and its targets.

    Shows scheduleExpression / eventPattern + the list of target
    ARNs so the agent can trace what a schedule fires.
    """
    try:
        await _stream(ctx, "clock", f"Describing EventBridge rule {name}.")
        rule = oc.eventbridge_describe_rule(name, event_bus_name=event_bus_name)
        if rule is None:
            return {"ok": False, "error": "rule not found", "error_type": "NotFound"}
        await _stream(ctx, "check-circle", f"Rule {rule.get('state')}, {len(rule.get('targets') or [])} target(s).")
        return {"ok": True, **rule}
    except Exception as exc:
        return _err(exc)


# ── Factory ──

def ops_tools() -> list[Any]:
    """Return all read-only AWS ops tools for the coding agent."""
    return [
        sfn_list_executions,
        sfn_describe_execution,
        sfn_get_execution_history,
        lambda_get_last_errors,
        logs_insights_query,
        ecs_describe_tasks,
        ecs_list_stopped_reasons,
        cloudwatch_alarms,
        eventbridge_describe_rule,
    ]


__all__ = [
    "sfn_list_executions",
    "sfn_describe_execution",
    "sfn_get_execution_history",
    "lambda_get_last_errors",
    "logs_insights_query",
    "ecs_describe_tasks",
    "ecs_list_stopped_reasons",
    "cloudwatch_alarms",
    "eventbridge_describe_rule",
    "ops_tools",
]
