"""Pure boto3-backed AWS read-only ops helpers.

All functions are read-only and take an injected `client_factory` so
tests can pass fakes without monkey-patching boto3. In production the
default factory lazily constructs `boto3.client(service_name)` with
ambient credentials.

Covered services:
  - Step Functions : list / describe execution, execution history
  - Lambda + Logs  : last errors for a function
  - Logs Insights  : ad-hoc log-group queries
  - ECS            : describe tasks / list stopped reasons
  - CloudWatch     : alarms in a given state
  - EventBridge    : describe rule + its targets

All list-ish helpers apply sensible caps and return plain dicts /
lists of dicts — easy to serialize into tool outputs.
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

log = logging.getLogger(__name__)


# ── Client factory ──

ClientFactory = Callable[[str], Any]


def _default_factory() -> ClientFactory:
    """Lazy default: import boto3 only when first called."""

    def _make(service: str) -> Any:
        import boto3  # lazy
        return boto3.client(service)

    return _make


_DEFAULT_FACTORY: ClientFactory | None = None
_FACTORY_LOCK = threading.Lock()


def get_default_factory() -> ClientFactory:
    global _DEFAULT_FACTORY
    with _FACTORY_LOCK:
        if _DEFAULT_FACTORY is None:
            _DEFAULT_FACTORY = _default_factory()
        return _DEFAULT_FACTORY


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            return str(value)
    return str(value)


def _since_seconds_to_epoch(seconds: int, *, now: float | None = None) -> int:
    base = now if now is not None else time.time()
    return int(base - max(0, int(seconds)))


# ── Step Functions ──


def sfn_list_executions(
    state_machine_arn: str,
    *,
    status_filter: str | None = None,
    max_results: int = 50,
    client_factory: ClientFactory | None = None,
) -> list[dict[str, Any]]:
    """List executions for a state machine. Returns plain dicts (not boto objects)."""
    client = (client_factory or get_default_factory())("stepfunctions")
    kwargs: dict[str, Any] = {"stateMachineArn": state_machine_arn, "maxResults": max_results}
    if status_filter:
        kwargs["statusFilter"] = status_filter
    resp = client.list_executions(**kwargs)
    out: list[dict[str, Any]] = []
    for e in resp.get("executions", [])[:max_results]:
        out.append({
            "executionArn": e.get("executionArn"),
            "name": e.get("name"),
            "status": e.get("status"),
            "startDate": _isoformat(e.get("startDate")),
            "stopDate": _isoformat(e.get("stopDate")),
        })
    return out


def sfn_describe_execution(
    execution_arn: str,
    *,
    client_factory: ClientFactory | None = None,
) -> dict[str, Any] | None:
    client = (client_factory or get_default_factory())("stepfunctions")
    try:
        resp = client.describe_execution(executionArn=execution_arn)
    except Exception as exc:  # noqa: BLE001
        log.warning("sfn describe_execution failed: %s", exc)
        return None
    return {
        "executionArn": resp.get("executionArn"),
        "stateMachineArn": resp.get("stateMachineArn"),
        "name": resp.get("name"),
        "status": resp.get("status"),
        "startDate": _isoformat(resp.get("startDate")),
        "stopDate": _isoformat(resp.get("stopDate")),
        "input": resp.get("input"),
        "output": resp.get("output"),
        "error": resp.get("error"),
        "cause": resp.get("cause"),
    }


def sfn_get_execution_history(
    execution_arn: str,
    *,
    max_results: int = 200,
    include_execution_data: bool = True,
    client_factory: ClientFactory | None = None,
) -> list[dict[str, Any]]:
    client = (client_factory or get_default_factory())("stepfunctions")
    out: list[dict[str, Any]] = []
    next_token: str | None = None
    while len(out) < max_results:
        kwargs: dict[str, Any] = {
            "executionArn": execution_arn,
            "maxResults": min(100, max_results - len(out)),
            "includeExecutionData": include_execution_data,
            "reverseOrder": False,
        }
        if next_token:
            kwargs["nextToken"] = next_token
        resp = client.get_execution_history(**kwargs)
        for event in resp.get("events", []):
            out.append({
                "id": event.get("id"),
                "timestamp": _isoformat(event.get("timestamp")),
                "type": event.get("type"),
                "previousEventId": event.get("previousEventId"),
                # ExecutionFailedEventDetails etc. — flatten the first *Details dict we see
                **_flatten_event_details(event),
            })
        next_token = resp.get("nextToken")
        if not next_token:
            break
    return out[:max_results]


def _flatten_event_details(event: dict[str, Any]) -> dict[str, Any]:
    for k, v in event.items():
        if k.endswith("EventDetails") and isinstance(v, dict):
            return {"details_kind": k, "details": v}
    return {}


# ── Lambda (via CloudWatch Logs) ──


@dataclass(frozen=True)
class LambdaErrorEvent:
    timestamp: Optional[str]
    message: str


def lambda_get_last_errors(
    function_name: str,
    *,
    lookback_seconds: int = 6 * 3600,
    max_events: int = 50,
    client_factory: ClientFactory | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent error-shaped log events for a Lambda."""
    client = (client_factory or get_default_factory())("logs")
    log_group = f"/aws/lambda/{function_name}"
    start = _since_seconds_to_epoch(lookback_seconds) * 1000
    try:
        resp = client.filter_log_events(
            logGroupName=log_group,
            startTime=start,
            filterPattern="?ERROR ?Exception ?TRACEBACK ?Traceback ?Task timed out",
            limit=max_events,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("lambda_get_last_errors filter_log_events failed: %s", exc)
        return []
    events: list[dict[str, Any]] = []
    for e in resp.get("events", []):
        ts_ms = e.get("timestamp")
        iso: Optional[str] = None
        if ts_ms:
            iso = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        events.append({"timestamp": iso, "message": (e.get("message") or "").strip()})
    return events


# ── CloudWatch Logs Insights ──


def logs_insights_query(
    log_group: str,
    query: str,
    *,
    since_seconds: int = 3600,
    max_results: int = 200,
    poll_interval_s: float = 1.0,
    timeout_s: float = 60.0,
    client_factory: ClientFactory | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Run a Logs Insights query, poll to completion, and return rows.

    Returns {status, rows[{field: value, ...}], statistics, query_id}.
    Times out politely if the query is still running.
    """
    client = (client_factory or get_default_factory())("logs")
    base = now if now is not None else time.time()
    end = int(base)
    start = end - max(0, int(since_seconds))
    resp = client.start_query(
        logGroupName=log_group,
        startTime=start,
        endTime=end,
        queryString=query,
        limit=max_results,
    )
    query_id = resp.get("queryId")
    if not query_id:
        return {"status": "Failed", "rows": [], "query_id": None, "error": "no queryId"}

    deadline = base + timeout_s
    last: dict[str, Any] | None = None
    while True:
        last = client.get_query_results(queryId=query_id)
        status = last.get("status")
        if status in {"Complete", "Failed", "Cancelled"}:
            break
        if (now if now is not None else time.time()) > deadline:
            return {
                "status": status or "Running",
                "rows": _rows_from_insights(last or {}),
                "query_id": query_id,
                "statistics": (last or {}).get("statistics"),
                "error": "timed_out_polling",
            }
        time.sleep(poll_interval_s)
    return {
        "status": (last or {}).get("status"),
        "rows": _rows_from_insights(last or {}),
        "statistics": (last or {}).get("statistics"),
        "query_id": query_id,
    }


def _rows_from_insights(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in payload.get("results") or []:
        rows.append({item.get("field") or "": item.get("value") or "" for item in row})
    return rows


# ── ECS ──


def ecs_describe_tasks(
    cluster: str,
    *,
    max_tasks: int = 100,
    client_factory: ClientFactory | None = None,
) -> list[dict[str, Any]]:
    client = (client_factory or get_default_factory())("ecs")
    resp = client.list_tasks(cluster=cluster, desiredStatus="RUNNING", maxResults=max_tasks)
    arns = resp.get("taskArns", [])
    if not arns:
        return []
    descr = client.describe_tasks(cluster=cluster, tasks=arns).get("tasks", [])
    out: list[dict[str, Any]] = []
    for t in descr:
        out.append({
            "taskArn": t.get("taskArn"),
            "taskDefinitionArn": t.get("taskDefinitionArn"),
            "lastStatus": t.get("lastStatus"),
            "desiredStatus": t.get("desiredStatus"),
            "healthStatus": t.get("healthStatus"),
            "startedAt": _isoformat(t.get("startedAt")),
            "stoppedAt": _isoformat(t.get("stoppedAt")),
            "stoppedReason": t.get("stoppedReason"),
        })
    return out


def ecs_list_stopped_reasons(
    cluster: str,
    service: str | None = None,
    *,
    max_tasks: int = 100,
    client_factory: ClientFactory | None = None,
) -> list[dict[str, Any]]:
    client = (client_factory or get_default_factory())("ecs")
    kwargs: dict[str, Any] = {"cluster": cluster, "desiredStatus": "STOPPED", "maxResults": max_tasks}
    if service:
        kwargs["serviceName"] = service
    arns = client.list_tasks(**kwargs).get("taskArns", [])
    if not arns:
        return []
    descr = client.describe_tasks(cluster=cluster, tasks=arns).get("tasks", [])
    out: list[dict[str, Any]] = []
    for t in descr:
        out.append({
            "taskArn": t.get("taskArn"),
            "lastStatus": t.get("lastStatus"),
            "stoppedAt": _isoformat(t.get("stoppedAt")),
            "stoppedReason": t.get("stoppedReason"),
            "containers": [
                {"name": c.get("name"), "exitCode": c.get("exitCode"), "reason": c.get("reason")}
                for c in t.get("containers", [])
            ],
        })
    return out


# ── CloudWatch alarms ──


def cloudwatch_alarms(
    state_value: str | None = None,
    *,
    name_prefix: str | None = None,
    max_results: int = 100,
    client_factory: ClientFactory | None = None,
) -> list[dict[str, Any]]:
    client = (client_factory or get_default_factory())("cloudwatch")
    kwargs: dict[str, Any] = {"MaxRecords": max_results}
    if state_value:
        kwargs["StateValue"] = state_value
    if name_prefix:
        kwargs["AlarmNamePrefix"] = name_prefix
    resp = client.describe_alarms(**kwargs)
    out: list[dict[str, Any]] = []
    for a in resp.get("MetricAlarms", [])[:max_results]:
        out.append({
            "name": a.get("AlarmName"),
            "state": a.get("StateValue"),
            "reason": a.get("StateReason"),
            "updated": _isoformat(a.get("StateUpdatedTimestamp")),
            "metric": a.get("MetricName"),
            "namespace": a.get("Namespace"),
            "threshold": a.get("Threshold"),
        })
    return out


# ── EventBridge ──


def eventbridge_describe_rule(
    name: str,
    *,
    event_bus_name: str | None = None,
    client_factory: ClientFactory | None = None,
) -> dict[str, Any] | None:
    client = (client_factory or get_default_factory())("events")
    kwargs: dict[str, Any] = {"Name": name}
    if event_bus_name:
        kwargs["EventBusName"] = event_bus_name
    try:
        rule = client.describe_rule(**kwargs)
        targets = client.list_targets_by_rule(**kwargs).get("Targets", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("eventbridge describe_rule failed: %s", exc)
        return None
    return {
        "name": rule.get("Name"),
        "arn": rule.get("Arn"),
        "state": rule.get("State"),
        "scheduleExpression": rule.get("ScheduleExpression"),
        "eventPattern": rule.get("EventPattern"),
        "description": rule.get("Description"),
        "eventBusName": rule.get("EventBusName"),
        "targets": [
            {"id": t.get("Id"), "arn": t.get("Arn"), "input": t.get("Input")}
            for t in targets
        ],
    }


__all__ = [
    "ClientFactory",
    "get_default_factory",
    "sfn_list_executions",
    "sfn_describe_execution",
    "sfn_get_execution_history",
    "lambda_get_last_errors",
    "logs_insights_query",
    "ecs_describe_tasks",
    "ecs_list_stopped_reasons",
    "cloudwatch_alarms",
    "eventbridge_describe_rule",
]
