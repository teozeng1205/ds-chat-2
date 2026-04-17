"""Unit tests for app.ops.ops_client using injected fake boto3 clients."""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from app.ops import ops_client as oc


# ── Fakes ──

class _FakeSFNClient:
    def __init__(self) -> None:
        self._list_calls: list[dict[str, Any]] = []

    def list_executions(self, **kwargs: Any) -> dict[str, Any]:
        self._list_calls.append(kwargs)
        return {"executions": [
            {"executionArn": "arn:aws:states:::execution/sm/e1", "name": "e1", "status": "SUCCEEDED",
             "startDate": _dt.datetime(2026, 4, 17, 10, 0, 0),
             "stopDate": _dt.datetime(2026, 4, 17, 10, 5, 0)},
            {"executionArn": "arn:aws:states:::execution/sm/e2", "name": "e2", "status": "FAILED",
             "startDate": _dt.datetime(2026, 4, 17, 11, 0, 0),
             "stopDate": _dt.datetime(2026, 4, 17, 11, 1, 0)},
        ]}

    def describe_execution(self, **_: Any) -> dict[str, Any]:
        return {"executionArn": "arn:aws:states:::execution/sm/e2", "stateMachineArn": "sm",
                "name": "e2", "status": "FAILED",
                "startDate": _dt.datetime(2026, 4, 17, 11, 0, 0),
                "stopDate": _dt.datetime(2026, 4, 17, 11, 1, 0),
                "input": "{}", "output": None,
                "error": "States.TaskFailed", "cause": "container exit 1"}

    def get_execution_history(self, **_: Any) -> dict[str, Any]:
        return {"events": [
            {"id": 1, "timestamp": _dt.datetime(2026, 4, 17, 11, 0, 0), "type": "ExecutionStarted",
             "previousEventId": 0, "executionStartedEventDetails": {"input": "{}"}},
            {"id": 2, "timestamp": _dt.datetime(2026, 4, 17, 11, 0, 10), "type": "TaskFailed",
             "previousEventId": 1, "taskFailedEventDetails": {"error": "oops"}},
        ]}


class _FakeLogsClient:
    def __init__(self) -> None:
        self.start_query_calls: list[dict[str, Any]] = []
        self._query_status: list[dict[str, Any]] = [
            {"status": "Running", "results": []},
            {"status": "Complete", "results": [
                [{"field": "@timestamp", "value": "2026-04-17T10:00:00Z"},
                 {"field": "@message", "value": "hello"}],
                [{"field": "@timestamp", "value": "2026-04-17T10:01:00Z"},
                 {"field": "@message", "value": "world"}],
            ], "statistics": {"recordsMatched": 2}},
        ]
        self._filter_events = [
            {"timestamp": 1744893600000, "message": "ERROR: something went wrong\n"},
            {"timestamp": 1744893601000, "message": "Task timed out after 30s\n"},
        ]

    def start_query(self, **kwargs: Any) -> dict[str, Any]:
        self.start_query_calls.append(kwargs)
        return {"queryId": "q-1"}

    def get_query_results(self, **_: Any) -> dict[str, Any]:
        # Pop from the head to simulate progression
        return self._query_status.pop(0) if self._query_status else {"status": "Complete", "results": []}

    def filter_log_events(self, **_: Any) -> dict[str, Any]:
        return {"events": list(self._filter_events)}


class _FakeECSClient:
    def list_tasks(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("desiredStatus") == "RUNNING":
            return {"taskArns": ["t1"]}
        return {"taskArns": ["t2"]}

    def describe_tasks(self, **kwargs: Any) -> dict[str, Any]:
        arns = kwargs.get("tasks", [])
        if arns == ["t1"]:
            return {"tasks": [{"taskArn": "t1", "lastStatus": "RUNNING",
                                "desiredStatus": "RUNNING", "healthStatus": "HEALTHY",
                                "startedAt": _dt.datetime(2026, 4, 17, 9, 0, 0),
                                "stoppedAt": None,
                                "containers": []}]}
        return {"tasks": [{"taskArn": "t2", "lastStatus": "STOPPED",
                           "stoppedAt": _dt.datetime(2026, 4, 17, 9, 5, 0),
                           "stoppedReason": "exit code 137 (OOM)",
                           "containers": [{"name": "app", "exitCode": 137, "reason": "OOMKilled"}]}]}


class _FakeCloudWatchClient:
    def describe_alarms(self, **_: Any) -> dict[str, Any]:
        return {"MetricAlarms": [
            {"AlarmName": "A1", "StateValue": "ALARM", "StateReason": "breach",
             "StateUpdatedTimestamp": _dt.datetime(2026, 4, 17, 9, 0, 0),
             "MetricName": "Errors", "Namespace": "AWS/Lambda", "Threshold": 5},
            {"AlarmName": "A2", "StateValue": "OK"},
        ]}


class _FakeEventsClient:
    def describe_rule(self, **_: Any) -> dict[str, Any]:
        return {"Name": "kb-refresh-nightly", "Arn": "arn:...", "State": "ENABLED",
                "ScheduleExpression": "cron(0 8 * * ? *)", "Description": "Nightly KB rebuild"}

    def list_targets_by_rule(self, **_: Any) -> dict[str, Any]:
        return {"Targets": [{"Id": "1", "Arn": "arn:ecs:...", "Input": "{}"}]}


def _factory(mapping: dict[str, Any]):
    def _make(service: str) -> Any:
        client = mapping.get(service)
        if client is None:
            raise AssertionError(f"no fake for {service}")
        return client
    return _make


# ── SFN ──


def test_sfn_list_executions() -> None:
    sfn = _FakeSFNClient()
    out = oc.sfn_list_executions("arn:sm", max_results=2, client_factory=_factory({"stepfunctions": sfn}))
    assert len(out) == 2
    assert out[0]["status"] == "SUCCEEDED"
    assert out[1]["status"] == "FAILED"
    assert out[0]["startDate"].startswith("2026-04-17T")


def test_sfn_describe_execution() -> None:
    sfn = _FakeSFNClient()
    out = oc.sfn_describe_execution("arn:e2", client_factory=_factory({"stepfunctions": sfn}))
    assert out is not None and out["status"] == "FAILED"
    assert out["error"] == "States.TaskFailed"


def test_sfn_get_execution_history_flattens_event_details() -> None:
    sfn = _FakeSFNClient()
    events = oc.sfn_get_execution_history("arn:e2", max_results=5,
                                          client_factory=_factory({"stepfunctions": sfn}))
    assert len(events) == 2
    assert events[1]["details_kind"] == "taskFailedEventDetails"
    assert events[1]["details"] == {"error": "oops"}


# ── Lambda via Logs ──


def test_lambda_get_last_errors_returns_iso_timestamps() -> None:
    logs = _FakeLogsClient()
    out = oc.lambda_get_last_errors("my-fn", lookback_seconds=3600, max_events=10,
                                     client_factory=_factory({"logs": logs}))
    assert len(out) == 2
    assert out[0]["timestamp"] is not None
    assert "ERROR" in out[0]["message"] or "timed out" in out[0]["message"]


# ── Logs Insights ──


def test_logs_insights_query_polls_to_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    logs = _FakeLogsClient()
    # Skip real sleeping
    monkeypatch.setattr(oc, "time", type("T", (), {"time": lambda *_: 1000.0, "sleep": lambda *_: None}))
    out = oc.logs_insights_query("/aws/lambda/fn", "fields @message", since_seconds=60,
                                  max_results=100, poll_interval_s=0.0,
                                  client_factory=_factory({"logs": logs}))
    assert out["status"] == "Complete"
    assert len(out["rows"]) == 2
    assert out["rows"][0]["@message"] == "hello"


# ── ECS ──


def test_ecs_describe_running_tasks() -> None:
    ecs = _FakeECSClient()
    out = oc.ecs_describe_tasks("my-cluster", client_factory=_factory({"ecs": ecs}))
    assert out == [{
        "taskArn": "t1", "taskDefinitionArn": None, "lastStatus": "RUNNING",
        "desiredStatus": "RUNNING", "healthStatus": "HEALTHY",
        "startedAt": "2026-04-17T09:00:00", "stoppedAt": None, "stoppedReason": None,
    }]


def test_ecs_list_stopped_reasons_includes_container_exit() -> None:
    ecs = _FakeECSClient()
    out = oc.ecs_list_stopped_reasons("my-cluster", client_factory=_factory({"ecs": ecs}))
    assert len(out) == 1
    assert out[0]["stoppedReason"].startswith("exit code 137")
    assert out[0]["containers"][0]["exitCode"] == 137


# ── CloudWatch alarms ──


def test_cloudwatch_alarms_in_alarm_state() -> None:
    cw = _FakeCloudWatchClient()
    out = oc.cloudwatch_alarms(state_value="ALARM", client_factory=_factory({"cloudwatch": cw}))
    # both returned; our fake doesn't filter (that's boto's job in reality), just shape is preserved
    assert any(a["state"] == "ALARM" for a in out)
    a1 = next(a for a in out if a["name"] == "A1")
    assert a1["reason"] == "breach"
    assert a1["metric"] == "Errors"


# ── EventBridge ──


def test_eventbridge_describe_rule_joins_targets() -> None:
    ev = _FakeEventsClient()
    out = oc.eventbridge_describe_rule("kb-refresh-nightly", client_factory=_factory({"events": ev}))
    assert out is not None
    assert out["scheduleExpression"] == "cron(0 8 * * ? *)"
    assert out["targets"] == [{"id": "1", "arn": "arn:ecs:...", "input": "{}"}]
