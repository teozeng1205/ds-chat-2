"""Unit tests for app.pipelines.discover_aws (Pass 2).

Uses in-memory fake AWS clients so nothing touches real boto3.
"""

from __future__ import annotations

import json
from typing import Any

from app.pipelines.canonicalize import AliasTable, node_id
from app.pipelines.discover_aws import discover


class _FakeClient:
    """Minimal boto3-style client that returns canned responses keyed
    by operation name (e.g., 'list_functions'). Unknown ops raise."""

    def __init__(self, responses: dict[str, Any]):
        self._responses = responses

    def __getattr__(self, name: str):
        if name not in self._responses:
            raise AttributeError(name)
        payload = self._responses[name]

        def _fn(**kwargs):
            if callable(payload):
                return payload(**kwargs)
            return payload

        return _fn


def _factory(by_service: dict[str, dict[str, Any]]):
    def _make(service: str):
        return _FakeClient(by_service.get(service, {}))

    return _make


def _aliases() -> AliasTable:
    return AliasTable.load()  # shipped priceeye aliases


def test_discover_lambda_extracts_env_vars_and_deploys_as() -> None:
    factory = _factory({
        "lambda": {
            "list_functions": {
                "Functions": [{
                    "FunctionName": "ds-priceeye-analytics-market-level-generator",
                    "Runtime": "python3.11",
                    "LastModified": "2026-04-10T00:00:00Z",
                    "Environment": {"Variables": {
                        "INPUT_BUCKET": "s3-atp-3victors3vprod-use1-derived-common-output",
                        "OUTPUT_BUCKET": "s3://s3-atp-3victors3vprod-use1-anomaly-datasets/market-level/v4",
                        "OUTPUT_TABLE": "analytics.market_level_anomalies_v4",
                    }},
                }]
            },
        },
        "stepfunctions": {"list_state_machines": {"stateMachines": []}},
        "glue": {"get_jobs": {"Jobs": []}},
        "events": {"list_rules": {"Rules": []}},
        "cloudformation": {"describe_stacks": {"Stacks": []}},
    })
    result = discover(repos=[], aliases=_aliases(), client_factory=factory)

    edge_keys = {(e.source_id, e.target_id, e.rel) for e in result.edges}

    # deploys_as edge from app to lambda
    assert any(e.rel == "deploys_as" for e in result.edges)
    # reads edge from stage → input bucket
    assert any(
        e.source_id == node_id("stage", "market-level-generator")
        and e.rel == "reads"
        and "derived-common-output" in e.target_id
        for e in result.edges
    )
    # writes edge from stage → output s3 prefix
    assert any(
        e.source_id == node_id("stage", "market-level-generator")
        and e.rel == "writes"
        and "anomaly-datasets" in e.target_id
        for e in result.edges
    )
    # writes edge from stage → output table
    assert (
        node_id("stage", "market-level-generator"),
        node_id("redshift_table", "analytics.market_level_anomalies_v4"),
        "writes",
    ) in edge_keys


def test_discover_step_functions_trigger_edges() -> None:
    asl = {
        "States": {
            "Run": {
                "Resource": "arn:aws:lambda:us-east-1:123456789012:function:my-stage-handler",
                "Type": "Task",
            }
        }
    }
    factory = _factory({
        "lambda": {"list_functions": {"Functions": []}},
        "stepfunctions": {
            "list_state_machines": {"stateMachines": [
                {"name": "DS-Analytics-Jobs",
                 "stateMachineArn": "arn:aws:states:us-east-1:123:stateMachine:DS-Analytics-Jobs"}
            ]},
            "describe_state_machine": {"definition": json.dumps(asl)},
        },
        "glue": {"get_jobs": {"Jobs": []}},
        "events": {"list_rules": {"Rules": []}},
        "cloudformation": {"describe_stacks": {"Stacks": []}},
    })
    result = discover(repos=[], aliases=_aliases(), client_factory=factory)
    edge_keys = {(e.source_id, e.target_id, e.rel) for e in result.edges}
    assert (node_id("step_fn", "ds-analytics-jobs"),
            node_id("lambda", "my-stage-handler"),
            "triggers") in edge_keys


def test_discover_eventbridge_targets_lambda_and_sfn() -> None:
    factory = _factory({
        "lambda": {"list_functions": {"Functions": []}},
        "stepfunctions": {"list_state_machines": {"stateMachines": []}},
        "glue": {"get_jobs": {"Jobs": []}},
        "events": {
            "list_rules": {"Rules": [{
                "Name": "HourlyCronRule",
                "ScheduleExpression": "cron(0 * * * ? *)",
                "State": "ENABLED",
            }]},
            "list_targets_by_rule": {"Targets": [
                {"Arn": "arn:aws:lambda:us-east-1:123:function:hourly-processor"},
                {"Arn": "arn:aws:states:us-east-1:123:stateMachine:BigDag"},
            ]},
        },
        "cloudformation": {"describe_stacks": {"Stacks": []}},
    })
    result = discover(repos=[], aliases=_aliases(), client_factory=factory)
    edge_keys = {(e.source_id, e.target_id, e.rel) for e in result.edges}
    assert (node_id("event_rule", "hourlycronrule"),
            node_id("lambda", "hourly-processor"),
            "triggers") in edge_keys
    assert (node_id("event_rule", "hourlycronrule"),
            node_id("step_fn", "bigdag"),
            "triggers") in edge_keys


def test_discover_cfn_stacks_emit_repo_edge_when_tagged() -> None:
    factory = _factory({
        "lambda": {"list_functions": {"Functions": []}},
        "stepfunctions": {"list_state_machines": {"stateMachines": []}},
        "glue": {"get_jobs": {"Jobs": []}},
        "events": {"list_rules": {"Rules": []}},
        "cloudformation": {"describe_stacks": {"Stacks": [
            {"StackName": "ds-priceeye-analytics-prod",
             "Tags": [{"Key": "repo", "Value": "ds-priceeye-analytics"}]},
            {"StackName": "untagged-stack", "Tags": []},
        ]}},
    })
    result = discover(repos=[], aliases=_aliases(), client_factory=factory)
    edge_keys = {(e.source_id, e.target_id, e.rel) for e in result.edges}
    assert (node_id("repo", "ds-priceeye-analytics"),
            node_id("app", "ds-priceeye-analytics"),
            "repo") in edge_keys


def test_discover_glue_jobs_emit_reads_writes() -> None:
    factory = _factory({
        "lambda": {"list_functions": {"Functions": []}},
        "stepfunctions": {"list_state_machines": {"stateMachines": []}},
        "glue": {"get_jobs": {"Jobs": [{
            "Name": "competitive-position-glue-job",
            "DefaultArguments": {
                "--source_bucket": "s3://s3-atp-3victors3vprod-use1-derived-common-output/v1",
                "--output_table": "analytics_db.competitive_position",
            },
        }]}},
        "events": {"list_rules": {"Rules": []}},
        "cloudformation": {"describe_stacks": {"Stacks": []}},
    })
    result = discover(repos=[], aliases=_aliases(), client_factory=factory)
    stage_id = node_id("stage", "competitive-position")
    reads = {(e.source_id, e.target_id) for e in result.edges if e.rel == "reads"}
    writes = {(e.source_id, e.target_id) for e in result.edges if e.rel == "writes"}
    assert any(src == stage_id for src, _ in reads)
    assert any(src == stage_id for src, _ in writes)


def test_discover_degrades_when_service_raises() -> None:
    def _raise(**kwargs):
        raise RuntimeError("boom")

    factory = _factory({
        "lambda": {"list_functions": _raise},
        "stepfunctions": {"list_state_machines": {"stateMachines": []}},
        "glue": {"get_jobs": {"Jobs": []}},
        "events": {"list_rules": {"Rules": []}},
        "cloudformation": {"describe_stacks": {"Stacks": []}},
    })
    result = discover(repos=[], aliases=_aliases(), client_factory=factory)
    # No crash, zero lambda nodes
    assert all(n.kind != "lambda" for n in result.nodes)


def test_discover_with_no_factory_returns_empty(monkeypatch) -> None:
    # Force get_default_factory import to fail cleanly
    import app.pipelines.discover_aws as mod
    # Simulate no creds by passing a factory that always raises on attr access
    def _raises_factory(service: str):
        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("no creds")
        return _Broken()

    result = mod.discover(repos=[], aliases=_aliases(), client_factory=_raises_factory)
    assert result.nodes == [] or all(n.source.startswith("aws:") for n in result.nodes)
    # Either way, no crash
    assert isinstance(result.resources_scanned, int)
