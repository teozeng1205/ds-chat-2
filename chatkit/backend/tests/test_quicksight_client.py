"""Unit tests for app.ops.quicksight_client."""

from __future__ import annotations

import datetime as _dt
from typing import Any

from app.ops import quicksight_client as qs


class _FakeSTS:
    def get_caller_identity(self) -> dict[str, Any]:
        return {"Account": "590183652635"}


class _FakeQuickSight:
    def __init__(self) -> None:
        self._dashboards = [
            {"DashboardId": "d-B6-anomalies", "Name": "B6_anomalies_v5",
             "Arn": "arn:...", "CreatedTime": _dt.datetime(2026, 1, 1),
             "LastUpdatedTime": _dt.datetime(2026, 4, 1), "PublishedVersionNumber": 3},
            {"DashboardId": "d-YY-monitoring", "Name": "YY_monitoring_v4",
             "Arn": "arn:...", "CreatedTime": _dt.datetime(2026, 1, 1),
             "LastUpdatedTime": _dt.datetime(2026, 4, 1), "PublishedVersionNumber": 2},
        ]
        self._embed_calls: list[dict[str, Any]] = []

    def get_paginator(self, name: str):
        assert name == "list_dashboards"
        outer = self

        class _P:
            def paginate(self, **_: Any):
                yield {"DashboardSummaryList": outer._dashboards}

        return _P()

    def generate_embed_url_for_anonymous_user(self, **kwargs: Any) -> dict[str, Any]:
        self._embed_calls.append(kwargs)
        return {"EmbedUrl": "https://quicksight.aws.example/embed?token=abc",
                "RequestId": "req-1",
                "AnonymousUserArn": "arn:anon:1"}


def _factory(mapping: dict[str, Any]):
    def _make(service: str) -> Any:
        return mapping[service]
    return _make


def test_list_dashboards_returns_all() -> None:
    qs_client = _FakeQuickSight()
    out = qs.list_dashboards(client_factory=_factory({"quicksight": qs_client, "sts": _FakeSTS()}))
    assert {d["name"] for d in out} == {"B6_anomalies_v5", "YY_monitoring_v4"}


def test_list_dashboards_filters_by_substring() -> None:
    qs_client = _FakeQuickSight()
    out = qs.list_dashboards(
        name_substring="b6",
        client_factory=_factory({"quicksight": qs_client, "sts": _FakeSTS()}),
    )
    assert [d["name"] for d in out] == ["B6_anomalies_v5"]


def test_generate_embed_url_clamps_session_lifetime() -> None:
    qs_client = _FakeQuickSight()
    out = qs.generate_anonymous_embed_url(
        "d-B6-anomalies",
        session_lifetime_minutes=9999,
        allowed_domain="https://chat.atpco.internal",
        client_factory=_factory({"quicksight": qs_client, "sts": _FakeSTS()}),
    )
    assert out["ok"] is True
    assert out["embed_url"].startswith("https://quicksight")
    # clamped to 600
    assert qs_client._embed_calls[0]["SessionLifetimeInMinutes"] == 600
    assert qs_client._embed_calls[0]["AllowedDomains"] == ["https://chat.atpco.internal"]


def test_generate_embed_url_handles_failure_gracefully() -> None:
    class _FailingQS:
        def generate_embed_url_for_anonymous_user(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("no quicksight enterprise")

    out = qs.generate_anonymous_embed_url(
        "d-x",
        client_factory=_factory({"quicksight": _FailingQS(), "sts": _FakeSTS()}),
    )
    assert out["ok"] is False
    assert out["error_type"] == "RuntimeError"


def test_catalog_tools_factory_registers_four() -> None:
    from app.tools.catalog_tools import catalog_tools
    tools = catalog_tools()
    names = {t.name for t in tools}
    assert names == {
        "glue_get_table", "glue_get_partitions",
        "quicksight_list_dashboards", "quicksight_get_embed_url",
    }
