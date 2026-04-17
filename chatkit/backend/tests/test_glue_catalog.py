"""Unit tests for app.investigation.glue_catalog.

Uses a fake boto3-style client injected via a fake session so we don't
hit AWS. Exercises get_table / get_partitions / discover_table with
realistic API shapes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.investigation.glue_catalog import GlueCatalog


class _FakeEntityNotFound(Exception):
    pass


def _page(items_key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {items_key: items}


class _FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]):
        self._pages = pages

    def paginate(self, **_: Any):
        yield from self._pages


class _FakeGlueClient:
    def __init__(self) -> None:
        self.exceptions = SimpleNamespace(EntityNotFoundException=_FakeEntityNotFound)
        self._databases = [{"Name": "analytics_db"}, {"Name": "monitoring_db"}]
        self._tables: dict[tuple[str, str], dict[str, Any]] = {
            ("analytics_db", "market_level_anomalies_v3"): {
                "DatabaseName": "analytics_db",
                "Name": "market_level_anomalies_v3",
                "Owner": "hadoop",
                "TableType": "EXTERNAL_TABLE",
                "Parameters": {"classification": "parquet"},
                "StorageDescriptor": {
                    "Location": "s3://bucket/market_level_anomalies/",
                    "Columns": [
                        {"Name": "customer", "Type": "string", "Comment": "customer code"},
                        {"Name": "origin", "Type": "string"},
                    ],
                },
                "PartitionKeys": [
                    {"Name": "sales_date", "Type": "int"},
                    {"Name": "customer", "Type": "string"},
                ],
            },
        }
        self._partitions: dict[tuple[str, str], list[dict[str, Any]]] = {
            ("analytics_db", "market_level_anomalies_v3"): [
                {"Values": ["20260416", "B6"], "StorageDescriptor": {"Location": "s3://.../B6/"}},
                {"Values": ["20260417", "B6"], "StorageDescriptor": {"Location": "s3://.../B6/"}},
            ],
        }

    def get_paginator(self, name: str):
        if name == "get_databases":
            return _FakePaginator([_page("DatabaseList", self._databases)])
        if name == "get_tables":
            outer = self

            class _TablesPaginator:
                def paginate(self, **kwargs: Any):
                    db = kwargs["DatabaseName"]
                    expression = kwargs.get("Expression")
                    matches = [
                        {"Name": t[1]}
                        for t in outer._tables.keys()
                        if t[0] == db and (not expression or expression in t[1])
                    ]
                    yield _page("TableList", matches)

            return _TablesPaginator()
        if name == "get_partitions":
            outer = self

            class _PartitionsPaginator:
                def paginate(self, **kwargs: Any):
                    key = (kwargs["DatabaseName"], kwargs["TableName"])
                    if key not in outer._partitions:
                        raise _FakeEntityNotFound(".".join(key))
                    yield _page("Partitions", list(outer._partitions[key]))

            return _PartitionsPaginator()
        raise AssertionError(f"unexpected paginator: {name}")

    def get_table(self, DatabaseName: str, Name: str):  # noqa: N803 (AWS uses PascalCase kwargs)
        key = (DatabaseName, Name)
        if key not in self._tables:
            raise _FakeEntityNotFound(f"{DatabaseName}.{Name}")
        return {"Table": self._tables[key]}


class _FakeSession:
    def __init__(self, client: _FakeGlueClient):
        self._client = client

    def client(self, name: str, region_name: str | None = None):  # noqa: ARG002
        assert name == "glue"
        return self._client


def _make_catalog() -> GlueCatalog:
    return GlueCatalog(session=_FakeSession(_FakeGlueClient()))


def test_list_databases() -> None:
    cat = _make_catalog()
    assert cat.list_databases() == ["analytics_db", "monitoring_db"]


def test_get_table_returns_columns_and_partition_keys() -> None:
    cat = _make_catalog()
    t = cat.get_table("analytics_db", "market_level_anomalies_v3")
    assert t is not None
    assert t.qualified == "analytics_db.market_level_anomalies_v3"
    assert [c.name for c in t.columns] == ["customer", "origin"]
    assert t.partition_key_names == ("sales_date", "customer")
    assert t.location == "s3://bucket/market_level_anomalies/"


def test_get_table_missing_returns_none() -> None:
    cat = _make_catalog()
    assert cat.get_table("analytics_db", "no_such_table") is None


def test_get_partitions_returns_values() -> None:
    cat = _make_catalog()
    parts = cat.get_partitions("analytics_db", "market_level_anomalies_v3")
    assert [p.values for p in parts] == [("20260416", "B6"), ("20260417", "B6")]


def test_get_partitions_missing_table_returns_empty() -> None:
    cat = _make_catalog()
    parts = cat.get_partitions("analytics_db", "no_such_table")
    assert parts == []


def test_discover_table_by_qualified_ref() -> None:
    cat = _make_catalog()
    hits = cat.discover_table("analytics_db.market_level_anomalies_v3")
    assert len(hits) == 1 and hits[0].name == "market_level_anomalies_v3"


def test_discover_table_by_bare_name_scans_dbs() -> None:
    cat = _make_catalog()
    hits = cat.discover_table("market_level_anomalies_v3")
    assert len(hits) == 1 and hits[0].database == "analytics_db"


def test_discover_table_unknown_returns_empty() -> None:
    cat = _make_catalog()
    assert cat.discover_table("mystery_table") == []
