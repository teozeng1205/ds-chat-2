"""Unit tests for app.ops.streams_client."""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

from app.ops import streams_client as sc


class _FakeKinesisClient:
    def __init__(self) -> None:
        self._shards = [{"ShardId": "shardId-000000000000"}, {"ShardId": "shardId-000000000001"}]
        self._records_per_shard = {
            "shardId-000000000000": [
                {"PartitionKey": "QL2", "SequenceNumber": "49634",
                 "ApproximateArrivalTimestamp": _dt.datetime(2026, 4, 17, 10, 0, 0),
                 "Data": json.dumps({"provider": "QL2", "status": "ok"}).encode("utf-8")},
                {"PartitionKey": "QL2", "SequenceNumber": "49635",
                 "ApproximateArrivalTimestamp": _dt.datetime(2026, 4, 17, 10, 0, 1),
                 "Data": b"plain-text-body"},
            ],
            "shardId-000000000001": [
                {"PartitionKey": "AA", "SequenceNumber": "12001",
                 "ApproximateArrivalTimestamp": _dt.datetime(2026, 4, 17, 10, 0, 5),
                 "Data": b"\x89PNG\r\n"},  # non-utf8 binary → base64 fallback
            ],
        }
        self._get_shard_iterator_calls: list[dict[str, Any]] = []

    def list_shards(self, **_: Any) -> dict[str, Any]:
        return {"Shards": list(self._shards)}

    def get_shard_iterator(self, **kwargs: Any) -> dict[str, Any]:
        self._get_shard_iterator_calls.append(kwargs)
        return {"ShardIterator": f"iter-{kwargs['ShardId']}"}

    def get_records(self, ShardIterator: str, **_: Any) -> dict[str, Any]:  # noqa: N803
        shard_id = ShardIterator[len("iter-"):]
        return {"Records": list(self._records_per_shard.get(shard_id, []))}


def _factory(mapping: dict[str, Any]):
    def _make(service: str) -> Any:
        return mapping[service]
    return _make


def test_kinesis_tail_returns_records_from_all_shards() -> None:
    client = _FakeKinesisClient()
    result = sc.kinesis_tail(
        "ingest-priceeye-raw-search",
        minutes=5,
        sample_per_shard=5,
        client_factory=_factory({"kinesis": client}),
    )
    assert result["stream"] == "ingest-priceeye-raw-search"
    assert result["shards_sampled"] == 2
    # All shards' records are returned (sample caps not exceeded here)
    assert result["records_returned"] == 3
    partitions = {r["partitionKey"] for r in result["records"]}
    assert partitions == {"QL2", "AA"}


def test_kinesis_tail_decodes_json_text_and_base64() -> None:
    client = _FakeKinesisClient()
    result = sc.kinesis_tail(
        "ingest-priceeye-raw-search",
        minutes=5,
        sample_per_shard=5,
        client_factory=_factory({"kinesis": client}),
    )
    # one JSON dict, one plain string, one base64 string
    kinds = [type(r["data"]).__name__ for r in result["records"]]
    assert "dict" in kinds  # JSON decode
    assert "str" in kinds   # text OR base64


def test_kinesis_tail_caps_to_sample_per_shard() -> None:
    client = _FakeKinesisClient()
    # sample_per_shard=1 should keep only one record per shard
    result = sc.kinesis_tail(
        "ingest-priceeye-raw-search",
        minutes=5,
        sample_per_shard=1,
        client_factory=_factory({"kinesis": client}),
    )
    per_shard: dict[str, int] = {}
    for r in result["records"]:
        per_shard[r["shardId"]] = per_shard.get(r["shardId"], 0) + 1
    assert all(n <= 1 for n in per_shard.values())


def test_kinesis_tail_requests_iterator_at_timestamp() -> None:
    client = _FakeKinesisClient()
    sc.kinesis_tail(
        "ingest-priceeye-raw-search",
        minutes=10,
        sample_per_shard=5,
        client_factory=_factory({"kinesis": client}),
        now=1745000000.0,
    )
    # All iterator calls use AT_TIMESTAMP with the computed since-time
    assert all(c["ShardIteratorType"] == "AT_TIMESTAMP" for c in client._get_shard_iterator_calls)
    expected_since = _dt.datetime.fromtimestamp(1745000000.0 - 600, tz=_dt.timezone.utc)
    assert all(c["Timestamp"] == expected_since for c in client._get_shard_iterator_calls)


# ── Tool wrapper factory ──


def test_streams_tools_factory_registers_one_tool() -> None:
    from app.tools.streams_tools import streams_tools
    tools = streams_tools()
    assert len(tools) == 1
    assert tools[0].name == "kinesis_tail"
