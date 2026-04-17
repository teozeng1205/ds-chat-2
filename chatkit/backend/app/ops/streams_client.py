"""Read-only Kinesis stream helpers.

Primary entry point is `kinesis_tail(stream, minutes, sample_per_shard)`,
which samples recent records from every shard of a stream. Useful for
"is QL2 sending data right now?" questions — the 10 ingest-*-raw-search
streams have 24h retention, so tailing is cheap.

All functions take a client_factory so tests can inject fakes.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import logging
import time
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


ClientFactory = Callable[[str], Any]


def _default_factory() -> ClientFactory:
    def _make(service: str) -> Any:
        import boto3
        return boto3.client(service)

    return _make


def _iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    iso = getattr(ts, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            return str(ts)
    return str(ts)


def _decode_data(data: bytes) -> Any:
    """Best-effort decode: JSON → dict; else utf-8 string; else base64."""
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception:
        pass
    try:
        return data.decode("utf-8")
    except Exception:
        return base64.b64encode(data).decode("ascii")


def kinesis_tail(
    stream_name: str,
    *,
    minutes: int = 5,
    sample_per_shard: int = 5,
    max_records_per_call: int = 200,
    client_factory: ClientFactory | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Sample recent records across all shards of a Kinesis stream.

    Returns {stream, shards_sampled, records_returned, records: [...]}.
    Each record carries partitionKey, sequenceNumber, timestamp,
    shardId, and decoded `data` (JSON / text / base64 fallback).
    """
    client = (client_factory or _default_factory())("kinesis")
    base = now if now is not None else time.time()
    since_dt = _dt.datetime.fromtimestamp(base - max(60, minutes * 60), tz=_dt.timezone.utc)

    shards = _list_shards(client, stream_name)
    out: list[dict[str, Any]] = []
    for shard in shards:
        shard_id = shard.get("ShardId")
        if not shard_id:
            continue
        iterator = _get_iterator_at(client, stream_name, shard_id, since_dt)
        if not iterator:
            continue
        taken = 0
        # Single GetRecords call is usually enough for a 5-minute window
        # of a modestly-sized stream. We cap the read to max_records_per_call.
        try:
            resp = client.get_records(ShardIterator=iterator, Limit=max_records_per_call)
        except Exception as exc:  # noqa: BLE001
            log.warning("kinesis get_records shard=%s failed: %s", shard_id, exc)
            continue
        records = resp.get("Records", [])
        # Evenly sample across the returned batch.
        step = max(1, len(records) // max(1, sample_per_shard)) if records else 1
        for i, r in enumerate(records):
            if i % step != 0:
                continue
            if taken >= sample_per_shard:
                break
            taken += 1
            out.append({
                "shardId": shard_id,
                "partitionKey": r.get("PartitionKey"),
                "sequenceNumber": r.get("SequenceNumber"),
                "approximateArrivalTimestamp": _iso(r.get("ApproximateArrivalTimestamp")),
                "data": _decode_data(r.get("Data") or b""),
            })

    return {
        "stream": stream_name,
        "since": since_dt.isoformat(),
        "shards_sampled": len([s for s in shards if s.get("ShardId")]),
        "records_returned": len(out),
        "records": out,
    }


def _list_shards(client: Any, stream_name: str) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"StreamName": stream_name} if not next_token else {"NextToken": next_token}
        try:
            resp = client.list_shards(**kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning("kinesis list_shards failed: %s", exc)
            return shards
        shards.extend(resp.get("Shards") or [])
        next_token = resp.get("NextToken")
        if not next_token:
            break
    return shards


def _get_iterator_at(client: Any, stream_name: str, shard_id: str, since: _dt.datetime) -> str | None:
    try:
        resp = client.get_shard_iterator(
            StreamName=stream_name,
            ShardId=shard_id,
            ShardIteratorType="AT_TIMESTAMP",
            Timestamp=since,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("kinesis get_shard_iterator shard=%s failed: %s", shard_id, exc)
        return None
    return resp.get("ShardIterator")


__all__ = ["kinesis_tail", "ClientFactory"]
