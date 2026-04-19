"""@function_tool wrappers for Kinesis stream inspection."""

from __future__ import annotations

import logging
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent

from ..ops import streams_client as sc
from ._common import TIMEOUT_AWS, tool_error

log = logging.getLogger(__name__)


async def _stream(ctx: RunContextWrapper[AgentContext], icon: str, text: str) -> None:  # type: ignore[type-arg]
    try:
        await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))
    except Exception:
        pass


@function_tool(timeout=TIMEOUT_AWS, failure_error_function=tool_error)
async def kinesis_tail(
    ctx: RunContextWrapper[AgentContext],
    stream_name: str,
    minutes: int = 5,
    sample_per_shard: int = 5,
) -> dict[str, Any]:
    """Sample recent records across all shards of a Kinesis stream (read-only).

    Great for "is QL2 sending data right now?" investigations against the
    10 ingest-*-raw-search streams. Each returned record carries
    partitionKey, sequenceNumber, arrival timestamp, shardId, and
    JSON-decoded `data` (falls back to utf-8 / base64).

    Args:
        stream_name: Kinesis stream name (e.g. ingest-priceeye-raw-search).
        minutes: Lookback window (default 5, minimum 1).
        sample_per_shard: Max records returned per shard (default 5).
    """
    try:
        await _stream(ctx, "clock", f"Tailing Kinesis stream {stream_name} (last {minutes}m).")
        result = sc.kinesis_tail(
            stream_name,
            minutes=max(1, int(minutes)),
            sample_per_shard=max(1, int(sample_per_shard)),
        )
        await _stream(
            ctx,
            "check-circle",
            f"{result.get('shards_sampled', 0)} shards, {result.get('records_returned', 0)} records.",
        )
        return {"ok": True, **result}
    except Exception as exc:
        log.exception("kinesis_tail failed")
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


def streams_tools() -> list[Any]:
    return [kinesis_tail]


__all__ = ["kinesis_tail", "streams_tools"]
