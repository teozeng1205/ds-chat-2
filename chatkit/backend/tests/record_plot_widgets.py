#!/usr/bin/env python3
"""Record widgets + progress events emitted during a plotting run.

Drives the real agent (same build_agent as production) and captures
every stream_widget / stream call so we can confirm the Card payloads
for publish_image + render_image actually carry a valid image URL.

Usage:
    .venv/bin/python tests/record_plot_widgets.py [--model gpt-5.4-mini]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from agents import Runner  # type: ignore[import]

from app.agents.ds_agent import build_agent


class _RecordingCtx:
    """Captures every widget + progress event for inspection."""

    def __init__(self, thread_id: str) -> None:
        self.thread = _Thread(thread_id)
        self.store = _Store()
        self.request_context: dict[str, Any] = {}
        self.widgets: list[dict[str, Any]] = []
        self.progress: list[dict[str, Any]] = []

    async def stream(self, event: Any) -> None:
        self.progress.append({
            "icon": getattr(event, "icon", None),
            "text": getattr(event, "text", None),
        })

    async def stream_widget(self, widget: Any, copy_text: str | None = None, **kwargs: Any) -> None:
        self.widgets.append({
            "type": type(widget).__name__,
            "copy_text": copy_text,
            "repr": repr(widget)[:1500],
        })


class _Thread:
    def __init__(self, thread_id: str) -> None:
        self.id = thread_id


class _Store:
    async def save_attachment(self, attachment: Any, **kwargs: Any) -> None:
        pass


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()

    agent = build_agent(args.model)
    ctx = _RecordingCtx(thread_id="widget-record-test")

    prompt = (
        "Generate 90 days of synthetic daily collection-request counts with a "
        "realistic upward trend plus noise. Plot it with a red trend line overlaid "
        "on the blue series. Save the PNG under /tmp and publish it as an image "
        "card so the user can see it. Keep it tight."
    )

    result = await Runner.run(
        agent,
        input=[{"role": "user", "content": prompt}],
        context=ctx,
        max_turns=30,
    )

    tool_calls: list[str] = []
    for item in result.new_items:
        if getattr(item, "type", "") == "tool_call_item":
            raw = getattr(item, "raw_item", {}) or {}
            if isinstance(raw, dict):
                tool_calls.append(str(raw.get("name") or "?"))
            else:
                tool_calls.append(str(getattr(raw, "name", None) or "?"))

    print(json.dumps({
        "answer": str(result.final_output or "")[:500],
        "tool_calls": tool_calls,
        "widget_count": len(ctx.widgets),
        "widgets": ctx.widgets,
        "progress_count": len(ctx.progress),
        "progress_last5": ctx.progress[-5:],
    }, indent=2, default=str))
    return 0 if ctx.widgets else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
