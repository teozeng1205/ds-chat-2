#!/usr/bin/env python3
"""Interactive CLI chat for the DS Chat investigation agent.

Usage:
    cd chatkit/backend
    .venv/bin/python tests/cli_chat.py --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Bootstrap path so `app.*` imports resolve when running from tests/.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from agents import Runner  # type: ignore[import]

from app.agents.ds_agent import build_agent
from app.investigation.runtime import cleanup_thread_workspace


def _new_thread_id() -> str:
    return f"cli-{uuid.uuid4().hex[:12]}"


class _CliThread:
    def __init__(self, thread_id: str) -> None:
        self.id = thread_id


class _CliStore:
    async def save_attachment(self, attachment: Any, **kwargs: Any) -> None:
        _ = (attachment, kwargs)


class _CliAgentContext:
    """Small context adapter with the fields/tools expected by runtime code."""

    def __init__(self, thread_id: str) -> None:
        self.thread = _CliThread(thread_id)
        self.store = _CliStore()
        self.request_context: dict[str, Any] = {}

    async def stream(self, event: Any) -> None:
        icon = str(getattr(event, "icon", "")).strip()
        text = str(getattr(event, "text", "")).strip()
        if icon or text:
            prefix = f"{icon} " if icon else ""
            print(f"[tool] {prefix}{text}".rstrip())

    async def stream_widget(self, widget: Any, **kwargs: Any) -> None:
        copy_text = kwargs.get("copy_text")
        if copy_text:
            print(f"[widget] {copy_text}")
            return
        widget_type = type(widget).__name__
        print(f"[widget] {widget_type} emitted")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI for DS Chat investigation agent.")
    parser.add_argument("--model", default="gpt-5-mini", help="Model for the agent (default: gpt-5-mini)")
    parser.add_argument("--max-turns", type=int, default=50, help="Max agent turns per user message (default: 50)")
    parser.add_argument("--thread-id", default=None, help="Optional fixed thread id")
    parser.add_argument(
        "--cleanup-mode",
        default="none",
        choices=["none", "ephemeral_manifest", "all"],
        help="Workspace cleanup mode after each turn (default: none)",
    )
    parser.add_argument(
        "--no-log-tools",
        action="store_false",
        dest="log_tools",
        help="Disable tool-call logs for each turn",
    )
    parser.set_defaults(log_tools=True)
    return parser.parse_args()


def _compact_json(text: str, max_len: int = 220) -> str:
    raw = text.strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
        rendered = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        rendered = raw
    if len(rendered) > max_len:
        return rendered[: max_len - 3] + "..."
    return rendered


def _log_tool_calls(result: Any) -> None:
    calls: dict[str, dict[str, str]] = {}
    ordered_ids: list[str] = []
    counter = 0
    for item in getattr(result, "new_items", []):
        item_type = getattr(item, "type", "")
        raw = getattr(item, "raw_item", None)
        if item_type == "tool_call_item":
            call_id = str(getattr(raw, "call_id", "") or getattr(item, "call_id", "") or f"call_{counter}")
            counter += 1
            name = str(getattr(raw, "name", "") or getattr(item, "name", "") or "unknown_tool")
            args = str(getattr(raw, "arguments", "") or "")
            calls[call_id] = {"name": name, "arguments": _compact_json(args), "output": ""}
            ordered_ids.append(call_id)
        elif item_type == "tool_call_output_item":
            call_id = str(getattr(raw, "call_id", "") or getattr(item, "call_id", ""))
            output = str(getattr(item, "output", "") or getattr(raw, "output", "") or "")
            if call_id not in calls:
                calls[call_id] = {"name": "unknown_tool", "arguments": "", "output": _compact_json(output)}
                ordered_ids.append(call_id)
            else:
                calls[call_id]["output"] = _compact_json(output)

    if not ordered_ids:
        print("[tool-call] none")
        return

    for idx, call_id in enumerate(ordered_ids, start=1):
        call = calls.get(call_id, {})
        name = call.get("name", "unknown_tool")
        args = call.get("arguments", "")
        output = call.get("output", "")
        if args:
            print(f"[tool-call {idx}] {name} args={args}")
        else:
            print(f"[tool-call {idx}] {name}")
        if output:
            print(f"[tool-output {idx}] {output}")


async def _run_repl(args: argparse.Namespace) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Export it (or source .env.local) before running.")
        return 1

    thread_id = args.thread_id or _new_thread_id()
    context = _CliAgentContext(thread_id=thread_id)
    agent = build_agent(args.model)
    conversation: list[dict[str, Any]] = []

    print(f"Thread: {thread_id}")
    print(f"Model: {args.model}")
    print("Commands: /exit, /quit, /reset, /thread")

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/thread":
            print(f"Current thread: {thread_id}")
            continue
        if user_text == "/reset":
            if args.cleanup_mode != "none":
                cleanup_thread_workspace(thread_id, mode=args.cleanup_mode)
            thread_id = _new_thread_id()
            context = _CliAgentContext(thread_id=thread_id)
            conversation = []
            print(f"Conversation reset. New thread: {thread_id}")
            continue

        conversation.append({"role": "user", "content": user_text})
        try:
            result = await Runner.run(
                agent,
                input=conversation,
                context=context,
                max_turns=args.max_turns,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Agent error: {exc}")
            continue

        answer = str(getattr(result, "final_output", "") or "").strip()
        if args.log_tools:
            _log_tool_calls(result)
        print(f"\nAgent> {answer or '(no output)'}")
        conversation = result.to_input_list()

        if args.cleanup_mode != "none":
            cleanup_thread_workspace(thread_id, mode=args.cleanup_mode)

    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run_repl(args))


if __name__ == "__main__":
    raise SystemExit(main())
