#!/usr/bin/env python3
"""Interactive CLI chat for the DS Chat investigation agent.

Usage:
    cd chatkit/backend
    .venv/bin/python tests/cli_chat.py --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Bootstrap path so `app.*` imports resolve when running from tests/.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from agents import Runner  # type: ignore[import]

from app.agents.investigation_agent import build_investigation_agent
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
    return parser.parse_args()


async def _run_repl(args: argparse.Namespace) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Export it (or source .env.local) before running.")
        return 1

    thread_id = args.thread_id or _new_thread_id()
    context = _CliAgentContext(thread_id=thread_id)
    agent = build_investigation_agent(args.model)
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
