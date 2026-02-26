#!/usr/bin/env python3
"""Live smoke test for DS Chat Investigation end-to-end pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import Runner
from chatkit.agents import AgentContext
from chatkit.types import ThreadMetadata

from app.agents.orchestrator import build_agent
from app.persistent_store import SQLiteStore, default_sqlite_path
from app.workspace_manager import WorkspaceManager


@dataclass
class Scenario:
    name: str
    prompt: str
    required_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...] = ()
    required_output_substrings: tuple[str, ...] = ()


def _clip(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(truncated)"


def _to_text(value: Any, max_chars: int = 6000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clip(value, max_chars=max_chars)
    try:
        return _clip(json.dumps(value, ensure_ascii=False, default=str), max_chars=max_chars)
    except Exception:
        return _clip(str(value), max_chars=max_chars)


def _extract_tool_name(raw_item: Any) -> str:
    if isinstance(raw_item, dict):
        return str(raw_item.get("name") or raw_item.get("function", {}).get("name") or "")
    return str(getattr(raw_item, "name", ""))


def _extract_tool_args(raw_item: Any) -> str:
    if isinstance(raw_item, dict):
        return _to_text(raw_item.get("arguments") or raw_item.get("function", {}).get("arguments") or "", max_chars=2000)
    return _to_text(getattr(raw_item, "arguments", ""), max_chars=2000)


def _bootstrap_aws_credentials(profile: str) -> None:
    proc = subprocess.run(
        ["granted", "credential-process", "--profile", profile, "--auto-login"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    os.environ["AWS_ACCESS_KEY_ID"] = str(payload.get("AccessKeyId") or "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = str(payload.get("SecretAccessKey") or "")
    os.environ["AWS_SESSION_TOKEN"] = str(payload.get("SessionToken") or "")
    os.environ.setdefault("AWS_REGION", "us-east-1")


def _run_scenario(
    *,
    scenario: Scenario,
    model: str,
    max_turns: int,
    store: SQLiteStore,
    workspace_manager: WorkspaceManager,
) -> dict[str, Any]:
    thread = ThreadMetadata(
        id=f"smoke_{uuid.uuid4().hex[:10]}",
        created_at=datetime.now(timezone.utc),
        title=f"Smoke {scenario.name}",
    )
    turn_id = f"turn_{uuid.uuid4().hex[:8]}"
    workspace = workspace_manager.create_turn_workspace(thread.id, turn_id)

    context = AgentContext(
        thread=thread,
        store=store,
        request_context={},
    )

    agent = build_agent(None, model)
    final_output = ""
    tool_calls: list[str] = []
    tool_outputs: list[str] = []
    debug_steps: list[dict[str, Any]] = []
    cleanup_report: dict[str, Any] | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = Runner.run_sync(agent, scenario.prompt, context=context, max_turns=max_turns)
        final_output = str(getattr(result, "final_output", "") or "")

        for index, item in enumerate(getattr(result, "new_items", []) or [], start=1):
            item_type = getattr(item, "type", "")
            raw_item = getattr(item, "raw_item", None)
            step: dict[str, Any] = {
                "index": index,
                "item_type": item_type,
            }
            if item_type == "tool_call_item":
                tool_name = _extract_tool_name(raw_item)
                tool_calls.append(tool_name)
                step["tool_name"] = tool_name
                step["arguments"] = _extract_tool_args(raw_item)
            elif item_type == "tool_call_output_item":
                output = getattr(item, "output", None)
                if output is None and isinstance(raw_item, dict):
                    output = raw_item.get("output")
                output_text = _to_text(output, max_chars=4000)
                tool_outputs.append(output_text)
                step["output"] = output_text
            elif item_type in {"handoff_call_item", "handoff_output_item"}:
                step["detail"] = _to_text(raw_item, max_chars=1500)
            elif item_type == "message_output_item":
                step["message_output"] = _to_text(raw_item, max_chars=2000)
            debug_steps.append(step)
    finally:
        cleanup_report = workspace.cleanup()
    ended_at = datetime.now(timezone.utc).isoformat()

    missing = [name for name in scenario.required_tools if name not in tool_calls]
    forbidden = [name for name in scenario.forbidden_tools if name in tool_calls]
    output_blob = "\n".join(tool_outputs)
    missing_output = [text for text in scenario.required_output_substrings if text not in output_blob]
    failed = bool(missing or forbidden or missing_output or not (cleanup_report or {}).get("deleted"))

    return {
        "scenario": scenario.name,
        "failed": failed,
        "missing_tools": missing,
        "forbidden_tools": forbidden,
        "missing_output_substrings": missing_output,
        "tool_calls": tool_calls,
        "prompt": scenario.prompt,
        "started_at": started_at,
        "ended_at": ended_at,
        "final_output": final_output,
        "debug_steps": debug_steps,
        "cleanup": cleanup_report,
    }


def _build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="top_site_issues",
            prompt=(
                "Run full end-to-end now with concrete counts. "
                "what are the top site issues for provider QL2 on 20260211 for customer AA?"
            ),
            required_tools=(
                "investigate_issue",
            ),
        ),
        Scenario(
            name="anomaly_deep_dive",
            prompt=(
                "Run full end-to-end now with concrete counts. "
                "investigate anomalies for customer B6 on 20260211 and summarize findings."
            ),
            required_tools=(
                "investigate_issue",
            ),
        ),
        Scenario(
            name="missing_partition_clarification",
            prompt="what are the top site issues for provider QL2 on 20260211",
            required_tools=("investigate_issue",),
        ),
        Scenario(
            name="mysql_fallback_resolution",
            prompt=(
                "Run full end-to-end now with concrete counts. "
                "what are the top site issues for provider UA on 20260211 for customer AA?"
            ),
            required_tools=(
                "resolve_entities",
                "investigate_issue",
            ),
        ),
        Scenario(
            name="provider_site_pipe_notation",
            prompt="what are the top site issues for QL2|AV",
            required_tools=("investigate_issue",),
        ),
        Scenario(
            name="impact_ambiguous",
            prompt="what is the impact",
            required_tools=(),
        ),
        Scenario(
            name="collection_anomalies_yersterday",
            prompt="what were the customer collection anomalies yersterday?",
            required_tools=(),
        ),
        Scenario(
            name="market_anomalies_distribution",
            prompt=(
                "what are the market anomalies today for customer B6, "
                "can you give me a distribution of impact score?"
            ),
            required_tools=(
                "investigate_issue",
            ),
        ),
    ]


def _render_markdown_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# DS Chat Investigation Smoke Report")
    lines.append("")
    lines.append(f"- Profile: `{payload.get('profile')}`")
    lines.append(f"- Model: `{payload.get('model')}`")
    lines.append(f"- Generated At (UTC): `{payload.get('generated_at')}`")
    lines.append("")

    reports = payload.get("reports", [])
    for report in reports:
        lines.append(f"## Scenario: `{report.get('scenario')}`")
        lines.append("")
        lines.append(f"- Failed: `{report.get('failed')}`")
        lines.append(f"- Missing tools: `{report.get('missing_tools')}`")
        lines.append(f"- Forbidden tools triggered: `{report.get('forbidden_tools')}`")
        lines.append(f"- Missing output checks: `{report.get('missing_output_substrings')}`")
        lines.append(f"- Started: `{report.get('started_at')}`")
        lines.append(f"- Ended: `{report.get('ended_at')}`")
        lines.append("")
        lines.append("### Prompt")
        lines.append("")
        lines.append("```text")
        lines.append(str(report.get("prompt", "")))
        lines.append("```")
        lines.append("")
        lines.append("### Debug Steps")
        lines.append("")
        for step in report.get("debug_steps", []):
            index = step.get("index")
            item_type = step.get("item_type")
            tool_name = step.get("tool_name")
            if tool_name:
                lines.append(f"{index}. `{item_type}` -> `{tool_name}`")
                args = step.get("arguments")
                if args:
                    lines.append("```text")
                    lines.append(str(args))
                    lines.append("```")
                output = step.get("output")
                if output:
                    lines.append("```text")
                    lines.append(str(output))
                    lines.append("```")
            else:
                lines.append(f"{index}. `{item_type}`")
                for key in ("detail", "message_output"):
                    value = step.get(key)
                    if value:
                        lines.append("```text")
                        lines.append(str(value))
                        lines.append("```")
        lines.append("")
        lines.append("### Final Model Output")
        lines.append("")
        lines.append("```text")
        lines.append(str(report.get("final_output", "")))
        lines.append("```")
        lines.append("")
        lines.append("### Cleanup")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(report.get("cleanup", {}), indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DS Chat Investigation E2E smoke pipeline checks.")
    parser.add_argument("--profile", default="3VDEV", help="AWS granted profile name (default: 3VDEV)")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to run for the agent.")
    parser.add_argument("--max-turns", type=int, default=40, help="Max model/tool turns per scenario.")
    parser.add_argument(
        "--scenarios",
        default="",
        help="Optional comma-separated scenario names to run (defaults to all).",
    )
    parser.add_argument(
        "--report-dir",
        default=str((Path(__file__).resolve().parents[1] / ".runtime" / "smoke_reports").resolve()),
        help="Directory to write full JSON/Markdown reports.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run live agent smoke test.", file=sys.stderr)
        return 2

    _bootstrap_aws_credentials(args.profile)

    store = SQLiteStore(default_sqlite_path())
    workspace_manager = WorkspaceManager()

    scenarios = _build_scenarios()
    if args.scenarios.strip():
        selected = {token.strip() for token in args.scenarios.split(",") if token.strip()}
        scenarios = [scenario for scenario in scenarios if scenario.name in selected]
        if not scenarios:
            print("No scenarios matched --scenarios filter.", file=sys.stderr)
            return 2

    reports = [
        _run_scenario(
            scenario=scenario,
            model=args.model,
            max_turns=max(10, int(args.max_turns)),
            store=store,
            workspace_manager=workspace_manager,
        )
        for scenario in scenarios
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "profile": args.profile,
        "model": args.model,
        "generated_at": generated_at,
        "reports": reports,
    }

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = (report_dir / f"smoke_investigation_{run_id}.json").resolve()
    md_path = (report_dir / f"smoke_investigation_{run_id}.md").resolve()

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(payload), encoding="utf-8")

    summary = {
        "profile": args.profile,
        "model": args.model,
        "generated_at": generated_at,
        "report_json": str(json_path),
        "report_md": str(md_path),
        "scenario_status": [
            {
                "scenario": report.get("scenario"),
                "failed": report.get("failed"),
                "tool_calls": report.get("tool_calls", []),
            }
            for report in reports
        ],
    }
    print(json.dumps(summary, indent=2))
    failed = [report for report in reports if report.get("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
