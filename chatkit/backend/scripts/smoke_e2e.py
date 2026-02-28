#!/usr/bin/env python3
"""Unified E2E smoke test for the DS Chat agentic investigation pipeline.

Replaces the legacy run_investigation_e2e.py and smoke_investigation_pipeline.py.
Runs test cases through the actual agentic loop using Runner.run().

Usage:
    cd chatkit/backend
    eval "$(assume 3VDEV)"
    .venv/bin/python scripts/smoke_e2e.py --profile 3VDEV --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Bootstrap path ──
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from agents import Runner  # type: ignore[import]

from app.agents.investigation_agent import build_investigation_agent
from app.investigation.runtime import cleanup_thread_workspace, get_runtime


def _bootstrap_aws_credentials(profile: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["zsh", "-lc", f"assume {profile} >/dev/null 2>&1; env -0"],
        capture_output=True,
        text=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"Failed to assume profile {profile}: {stderr.strip() or 'unknown error'}")

    output = proc.stdout.decode("utf-8", errors="replace")
    loaded = 0
    for pair in output.split("\x00"):
        if not pair or "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        if key.startswith("AWS_"):
            os.environ[key] = value
            loaded += 1

    if loaded == 0:
        fallback = subprocess.run(
            ["granted", "credential-process", "--profile", profile, "--auto-login"],
            capture_output=True,
            text=True,
        )
        if fallback.returncode != 0:
            stderr = fallback.stderr or ""
            raise RuntimeError(f"Credential fallback failed for {profile}: {stderr.strip() or 'unknown error'}")
        payload = json.loads(fallback.stdout)
        os.environ["AWS_ACCESS_KEY_ID"] = str(payload.get("AccessKeyId") or "")
        os.environ["AWS_SECRET_ACCESS_KEY"] = str(payload.get("SecretAccessKey") or "")
        os.environ["AWS_SESSION_TOKEN"] = str(payload.get("SessionToken") or "")
        loaded = 3

    os.environ.setdefault("AWS_REGION", "us-east-1")
    return {"profile": profile, "env_keys_loaded": loaded}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
    else:
        cases = payload
    if not isinstance(cases, list):
        raise ValueError("Cases file must contain a list or object with a 'cases' list.")
    return [case for case in cases if isinstance(case, dict)]


class _MinimalAgentContext:
    """Minimal context for E2E testing without the full chatkit server."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self._thread = _MinimalThread(thread_id)

    @property
    def thread(self) -> Any:
        return self._thread

    async def stream(self, event: Any) -> None:
        """No-op stream handler for E2E."""

    async def stream_widget(self, widget: Any, **kwargs: Any) -> None:
        """No-op widget handler for E2E."""


class _MinimalThread:
    def __init__(self, thread_id: str) -> None:
        self.id = thread_id


class _MinimalStore:
    """No-op store for E2E."""

    async def save_attachment(self, attachment: Any, **kwargs: Any) -> None:
        pass


def _extract_tool_calls(result: Any) -> list[dict[str, Any]]:
    """Extract tool call info from a Runner.run() result."""
    tool_calls: list[dict[str, Any]] = []
    # Walk through new_items looking for tool call outputs
    for item in getattr(result, "new_items", []):
        item_type = getattr(item, "type", "")
        if item_type == "tool_call_item":
            tool_calls.append({
                "tool": getattr(item, "name", getattr(item, "tool_name", "unknown")),
                "call_id": getattr(item, "call_id", ""),
            })
        elif item_type == "function_call_output":
            # Match back to existing tool calls if possible
            call_id = getattr(item, "call_id", "")
            output = getattr(item, "output", "")
            for tc in tool_calls:
                if tc.get("call_id") == call_id:
                    tc["output_preview"] = str(output)[:200]
                    break
    return tool_calls


def _check_assertions(
    case: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """Check assertions defined in the test case."""
    assertions = case.get("assertions")
    if not assertions:
        return {"checked": False, "reason": "no assertions defined"}

    results: dict[str, Any] = {"checked": True, "passed": True, "details": []}

    # min_tool_calls
    min_tc = assertions.get("min_tool_calls")
    if min_tc is not None:
        ok = len(tool_calls) >= min_tc
        results["details"].append({
            "assertion": "min_tool_calls",
            "expected": min_tc,
            "actual": len(tool_calls),
            "passed": ok,
        })
        if not ok:
            results["passed"] = False

    # required_tools
    required = assertions.get("required_tools", [])
    actual_tools = {tc.get("tool", "") for tc in tool_calls}
    for tool_name in required:
        ok = tool_name in actual_tools
        results["details"].append({
            "assertion": "required_tool",
            "expected": tool_name,
            "present": ok,
            "passed": ok,
        })
        if not ok:
            results["passed"] = False

    # answer_contains
    for keyword in assertions.get("answer_contains", []):
        ok = keyword.lower() in answer.lower()
        results["details"].append({
            "assertion": "answer_contains",
            "keyword": keyword,
            "passed": ok,
        })
        if not ok:
            results["passed"] = False

    return results


async def run_case(
    agent: Any,
    case: dict[str, Any],
    max_turns: int = 30,
) -> dict[str, Any]:
    """Run a single E2E test case through the agentic loop."""
    thread_id = str(case.get("thread_id") or f"thread-e2e-{case.get('name', 'unknown')}")
    question = str(case.get("question", ""))
    context = _MinimalAgentContext(thread_id)

    started = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "name": case.get("name"),
        "thread_id": thread_id,
        "question": question,
        "started_at": started.isoformat(),
    }

    try:
        result = await Runner.run(
            agent,
            input=[{"role": "user", "content": question}],
            context=context,
            max_turns=max_turns,
        )

        tool_calls = _extract_tool_calls(result)
        answer = str(getattr(result, "final_output", "") or "")

        report["tool_calls"] = tool_calls
        report["tool_call_count"] = len(tool_calls)
        report["answer"] = answer
        report["answer_length"] = len(answer)
        report["assertions"] = _check_assertions(case, tool_calls, answer)
        report["failed"] = False

    except Exception as exc:
        report["failed"] = True
        report["error"] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["ended_at"] = datetime.now(timezone.utc).isoformat()
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        report["elapsed_seconds"] = round(elapsed, 1)
        try:
            cleanup_thread_workspace(thread_id, mode="ephemeral_manifest")
        except Exception as cleanup_exc:
            report["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"

    return report


def _render_markdown_report(payload: dict[str, Any]) -> str:
    """Render a human-readable markdown report."""
    lines: list[str] = [
        "# E2E Smoke Test Report",
        "",
        f"- Generated: {payload.get('generated_at')}",
        f"- Model: {payload.get('model')}",
        f"- Cases: {len(payload.get('reports', []))}",
        "",
    ]
    for report in payload.get("reports", []):
        name = report.get("name", "unknown")
        failed = report.get("failed", False)
        status = "FAIL" if failed else "PASS"
        lines.append(f"## [{status}] {name}")
        lines.append("")
        lines.append(f"- Question: {report.get('question')}")
        lines.append(f"- Elapsed: {report.get('elapsed_seconds', '?')}s")
        lines.append(f"- Tool calls: {report.get('tool_call_count', 0)}")
        lines.append("")

        if report.get("error"):
            lines.append(f"**Error:** `{report['error'].get('error_type')}: {report['error'].get('message')}`")
            lines.append("")

        tool_calls = report.get("tool_calls", [])
        if tool_calls:
            tool_names = [tc.get("tool", "?") for tc in tool_calls]
            lines.append(f"**Tool sequence:** {' -> '.join(tool_names)}")
            lines.append("")

        assertions = report.get("assertions", {})
        if assertions.get("checked"):
            passed = assertions.get("passed", False)
            lines.append(f"**Assertions:** {'ALL PASSED' if passed else 'SOME FAILED'}")
            for detail in assertions.get("details", []):
                mark = "pass" if detail.get("passed") else "FAIL"
                lines.append(f"  - [{mark}] {detail.get('assertion')}: {json.dumps({k: v for k, v in detail.items() if k not in ('assertion', 'passed')})}")
            lines.append("")

        answer = report.get("answer", "")
        if answer:
            truncated = answer[:500] + ("..." if len(answer) > 500 else "")
            lines.append("**Answer preview:**")
            lines.append("```")
            lines.append(truncated)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


async def run_all(args: argparse.Namespace) -> int:
    """Main async entry point."""
    if not args.skip_bootstrap:
        cred = _bootstrap_aws_credentials(args.profile)
    else:
        cred = {"profile": args.profile, "skipped": True}

    cases_path = Path(args.cases_file).expanduser().resolve()
    cases = _load_cases(cases_path)

    if args.scenarios:
        selected = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        cases = [c for c in cases if str(c.get("name")) in selected]

    if not cases:
        print("No test cases selected.")
        return 1

    agent = build_investigation_agent(args.model)
    print(f"Agent: {agent.name}, tools: {len(agent.tools)}, model: {args.model}")
    print(f"Running {len(cases)} E2E test cases...\n")

    reports: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        name = case.get("name", f"case_{idx}")
        print(f"[{idx}/{len(cases)}] {name} ...", flush=True)
        report = await run_case(agent, case, max_turns=args.max_turns)
        status = "FAIL" if report.get("failed") else "PASS"
        elapsed = report.get("elapsed_seconds", "?")
        tc_count = report.get("tool_call_count", 0)
        print(f"  -> [{status}] {elapsed}s, {tc_count} tool calls")

        assertions = report.get("assertions", {})
        if assertions.get("checked") and not assertions.get("passed"):
            for detail in assertions.get("details", []):
                if not detail.get("passed"):
                    print(f"  -> ASSERTION FAIL: {detail}")

        reports.append(report)

    # Write reports
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "model": args.model,
        "max_turns": args.max_turns,
        "credential_bootstrap": cred,
        "cases_file": str(cases_path),
        "reports": reports,
    }

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"e2e_smoke_{stamp}.json"
    md_path = report_dir / f"e2e_smoke_{stamp}.md"

    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown_report(payload), encoding="utf-8")

    # Summary
    passed = sum(1 for r in reports if not r.get("failed"))
    failed = len(reports) - passed
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(reports)} cases")
    print(f"Reports: {json_path}")
    print(f"         {md_path}")

    return 1 if failed > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E2E smoke tests for DS Chat investigation agent.")
    parser.add_argument("--profile", default="3VDEV", help="Credential profile for assume (default: 3VDEV)")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use for the agent (default: gpt-4.1-mini)")
    parser.add_argument("--max-turns", type=int, default=30, help="Max agentic turns per case (default: 30)")
    parser.add_argument(
        "--cases-file",
        default=str(BACKEND_ROOT / "tests" / "e2e_investigation_cases.json"),
        help="Path to JSON test-case file",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Optional comma-separated scenario names to run (default: all)",
    )
    parser.add_argument(
        "--report-dir",
        default=str(BACKEND_ROOT / ".runtime" / "e2e_reports"),
        help="Directory for report output",
    )
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip AWS credential bootstrap")
    args = parser.parse_args()

    return asyncio.run(run_all(args))


if __name__ == "__main__":
    raise SystemExit(main())
