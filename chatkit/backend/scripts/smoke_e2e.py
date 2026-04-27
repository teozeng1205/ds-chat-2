#!/usr/bin/env python3
"""Unified E2E smoke test for the DS Chat agentic investigation pipeline.

Runs test cases through the actual agentic loop using Runner.run().

Usage:
    cd chatkit/backend
    eval "$(assume 3VDEV)"
    .venv/bin/python scripts/smoke_e2e.py --profile 3VDEV --model gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Bootstrap path ──
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from agents import Runner, gen_trace_id, trace  # noqa: E402  # type: ignore[import]

from app.agents.ds_agent import build_agent as build_investigation_agent  # noqa: E402
from app.investigation.runtime import cleanup_thread_workspace  # noqa: E402


_HOSTED_TOOL_TYPE_NAMES = {
    "web_search_call": "web_search",
    "file_search_call": "file_search",
    "computer_call": "computer",
    "code_interpreter_call": "code_interpreter",
}

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
        self.store = _MinimalStore()
        self.request_context: dict[str, Any] = {}

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
    for item in getattr(result, "new_items", []):
        item_type = getattr(item, "type", "")
        if item_type == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            raw_type = _raw_value(raw, "type", "")
            name = (
                _raw_value(raw, "name")
                or _HOSTED_TOOL_TYPE_NAMES.get(str(raw_type), "")
                or getattr(item, "name", "")
                or "unknown"
            )
            call_id = _raw_value(raw, "call_id", "") or getattr(item, "call_id", "")
            arguments = _raw_value(raw, "arguments", "") or ""
            tool_calls.append({
                "tool": name,
                "call_id": call_id,
                "arguments": arguments,
                "raw_type": raw_type,
            })
        elif item_type == "tool_call_output_item":
            # Match output back to existing tool calls
            raw = getattr(item, "raw_item", None)
            call_id = (
                _raw_value(raw, "call_id", "")
                or getattr(item, "call_id", "")
                or getattr(getattr(item, "agent_call", None), "call_id", "")
            )
            output = (
                getattr(item, "output", None)
                or _raw_value(raw, "output", None)
                or _raw_value(raw, "content", None)
                or ""
            )
            matched = False
            for tc in tool_calls:
                if call_id and tc.get("call_id") == call_id:
                    tc["output"] = str(output)
                    matched = True
                    break
            if not matched:
                for tc in reversed(tool_calls):
                    if "output" not in tc:
                        tc["output"] = str(output)
                        break
    return tool_calls


def _raw_value(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _is_retryable_agent_error(exc: Exception) -> bool:
    error_type = type(exc).__name__.lower()
    message = str(exc).lower()
    retryable_fragments = (
        "request timed out",
        "rate limit",
        "connection error",
        "connection reset",
        "temporarily unavailable",
    )
    return "timeout" in error_type or any(fragment in message for fragment in retryable_fragments)


def _tool_error_type(output: str) -> str | None:
    if "'ok': False" not in output and '"ok": false' not in output.lower():
        return None
    for pattern in (
        r"'error_type': '([^']+)'",
        r'"error_type"\s*:\s*"([^"]+)"',
    ):
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    return "ToolError"


def _check_assertions(case: dict[str, Any], tool_calls: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    assertions = case.get("assertions")
    if not isinstance(assertions, dict):
        return {"checked": False, "passed": True, "details": []}

    details: list[dict[str, Any]] = []
    passed = True

    def add(assertion: str, ok: bool, **extra: Any) -> None:
        nonlocal passed
        details.append({"assertion": assertion, "passed": ok, **extra})
        if not ok:
            passed = False

    tool_names = [str(tc.get("tool", "")) for tc in tool_calls]
    tool_set = set(tool_names)

    min_tool_calls = assertions.get("min_tool_calls")
    if min_tool_calls is not None:
        add("min_tool_calls", len(tool_calls) >= int(min_tool_calls), expected=min_tool_calls, actual=len(tool_calls))

    max_tool_calls = assertions.get("max_tool_calls")
    if max_tool_calls is not None:
        add("max_tool_calls", len(tool_calls) <= int(max_tool_calls), expected=max_tool_calls, actual=len(tool_calls))

    for tool_name in assertions.get("required_tools", []) or []:
        add("required_tool", str(tool_name) in tool_set, expected=tool_name, actual=tool_names)

    for tool_name in assertions.get("forbidden_tools", []) or []:
        add("forbidden_tool", str(tool_name) not in tool_set, expected_absent=tool_name, actual=tool_names)

    answer_lower = answer.lower()
    for keyword in assertions.get("answer_contains", []) or []:
        add("answer_contains", str(keyword).lower() in answer_lower, keyword=keyword)

    for keyword in assertions.get("answer_not_contains", []) or []:
        add("answer_not_contains", str(keyword).lower() not in answer_lower, keyword=keyword)

    tool_errors: list[dict[str, Any]] = []
    for idx, tc in enumerate(tool_calls, 1):
        error_type = _tool_error_type(str(tc.get("output", "")))
        if error_type:
            tool_errors.append({"index": idx, "tool": tc.get("tool"), "error_type": error_type})

    fail_on = {str(item) for item in assertions.get("fail_on_tool_error_types", []) or []}
    if fail_on:
        offending = [err for err in tool_errors if err.get("error_type") in fail_on]
        add("fail_on_tool_error_types", not offending, expected_absent=sorted(fail_on), actual=offending)

    max_tool_errors = assertions.get("max_tool_errors")
    if max_tool_errors is not None:
        add("max_tool_errors", len(tool_errors) <= int(max_tool_errors), expected=max_tool_errors, actual=tool_errors)

    return {"checked": True, "passed": passed, "details": details}


async def run_case(
    agent: Any,
    case: dict[str, Any],
    max_turns: int = 30,
    timeout_seconds: int = 900,
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
        "max_turns": max_turns,
        "timeout_seconds": timeout_seconds,
    }

    try:
        max_attempts = 2
        transient_errors: list[dict[str, str]] = []
        result: Any | None = None
        for attempt in range(1, max_attempts + 1):
            trace_id = gen_trace_id()
            report["trace_id"] = trace_id
            try:
                with trace(
                    "DS Chat E2E smoke case",
                    trace_id=trace_id,
                    group_id=thread_id,
                    metadata={
                        "case": str(case.get("name") or ""),
                        "thread_id": thread_id,
                        "attempt": str(attempt),
                    },
                ):
                    result = await asyncio.wait_for(
                        Runner.run(
                            agent,
                            input=[{"role": "user", "content": question}],
                            context=context,
                            max_turns=max_turns,
                        ),
                        timeout=timeout_seconds,
                    )
                break
            except asyncio.TimeoutError:
                raise
            except Exception as exc:
                if attempt >= max_attempts or not _is_retryable_agent_error(exc):
                    raise
                transient_errors.append({
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })
                await asyncio.sleep(1)

        if result is None:
            raise RuntimeError("Agent returned no result.")

        if transient_errors:
            report["retry_count"] = len(transient_errors)
            report["transient_errors"] = transient_errors

        tool_calls = _extract_tool_calls(result)
        answer = str(getattr(result, "final_output", "") or "")

        report["tool_calls"] = tool_calls
        report["tool_call_count"] = len(tool_calls)
        report["answer"] = answer
        report["answer_length"] = len(answer)
        assertion_result = _check_assertions(case, tool_calls, answer)
        report["assertions"] = assertion_result
        assertion_failed = assertion_result.get("checked") and not assertion_result.get("passed", True)
        report["failed"] = bool(assertion_failed)
        if assertion_failed:
            report["failure_kind"] = "assertion"

    except asyncio.TimeoutError:
        report["failed"] = True
        report["failure_kind"] = "timeout"
        report["error"] = {
            "error_type": "CaseTimeout",
            "message": f"Case exceeded {timeout_seconds}s wall-clock timeout.",
        }
    except Exception as exc:
        report["failed"] = True
        error_type = type(exc).__name__
        report["failure_kind"] = "max_turns" if error_type == "MaxTurnsExceeded" else "error"
        report["error"] = {
            "error_type": error_type,
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
    """Render a human-readable markdown report with full output and tool traces."""
    lines: list[str] = [
        "# E2E Smoke Test Report",
        "",
        f"- Generated: {payload.get('generated_at')}",
        f"- Model: {payload.get('model')}",
        f"- Max turns: {payload.get('max_turns')}",
        f"- Case timeout: {payload.get('case_timeout_seconds')}s",
        f"- Cases: {len(payload.get('reports', []))}",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| # | Case | Status | Failure | Elapsed | Tools |")
    lines.append("|---|------|--------|---------|---------|-------|")
    for idx, report in enumerate(payload.get("reports", []), 1):
        name = report.get("name", "unknown")
        status = "FAIL" if report.get("failed") else "PASS"
        failure = report.get("failure_kind", "")
        elapsed = report.get("elapsed_seconds", "?")
        tc_count = report.get("tool_call_count", 0)
        lines.append(f"| {idx} | {name} | {status} | {failure} | {elapsed}s | {tc_count} |")
    lines.append("")

    # Detailed per-case sections
    for report in payload.get("reports", []):
        name = report.get("name", "unknown")
        failed = report.get("failed", False)
        status = "FAIL" if failed else "PASS"
        lines.append(f"## [{status}] {name}")
        lines.append("")
        lines.append(f"- **Question:** {report.get('question')}")
        lines.append(f"- **Elapsed:** {report.get('elapsed_seconds', '?')}s")
        lines.append(f"- **Tool calls:** {report.get('tool_call_count', 0)}")
        if report.get("failure_kind"):
            lines.append(f"- **Failure kind:** {report.get('failure_kind')}")
        if report.get("trace_id"):
            lines.append(f"- **Trace ID:** `{report.get('trace_id')}`")
        lines.append("")

        assertions = report.get("assertions", {})
        if assertions.get("checked"):
            lines.append(f"### Assertions: {'ALL PASSED' if assertions.get('passed') else 'SOME FAILED'}")
            lines.append("")
            for detail in assertions.get("details", []):
                mark = "PASS" if detail.get("passed") else "FAIL"
                payload = json.dumps({k: v for k, v in detail.items() if k not in {"assertion", "passed"}}, default=str)
                lines.append(f"- [{mark}] {detail.get('assertion')}: {payload}")
            lines.append("")

        if report.get("error"):
            lines.append(f"**Error:** `{report['error'].get('error_type')}: {report['error'].get('message')}`")
            tb = report["error"].get("traceback", "")
            if tb:
                lines.append("")
                lines.append("<details><summary>Traceback</summary>")
                lines.append("")
                lines.append("```")
                lines.append(tb.strip())
                lines.append("```")
                lines.append("</details>")
            lines.append("")

        # Tool call traces
        tool_calls = report.get("tool_calls", [])
        if tool_calls:
            tool_names = [tc.get("tool", "?") for tc in tool_calls]
            lines.append(f"### Tool Trace ({len(tool_calls)} calls)")
            lines.append("")
            lines.append(f"**Sequence:** {' -> '.join(tool_names)}")
            lines.append("")
            for i, tc in enumerate(tool_calls, 1):
                tool_name = tc.get("tool", "?")
                lines.append(f"#### {i}. `{tool_name}`")
                lines.append("")
                # Arguments
                args_str = tc.get("arguments", "")
                if args_str:
                    try:
                        args_obj = json.loads(args_str)
                        args_formatted = json.dumps(args_obj, indent=2)
                    except (json.JSONDecodeError, TypeError):
                        args_formatted = args_str
                    lines.append("**Input:**")
                    lines.append("```json")
                    lines.append(args_formatted)
                    lines.append("```")
                    lines.append("")
                # Output
                output = tc.get("output", "")
                if output:
                    # Truncate very long outputs for readability
                    display = output if len(output) <= 2000 else output[:2000] + f"\n... ({len(output)} chars total)"
                    lines.append("**Output:**")
                    lines.append("```")
                    lines.append(display)
                    lines.append("```")
                    lines.append("")
            lines.append("")

        # Full agent answer
        answer = report.get("answer", "")
        if answer:
            lines.append("### Final Agent Output")
            lines.append("")
            lines.append("```")
            lines.append(answer)
            lines.append("```")
            lines.append("")

        lines.append("---")
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

    if args.master:
        cases = [c for c in cases if c.get("master")]
        if args.model == "gpt-5-mini":  # override default only when --master is passed
            args.model = "gpt-5.2"
        args.max_turns = max(args.max_turns, 100)
    elif args.scenarios:
        selected = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        cases = [c for c in cases if str(c.get("name")) in selected]
    else:
        total_cases = len(cases)
        cases = [c for c in cases if not c.get("master")]
        skipped = total_cases - len(cases)
        if skipped:
            print(f"Skipping {skipped} master E2E case(s). Use --master or --scenarios to run them.")

    if not cases:
        print("No test cases selected.")
        return 1

    agent = build_investigation_agent(args.model)
    print(f"Agent: {agent.name}, tools: {len(agent.tools)}, model: {args.model}")
    print(
        f"Running {len(cases)} E2E test cases "
        f"(concurrency={args.concurrency}, max_turns={args.max_turns}, "
        f"case_timeout={args.case_timeout_seconds}s)...\n"
    )

    sem = asyncio.Semaphore(args.concurrency)
    total = len(cases)

    async def _run_with_sem(idx: int, case: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            name = case.get("name", f"case_{idx}")
            print(f"[{idx}/{total}] {name} starting ...", flush=True)
            report = await run_case(
                agent,
                case,
                max_turns=args.max_turns,
                timeout_seconds=args.case_timeout_seconds,
            )
            status = "FAIL" if report.get("failed") else "PASS"
            failure_kind = f" ({report.get('failure_kind')})" if report.get("failure_kind") else ""
            elapsed = report.get("elapsed_seconds", "?")
            tc_count = report.get("tool_call_count", 0)
            print(f"[{idx}/{total}] {name} -> [{status}]{failure_kind} {elapsed}s, {tc_count} tool calls", flush=True)
            return report

    tasks = [_run_with_sem(idx, case) for idx, case in enumerate(cases, 1)]
    reports: list[dict[str, Any]] = list(await asyncio.gather(*tasks))

    # Write reports
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "model": args.model,
        "max_turns": args.max_turns,
        "case_timeout_seconds": args.case_timeout_seconds,
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
    parser.add_argument("--model", default="gpt-5.4-mini", help="Model to use for the agent (default: gpt-5.4-mini)")
    parser.add_argument("--max-turns", type=int, default=100, help="Max agentic turns per case (default: 100)")
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=900,
        help="Wall-clock timeout per case in seconds (default: 900)",
    )
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
    parser.add_argument("--concurrency", type=int, default=5, help="Max parallel cases (default: 5)")
    parser.add_argument("--master", action="store_true", help="Run only master-tagged long-running cases with gpt-5.2 and 100 max turns")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip AWS credential bootstrap")
    args = parser.parse_args()

    return asyncio.run(run_all(args))


if __name__ == "__main__":
    raise SystemExit(main())
