#!/usr/bin/env python3
"""E2E smoke test runner for the DS Chat investigation agent.

Runs all cases in e2e_investigation_cases.json, checks assertions,
and writes two output files:
  - e2e_smoke_<timestamp>.log   (full JSONL trace)
  - e2e_smoke_<timestamp>.md    (human-readable markdown report)

Usage:
    cd chatkit/backend
    .venv/bin/python tests/run_e2e_smoke.py [--model gpt-5-mini] [--include-master]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import subprocess as _subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from agents import Runner  # type: ignore[import]

from app.agents.ds_agent import build_agent
from app.investigation.runtime import cleanup_thread_workspace
from app.tracing import install_sqlite_tracing

# Install tracing for smoke runs too, so we can inspect what the agent
# actually did afterwards. Idempotent — safe to call multiple times.
try:
    install_sqlite_tracing()
except Exception as _exc:  # noqa: BLE001
    print(f"[warn] tracing install failed: {_exc}")


# ── CLI context stubs ──

class _CliThread:
    def __init__(self, thread_id: str) -> None:
        self.id = thread_id

class _CliStore:
    async def save_attachment(self, attachment: Any, **kwargs: Any) -> None:
        pass

class _CliAgentContext:
    def __init__(self, thread_id: str) -> None:
        self.thread = _CliThread(thread_id)
        self.store = _CliStore()
        self.request_context: dict[str, Any] = {}
        self.progress_events: list[str] = []

    async def stream(self, event: Any) -> None:
        icon = str(getattr(event, "icon", "")).strip()
        text = str(getattr(event, "text", "")).strip()
        if text:
            self.progress_events.append(f"{icon} {text}".strip())

    async def stream_widget(self, widget: Any, copy_text: str | None = None, **kwargs: Any) -> None:
        pass


# ── Tool call extraction ──

def _compact_json(text: str, max_len: int = 300) -> str:
    raw = text.strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
        rendered = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        rendered = raw
    return rendered[:max_len - 3] + "..." if len(rendered) > max_len else rendered


def _raw_get(raw: Any, key: str, default: str = "") -> str:
    """Get a field from raw_item whether it's a dict or an object."""
    if raw is None:
        return default
    if isinstance(raw, dict):
        return str(raw.get(key) or default)
    return str(getattr(raw, key, None) or default)


def _extract_tool_calls(result: Any) -> list[dict[str, str]]:
    calls: dict[str, dict[str, str]] = {}
    ordered_ids: list[str] = []
    counter = 0
    for item in getattr(result, "new_items", []):
        item_type = getattr(item, "type", "")
        raw = getattr(item, "raw_item", None)
        if item_type == "tool_call_item":
            # Responses API: raw may be a dict with "id"/"call_id"/"name"/"arguments"
            call_id = _raw_get(raw, "call_id") or _raw_get(raw, "id") or f"call_{counter}"
            counter += 1
            name = _raw_get(raw, "name") or "unknown_tool"
            args = _raw_get(raw, "arguments")
            calls[call_id] = {"name": name, "arguments": _compact_json(args), "output": ""}
            ordered_ids.append(call_id)
        elif item_type == "tool_call_output_item":
            # Responses API: call_id may be on raw dict or on item directly
            call_id = (
                _raw_get(raw, "call_id")
                or str(getattr(item, "call_id", "") or "")
            )
            # output may be on item.output (str) or raw["output"]
            output = (
                str(getattr(item, "output", None) or "")
                or _raw_get(raw, "output")
            )
            if call_id and call_id in calls:
                calls[call_id]["output"] = _compact_json(output)
            elif output:  # unmatched output — show as unknown_tool
                uid = f"out_{counter}"
                calls[uid] = {"name": "unknown_tool", "arguments": "", "output": _compact_json(output)}
                ordered_ids.append(uid)
    return [calls[cid] for cid in ordered_ids]


# ── Assertion checking ──

def _check_assertions(
    assertions: dict[str, Any],
    tool_calls: list[dict[str, str]],
    answer: str,
) -> list[str]:
    failures: list[str] = []
    tool_names = [c["name"] for c in tool_calls]

    min_calls = assertions.get("min_tool_calls", 0)
    if len(tool_calls) < min_calls:
        failures.append(f"min_tool_calls: expected ≥{min_calls}, got {len(tool_calls)}")

    for required in assertions.get("required_tools", []):
        if required not in tool_names:
            failures.append(f"required_tool '{required}' not called (called: {tool_names})")

    answer_lower = answer.lower()
    for phrase in assertions.get("answer_contains", []):
        if phrase.lower() not in answer_lower:
            failures.append(f"answer_contains: '{phrase}' not found in answer")

    return failures


# ── Pre-flight checks ──

def _check_aws_creds() -> bool:
    """Return True if any AWS credentials are valid (STS or S3 fallback)."""
    try:
        r = _subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True, timeout=15
        )
        if r.returncode == 0:
            return True
    except Exception:
        pass
    # Fallback: try a lightweight S3 access (cross-account bucket policy, no STS needed)
    try:
        r = _subprocess.run(
            ["aws", "s3", "ls", "s3://s3-atp-3victors-3vprod-use1-anomaly-datasets/", "--page-size", "1"],
            capture_output=True, timeout=15
        )
        return r.returncode == 0
    except Exception:
        return False


# ── Main runner ──

async def run_case(
    case: dict[str, Any],
    agent: Any,
    model: str,
    aws_creds_ok: bool = True,
) -> dict[str, Any]:
    thread_id = case.get("thread_id") or f"smoke-{uuid.uuid4().hex[:10]}"

    # Skip cred-dependent cases when AWS credentials are unavailable
    if case.get("requires_aws_creds") and not aws_creds_ok:
        return {
            "name": case["name"],
            "master": case.get("master", False),
            "question": case["question"],
            "model": model,
            "thread_id": thread_id,
            "elapsed_s": 0.0,
            "passed": False,
            "skipped": True,
            "skip_reason": "aws_creds_unavailable",
            "error": None,
            "failures": [],
            "tool_calls": [],
            "progress_events": [],
            "answer": "",
        }

    context = _CliAgentContext(thread_id=thread_id)
    question = case["question"]
    assertions = case.get("assertions", {})

    started = time.time()
    error: str | None = None
    tool_calls: list[dict[str, str]] = []
    answer = ""

    try:
        result = await Runner.run(
            agent,
            input=[{"role": "user", "content": question}],
            context=context,
            max_turns=60,
        )
        answer = str(getattr(result, "final_output", "") or "").strip()
        tool_calls = _extract_tool_calls(result)
    except Exception as exc:
        error = str(exc)

    elapsed = round(time.time() - started, 1)
    failures = [] if error else _check_assertions(assertions, tool_calls, answer)
    passed = error is None and len(failures) == 0

    return {
        "name": case["name"],
        "master": case.get("master", False),
        "question": question,
        "model": model,
        "thread_id": thread_id,
        "elapsed_s": elapsed,
        "passed": passed,
        "skipped": False,
        "skip_reason": None,
        "error": error,
        "failures": failures,
        "tool_calls": tool_calls,
        "progress_events": context.progress_events,
        "answer": answer,
    }


# ── Report writers ──

def write_jsonl_log(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=True, default=str) + "\n")


def write_markdown_report(results: list[dict[str, Any]], path: Path, model: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped"))
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed - skipped

    lines: list[str] = [
        f"# E2E Smoke Test Report",
        f"",
        f"**Date:** {now}  ",
        f"**Model:** `{model}`  ",
        f"**Results:** {passed}/{total} passed, {skipped} skipped (no AWS creds)  ",
        f"",
        f"---",
        f"",
    ]

    # Summary table
    lines += [
        "## Summary",
        "",
        "| # | Case | Status | Tools Called | Time |",
        "|---|------|--------|-------------|------|",
    ]
    for i, r in enumerate(results, 1):
        if r.get("skipped"):
            status = "⏭ SKIP"
        elif r["passed"]:
            status = "✅ PASS"
        elif r["error"]:
            status = "💥 ERROR"
        else:
            status = "❌ FAIL"
        tools = ", ".join(c["name"] for c in r["tool_calls"]) or "—"
        if len(tools) > 60:
            tools = tools[:57] + "..."
        master_tag = " ⭐" if r["master"] else ""
        lines.append(f"| {i} | `{r['name']}`{master_tag} | {status} | {tools} | {r['elapsed_s']}s |")

    lines += ["", "---", ""]

    # Per-case detail
    lines += ["## Case Details", ""]
    for i, r in enumerate(results, 1):
        if r.get("skipped"):
            status = "⏭ SKIP"
        elif r["passed"]:
            status = "✅ PASS"
        elif r["error"]:
            status = "💥 ERROR"
        else:
            status = "❌ FAIL"
        master_tag = " ⭐ master" if r["master"] else ""
        lines += [
            f"### {i}. `{r['name']}`{master_tag} — {status}",
            f"",
            f"**Question:** {r['question']}",
            f"",
            f"**Thread:** `{r['thread_id']}` | **Model:** `{r['model']}` | **Time:** {r['elapsed_s']}s",
            f"",
        ]

        if r["error"]:
            lines += [f"**Error:** `{r['error']}`", ""]

        if r["failures"]:
            lines += ["**Assertion failures:**", ""]
            for f_ in r["failures"]:
                lines.append(f"- {f_}")
            lines.append("")

        if r["tool_calls"]:
            lines += ["**Tool call trace:**", ""]
            for j, tc in enumerate(r["tool_calls"], 1):
                lines.append(f"**{j}. `{tc['name']}`**")
                if tc["arguments"]:
                    lines += [f"```json", tc["arguments"], "```"]
                if tc["output"]:
                    lines += [f"```json", tc["output"], "```"]
            lines.append("")

        if r["answer"]:
            # Truncate very long answers
            answer_display = r["answer"] if len(r["answer"]) <= 2000 else r["answer"][:1997] + "..."
            lines += [
                "**Agent answer:**",
                "",
                answer_display,
                "",
            ]

        lines += ["---", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ──

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E smoke test runner")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--include-master", action="store_true", help="Also run master (long) cases")
    parser.add_argument("--case", default=None, help="Run only a specific case by name")
    parser.add_argument("--out-dir", default="tests/smoke_reports", help="Output directory for reports")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent cases (default: 5)")
    return parser.parse_args()


async def main() -> int:
    args = _parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return 1

    cases_path = BACKEND_ROOT / "tests" / "e2e_investigation_cases.json"
    cases: list[dict[str, Any]] = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]

    if args.case:
        cases = [c for c in cases if c["name"] == args.case]
        if not cases:
            print(f"No case named '{args.case}'")
            return 1
    elif not args.include_master:
        cases = [c for c in cases if not c.get("master")]

    out_dir = BACKEND_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"e2e_smoke_{ts}.log"
    md_path = out_dir / f"e2e_smoke_{ts}.md"

    aws_creds_ok = _check_aws_creds()
    if not aws_creds_ok:
        print("⚠️  AWS credentials unavailable — cred-dependent cases will be SKIPPED")

    agent = build_agent(args.model)

    print(f"Running {len(cases)} cases with model={args.model} (concurrency={args.concurrency})")
    print(f"Output: {log_path.name}, {md_path.name}")
    print()

    async def _run_with_progress(
        sem: asyncio.Semaphore, i: int, total: int, case: dict[str, Any], model: str
    ) -> tuple[int, dict[str, Any]]:
        async with sem:
            master_tag = " [master]" if case.get("master") else ""
            if case.get("requires_aws_creds") and not aws_creds_ok:
                print(f"[{i}/{total}] {case['name']}{master_tag} SKIP (no AWS creds)")
                result = await run_case(case, agent, model, aws_creds_ok=aws_creds_ok)
                return i, result
            print(f"[{i}/{total}] {case['name']}{master_tag} starting...", flush=True)
            result = await run_case(case, agent, model, aws_creds_ok=aws_creds_ok)
            status = "PASS" if result["passed"] else ("ERROR" if result["error"] else "FAIL")
            print(f"[{i}/{total}] {case['name']}{master_tag} {status} ({result['elapsed_s']}s, {len(result['tool_calls'])} tool calls)")
            if result["error"]:
                print(f"       [{case['name']}] error: {result['error']}")
            elif result["failures"]:
                for f_ in result["failures"]:
                    print(f"       [{case['name']}] fail: {f_}")
            return i, result

    sem = asyncio.Semaphore(args.concurrency)
    total = len(cases)
    tasks = [_run_with_progress(sem, i, total, case, args.model) for i, case in enumerate(cases, 1)]
    indexed = await asyncio.gather(*tasks)
    # Restore original order for the report
    results: list[dict[str, Any]] = [r for _, r in sorted(indexed, key=lambda x: x[0])]

    # Write outputs
    write_jsonl_log(results, log_path)
    write_markdown_report(results, md_path, args.model)

    passed = sum(1 for r in results if r["passed"])
    skipped = sum(1 for r in results if r.get("skipped"))
    total = len(results)
    failed = total - passed - skipped
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {skipped} skipped (no AWS creds)")
    print(f"Log:      {log_path}")
    print(f"Report:   {md_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
