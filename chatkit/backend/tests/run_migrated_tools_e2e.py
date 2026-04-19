#!/usr/bin/env python3
"""Focused E2E smoke tests for the migrated tool surface.

Exercises the three changes with real Runner.run() calls against the
shipping agent:

  1. hosted `apply_patch` (OpenAI Agents SDK) — multi-hunk edits
  2. `edit_file` with mode='insert'
  3. `fetch_url` with offset pagination

Each case uses a throwaway scratch directory under /tmp so the agent
can't touch the real tree. A test passes when both (a) the right tool
was called by the model, and (b) the observable side-effect is correct
(e.g. the file on disk matches the expected content after apply_patch).

Usage:
    cd chatkit/backend
    .venv/bin/python tests/run_migrated_tools_e2e.py [--model gpt-5-mini]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from agents import Runner

from app.agents.ds_agent import build_agent, _model_supports_apply_patch

# Re-use context stubs from the main smoke runner
from tests.run_e2e_smoke import (  # type: ignore[import-not-found]
    _CliAgentContext,
    _extract_tool_calls,
)


# ── Scenarios ─────────────────────────────────────────────────────────


def _scratch(name: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"e2e-{name}-"))
    return root


async def _run_case(
    agent: Any,
    *,
    name: str,
    question: str,
    expected_tool: str,
    assert_fn,  # (tool_calls, answer, scratch_dir) -> list[str] of failures
    scratch_dir: Path,
    model: str,
) -> dict[str, Any]:
    thread_id = f"migrated-{name}-{uuid.uuid4().hex[:6]}"
    ctx = _CliAgentContext(thread_id=thread_id)
    started = time.time()
    error: str | None = None
    answer = ""
    tool_calls: list[dict[str, str]] = []

    try:
        result = await Runner.run(
            agent,
            input=[{"role": "user", "content": question}],
            context=ctx,
            max_turns=20,
        )
        answer = str(getattr(result, "final_output", "") or "").strip()
        tool_calls = _extract_tool_calls(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = round(time.time() - started, 1)
    failures: list[str] = []
    if error:
        failures.append(f"runner_error: {error}")
    else:
        names_called = [c["name"] for c in tool_calls]
        if expected_tool not in names_called:
            failures.append(
                f"expected_tool '{expected_tool}' not called; got {names_called}"
            )
        failures.extend(assert_fn(tool_calls, answer, scratch_dir))

    return {
        "name": name,
        "model": model,
        "passed": not failures,
        "failures": failures,
        "tool_calls": tool_calls,
        "answer": answer[:2000],
        "elapsed_s": elapsed,
        "scratch": str(scratch_dir),
    }


# ── Case 1: apply_patch multi-hunk edit ───────────────────────────────


async def case_apply_patch(agent: Any, model: str) -> dict[str, Any]:
    scratch = _scratch("apply-patch")
    target = scratch / "math.py"
    target.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    question = (
        f"I have a Python file at `{target}`. Please use the apply_patch tool "
        f"to rename the `add` function to `sum_two` AND the `multiply` function "
        f"to `product` in a single multi-hunk patch. Do not use edit_file. "
        f"After applying, read the file back to confirm both changes landed."
    )

    def _assert(tool_calls, answer, sdir):
        failures: list[str] = []
        body = target.read_text(encoding="utf-8")
        if "def sum_two(" not in body:
            failures.append("file still contains old `def add(` — patch not applied")
        if "def product(" not in body:
            failures.append("file still contains old `def multiply(` — patch not applied")
        return failures

    return await _run_case(
        agent, name="apply_patch_multi_hunk",
        question=question, expected_tool="apply_patch",
        assert_fn=_assert, scratch_dir=scratch, model=model,
    )


# ── Case 2: edit_file mode='insert' ───────────────────────────────────


async def case_edit_insert(agent: Any, model: str) -> dict[str, Any]:
    scratch = _scratch("edit-insert")
    target = scratch / "config.ini"
    target.write_text("[server]\nhost = localhost\nport = 8000\n", encoding="utf-8")
    question = (
        f"I have a config file at `{target}`. Please use the edit_file tool "
        f"with mode='insert' to insert the line `debug = true` right after "
        f"the `port = 8000` line. Do not rewrite the whole file. Confirm "
        f"with read_file afterwards."
    )

    def _assert(tool_calls, answer, sdir):
        failures: list[str] = []
        body = target.read_text(encoding="utf-8")
        if "debug = true" not in body:
            failures.append("inserted line not present in file")
        # Check ordering: port = 8000 must come before debug = true
        if body.index("port = 8000") > body.index("debug = true"):
            failures.append("inserted line is in the wrong position")
        # Check edit_file was invoked with mode='insert'
        used_insert = any(
            c["name"] == "edit_file" and "insert" in c.get("arguments", "")
            for c in tool_calls
        )
        if not used_insert:
            failures.append("edit_file was called without mode='insert' — fell back to str_replace")
        return failures

    return await _run_case(
        agent, name="edit_file_insert_mode",
        question=question, expected_tool="edit_file",
        assert_fn=_assert, scratch_dir=scratch, model=model,
    )


# ── Case 3: fetch_url pagination ──────────────────────────────────────


async def case_fetch_url_pagination(agent: Any, model: str) -> dict[str, Any]:
    scratch = _scratch("fetch-url")
    question = (
        "Please use fetch_url with max_chars=2000 to read "
        "`https://raw.githubusercontent.com/python/cpython/main/README.rst`. "
        "After the first call, if the response indicates there's more content, "
        "call fetch_url AGAIN with an offset that resumes reading. Then "
        "summarize what you learned about the project."
    )

    def _assert(tool_calls, answer, sdir):
        failures: list[str] = []
        fetch_calls = [c for c in tool_calls if c["name"] == "fetch_url"]
        if len(fetch_calls) < 2:
            failures.append(
                f"expected ≥2 fetch_url calls (first + paginated follow-up); "
                f"got {len(fetch_calls)}"
            )
        # Second call should carry a non-zero offset
        if len(fetch_calls) >= 2:
            second_args = fetch_calls[1].get("arguments") or ""
            if '"offset"' not in second_args or '"offset":0' in second_args:
                failures.append(
                    f"second fetch_url didn't use a non-zero offset: {second_args}"
                )
        return failures

    return await _run_case(
        agent, name="fetch_url_pagination",
        question=question, expected_tool="fetch_url",
        assert_fn=_assert, scratch_dir=scratch, model=model,
    )


# ── Runner + report ───────────────────────────────────────────────────


async def _run_all(model: str) -> list[dict[str, Any]]:
    agent = build_agent(model)
    apply_patch_supported = _model_supports_apply_patch(model)
    results = []
    for fn in (case_apply_patch, case_edit_insert, case_fetch_url_pagination):
        print(f"\n=== {fn.__name__} ===", flush=True)
        if fn is case_apply_patch and not apply_patch_supported:
            results.append({
                "name": "apply_patch_multi_hunk",
                "model": model,
                "passed": True,
                "skipped": True,
                "failures": [],
                "tool_calls": [],
                "answer": "",
                "elapsed_s": 0.0,
                "scratch": "",
            })
            print(f"  SKIP — {model} doesn't support hosted apply_patch", flush=True)
            continue
        r = await fn(agent, model)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status} in {r['elapsed_s']}s — tools: "
              f"{[c['name'] for c in r['tool_calls']]}", flush=True)
        if not r["passed"]:
            for f in r["failures"]:
                print(f"    - {f}", flush=True)
    return results


def _print_report(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("MIGRATED-TOOLS E2E SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped"))
    print(f"Passed: {passed}/{total}  Skipped: {skipped}")
    for r in results:
        if r.get("skipped"):
            mark = "SKIP"
        elif r["passed"]:
            mark = "PASS"
        else:
            mark = "FAIL"
        print(f"  [{mark}] {r['name']} ({r['elapsed_s']}s)")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--json", type=Path, default=None, help="write full results to JSON")
    args = p.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; skipping.")
        return 0

    results = asyncio.run(_run_all(args.model))
    _print_report(results)

    if args.json:
        args.json.write_text(
            json.dumps(results, ensure_ascii=True, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nFull results → {args.json}")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
