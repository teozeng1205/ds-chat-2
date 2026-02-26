#!/usr/bin/env python3
"""Run InvestigationRuntime E2E prompts from a JSON test-case file."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.investigation.runtime import InvestigationRuntime, cleanup_thread_workspace


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
    else:
        cases = payload
    if not isinstance(cases, list):
        raise ValueError("Cases file must contain a list or an object with a 'cases' list.")
    return [case for case in cases if isinstance(case, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run investigation E2E prompts from file.")
    parser.add_argument(
        "--cases-file",
        default=str((Path(__file__).resolve().parents[1] / "tests" / "e2e_investigation_cases.json").resolve()),
        help="JSON file with investigation prompts.",
    )
    parser.add_argument(
        "--cleanup-mode",
        default="ephemeral_manifest",
        help="Workspace cleanup mode after each case.",
    )
    args = parser.parse_args()

    cases_file = Path(args.cases_file).expanduser().resolve()
    runtime = InvestigationRuntime()
    cases = _load_cases(cases_file)

    reports: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        name = str(case.get("name") or f"case_{idx}")
        thread_id = str(case.get("thread_id") or f"thread-{name}")
        question = str(case.get("question") or "")
        sales_date = case.get("sales_date")
        constraints = case.get("constraints")

        started_at = datetime.now(timezone.utc).isoformat()
        report: dict[str, Any] = {
            "index": idx,
            "name": name,
            "thread_id": thread_id,
            "question": question,
            "started_at": started_at,
        }
        try:
            report["result"] = runtime.investigate_issue(
                thread_id=thread_id,
                question=question,
                sales_date=sales_date,
                constraints=constraints,
            )
        except Exception as exc:  # noqa: BLE001
            report["error"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            try:
                report["cleanup"] = cleanup_thread_workspace(thread_id=thread_id, mode=args.cleanup_mode)
            except Exception as cleanup_exc:  # noqa: BLE001
                report["cleanup_error"] = {
                    "error_type": type(cleanup_exc).__name__,
                    "message": str(cleanup_exc),
                }
        report["ended_at"] = datetime.now(timezone.utc).isoformat()
        reports.append(report)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_file": str(cases_file),
        "cleanup_mode": args.cleanup_mode,
        "reports": reports,
    }
    print(json.dumps(payload, indent=2, default=str))

    has_errors = any("error" in report for report in reports)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
