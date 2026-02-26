#!/usr/bin/env python3
"""Run InvestigationRuntime E2E prompts from a JSON case file with full activity logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.investigation.runtime import InvestigationRuntime, cleanup_thread_workspace


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run investigation E2E prompts from file.")
    parser.add_argument("--profile", default="3VDEV", help="Credential profile for `assume` (default: 3VDEV)")
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
    parser.add_argument(
        "--report-dir",
        default=str((Path(__file__).resolve().parents[1] / ".runtime" / "e2e_reports").resolve()),
        help="Directory for full report payload.",
    )
    args = parser.parse_args()

    bootstrap = _bootstrap_aws_credentials(args.profile)

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
            result = runtime.investigate_issue(
                thread_id=thread_id,
                question=question,
                sales_date=sales_date,
                constraints=constraints,
            )
            report["result"] = result
            run_id = str(result.get("run_id"))
            activity_log = Path("/tmp") / "ds-chat-investigation" / thread_id / run_id / "activity.jsonl"
            report["activity_log_path"] = str(activity_log)
            report["activity_log"] = activity_log.read_text(encoding="utf-8", errors="replace") if activity_log.exists() else ""
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
        "credential_bootstrap": bootstrap,
        "reports": reports,
    }

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"investigation_e2e_{stamp}.json"
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(json.dumps({"report": str(report_path), "cases": len(reports)}, indent=2))

    has_errors = any("error" in report for report in reports)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
