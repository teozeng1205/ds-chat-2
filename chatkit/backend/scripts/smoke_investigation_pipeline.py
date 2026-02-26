#!/usr/bin/env python3
"""Runtime-level smoke test for DS Chat autonomous investigation pipeline."""

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
    rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Cases file must be a list or an object with a 'cases' list")
    return [item for item in rows if isinstance(item, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run investigation pipeline smoke scenarios.")
    parser.add_argument("--profile", default="3VDEV", help="Credential profile for `assume` (default: 3VDEV)")
    parser.add_argument("--model", default="runtime", help="Compatibility arg; retained for verify script")
    parser.add_argument("--max-turns", type=int, default=40, help="Compatibility arg; retained for verify script")
    parser.add_argument(
        "--cases-file",
        default=str((Path(__file__).resolve().parents[1] / "tests" / "e2e_investigation_cases.json").resolve()),
        help="Path to JSON test-case file",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Optional comma-separated scenario names from case file",
    )
    parser.add_argument(
        "--report-dir",
        default=str((Path(__file__).resolve().parents[1] / ".runtime" / "smoke_reports").resolve()),
        help="Directory for full JSON and markdown reports",
    )
    args = parser.parse_args()

    cred_info = _bootstrap_aws_credentials(args.profile)

    cases_path = Path(args.cases_file).expanduser().resolve()
    cases = _load_cases(cases_path)
    if args.scenarios.strip():
        selected = {token.strip() for token in args.scenarios.split(",") if token.strip()}
        cases = [case for case in cases if str(case.get("name")) in selected]
    if not cases:
        raise SystemExit("No cases selected.")

    runtime = InvestigationRuntime()
    reports: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        name = str(case.get("name") or f"case_{idx}")
        thread_id = str(case.get("thread_id") or f"thread_{name}")
        question = str(case.get("question") or "")
        sales_date = case.get("sales_date")
        constraints = case.get("constraints")

        started = datetime.now(timezone.utc).isoformat()
        row: dict[str, Any] = {
            "index": idx,
            "scenario": name,
            "thread_id": thread_id,
            "question": question,
            "started_at": started,
        }

        try:
            result = runtime.investigate_issue(
                thread_id=thread_id,
                question=question,
                sales_date=sales_date,
                constraints=constraints,
            )
            run_id = str(result.get("run_id"))
            log_path = Path("/tmp") / "ds-chat-investigation" / thread_id / run_id / "activity.jsonl"
            row["result"] = result
            row["activity_log_path"] = str(log_path)
            if log_path.exists():
                row["activity_log"] = log_path.read_text(encoding="utf-8", errors="replace")
            row["failed"] = bool(result.get("errors"))
        except Exception as exc:  # noqa: BLE001
            row["failed"] = True
            row["error"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            row["cleanup"] = cleanup_thread_workspace(thread_id=thread_id, mode="ephemeral_manifest")
            row["ended_at"] = datetime.now(timezone.utc).isoformat()

        reports.append(row)

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "profile": args.profile,
        "credential_bootstrap": cred_info,
        "model": args.model,
        "max_turns": args.max_turns,
        "cases_file": str(cases_path),
        "reports": reports,
    }

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"smoke_investigation_{run_stamp}.json"
    log_path = report_dir / f"smoke_investigation_{run_stamp}.log"

    json_text = json.dumps(payload, indent=2, default=str)
    json_path.write_text(json_text, encoding="utf-8")
    log_path.write_text(json_text, encoding="utf-8")

    print(json.dumps({
        "report_json": str(json_path),
        "report_log": str(log_path),
        "failed_scenarios": [item.get("scenario") for item in reports if item.get("failed")],
    }, indent=2))

    return 1 if any(item.get("failed") for item in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
