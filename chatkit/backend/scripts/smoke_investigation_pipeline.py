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


def _normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: Any, max_len: int = 220) -> str:
    text = _normalize_whitespace(value)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}…"


def _parse_activity_log(activity_log: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in (activity_log or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _event_signature(event: str, payload: dict[str, Any]) -> str:
    if event == "plan_start":
        return "plan_start"
    if event == "extract_sql":
        return f"extract_sql[{payload.get('dataset_id')}:{payload.get('row_count')}]"
    if event == "extract_s3":
        return f"extract_s3[{payload.get('dataset_id')}:{payload.get('row_count')}]"
    if event == "analysis_complete":
        return f"analysis_complete[{payload.get('analysis_id')}]"
    if event == "operator_run_python_start":
        return "operator_run_python_start"
    if event == "operator_run_python_done":
        return "operator_run_python_done"
    if event == "action_observation":
        action = str(payload.get("action") or "unknown")
        if action in {"extract_sql", "extract_s3"}:
            return f"action:{action}[{payload.get('dataset_id')}:{payload.get('row_count')}]"
        return f"action:{action}"
    if event == "investigation_complete":
        return f"investigation_complete[{payload.get('strategy')}]"
    return event


def _event_details(index: int, event: str, payload: dict[str, Any]) -> list[str]:
    if event == "plan_start":
        return [
            f"{index}. `plan_start(question=..., sales_date={payload.get('sales_date')})`",
            f"   - question: {_truncate(payload.get('question'), max_len=300)}",
        ]
    if event == "extract_sql":
        return [
            (
                f"{index}. `extract_sql(datasource={payload.get('datasource')}, "
                f"dataset_id={payload.get('dataset_id')}, row_count={payload.get('row_count')}, "
                f"elapsed_ms={payload.get('elapsed_ms')})`"
            ),
            f"   - query: `{_truncate(payload.get('query'), max_len=400)}`",
        ]
    if event == "extract_s3":
        lines = [
            (
                f"{index}. `extract_s3(dataset_id={payload.get('dataset_id')}, "
                f"row_count={payload.get('row_count')}, elapsed_ms={payload.get('elapsed_ms')})`"
            )
        ]
        s3_uri = payload.get("s3_uri")
        if s3_uri:
            lines.append(f"   - s3_uri: `{s3_uri}`")
        return lines
    if event == "analysis_complete":
        return [
            (
                f"{index}. `analysis_complete(analysis_id={payload.get('analysis_id')}, "
                f"analysis_mode={payload.get('analysis_mode')}, dataset_ids={payload.get('dataset_ids')})`"
            )
        ]
    if event == "operator_run_python_start":
        return [
            (
                f"{index}. `operator_run_python_start(approval_policy={payload.get('approval_policy')}, "
                f"sandbox_mode={payload.get('sandbox_mode')})`"
            )
        ]
    if event == "operator_run_python_done":
        return [
            (
                f"{index}. `operator_run_python_done(created_datasets={payload.get('created_datasets')}, "
                f"created_analyses={payload.get('created_analyses')})`"
            )
        ]
    if event == "action_observation":
        action = str(payload.get("action") or "unknown")
        lines = [
            f"{index}. `action_observation(action={action}, ok={payload.get('ok')}, step={payload.get('step')})`"
        ]
        if action == "resolve_entities":
            entities = payload.get("entities") or {}
            lines.append(
                (
                    "   - entities: "
                    f"providers={entities.get('providers')}, "
                    f"sites={entities.get('sites')}, "
                    f"customers={entities.get('customers')}"
                )
            )
        elif action == "retrieve_knowledge":
            lines.append(f"   - candidate_tables={len(payload.get('candidate_tables') or [])}")
            lines.append(f"   - task_cards={payload.get('task_cards')}")
        elif action == "inspect_table_metadata":
            lines.append(
                (
                    "   - metadata: "
                    f"table={payload.get('table_name')}, "
                    f"columns={payload.get('columns')}, "
                    f"partitions={payload.get('partitions')}"
                )
            )
        elif action in {"extract_sql", "extract_s3"}:
            lines.append(
                f"   - dataset_id={payload.get('dataset_id')}, row_count={payload.get('row_count')}"
            )
        elif action == "run_analysis":
            lines.append(f"   - analysis_id={payload.get('analysis_id')}")
        elif action == "run_python":
            lines.append(
                (
                    "   - run_python: "
                    f"created_datasets={payload.get('created_datasets')}, "
                    f"stdout={_truncate(payload.get('stdout'), max_len=200)!r}"
                )
            )
        return lines
    if event == "investigation_complete":
        return [
            (
                f"{index}. `investigation_complete(strategy={payload.get('strategy')}, "
                f"dataset_count={payload.get('dataset_count')}, error_count={payload.get('error_count')})`"
            )
        ]
    return [f"{index}. `{event}({_truncate(payload, max_len=240)})`"]


def _render_readable_report(payload: dict[str, Any], source_json_path: Path) -> str:
    lines: list[str] = [
        "# Smoke Test Function Calls and Outputs",
        "",
        f"- Source report: `{source_json_path}`",
        f"- Generated at: `{payload.get('generated_at')}`",
        "",
    ]
    for report in payload.get("reports", []):
        if not isinstance(report, dict):
            continue
        scenario = str(report.get("scenario") or "unknown_scenario")
        result = report.get("result") if isinstance(report.get("result"), dict) else {}

        lines.extend(
            [
                f"## {scenario}",
                "",
                f"- Thread: `{report.get('thread_id')}`",
                f"- Run ID: `{result.get('run_id')}`",
                f"- Question: {report.get('question')}",
                "",
            ]
        )

        events = _parse_activity_log(str(report.get("activity_log") or ""))
        if events:
            sequence = []
            for event in events:
                event_name = str(event.get("event") or "")
                event_payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                sequence.append(_event_signature(event_name, event_payload))
            lines.extend(
                [
                    "### Call Sequence",
                    "",
                    f"`{' -> '.join(sequence)}`",
                    "",
                    "### Calls (ordered)",
                    "",
                ]
            )
            for index, event in enumerate(events, start=1):
                event_name = str(event.get("event") or "")
                event_payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                lines.extend(_event_details(index, event_name, event_payload))
        else:
            lines.extend(
                [
                    "### Call Sequence",
                    "",
                    "`<no activity log events captured>`",
                    "",
                ]
            )

        answer = str(result.get("answer") or "").strip()
        if "\\n" in answer and "\n" not in answer:
            answer = answer.replace("\\n", "\n")
        lines.extend(
            [
                "",
                "### Output",
                "",
                "```markdown",
                answer or "<empty>",
                "```",
                "",
            ]
        )
        if isinstance(report.get("error"), dict):
            lines.extend(
                [
                    "### Runtime Error",
                    "",
                    "```text",
                    str(report.get("error")),
                    "```",
                    "",
                ]
            )

    return "\n".join(lines)


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
    markdown_path = report_dir / f"smoke_investigation_{run_stamp}.md"

    json_text = json.dumps(payload, indent=2, default=str)
    json_path.write_text(json_text, encoding="utf-8")
    log_path.write_text(json_text, encoding="utf-8")
    markdown_text = _render_readable_report(payload=payload, source_json_path=json_path)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    print(json.dumps({
        "report_json": str(json_path),
        "report_log": str(log_path),
        "report_readable_markdown": str(markdown_path),
        "failed_scenarios": [item.get("scenario") for item in reports if item.get("failed")],
    }, indent=2))

    return 1 if any(item.get("failed") for item in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
