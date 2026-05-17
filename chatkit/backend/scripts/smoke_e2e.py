#!/usr/bin/env python3
"""Unified E2E smoke test for the DS Chat agentic investigation pipeline.

Runs test cases through the actual agentic loop using Runner.run().

Usage:
    cd chatkit/backend
    eval "$(assume 3VDEV)"
    .venv/bin/python scripts/smoke_e2e.py --profile 3VDEV --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import ast
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

from agents import RunConfig, Runner, gen_trace_id, trace  # noqa: E402  # type: ignore[import]

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
    lowered = str(output or "").lower()
    if "error: command blocked by guardrails" in lowered:
        return "GuardrailBlocked"
    has_false_ok = "'ok': false" in lowered or '"ok": false' in lowered
    if re.search(r"\b(command timed out|timed out after|case exceeded)\b", lowered) or (
        has_false_ok and re.search(r"\b(timed out|timeout)\b", lowered)
    ):
        return "ToolTimeout"
    if not has_false_ok:
        return None
    for pattern in (
        r"'error_type': '([^']+)'",
        r'"error_type"\s*:\s*"([^"]+)"',
    ):
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    return "ToolError"


def _parse_tool_output(output: str) -> Any:
    text = str(output or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Tool outputs are often Python dict reprs. Pandas values such as
    # Timestamp('2026-03-07 13:01:00') are not valid literals, but the
    # surrounding dict still contains useful row_count/preview evidence.
    literalish = re.sub(r"\bTimestamp\((['\"][^'\"]*['\"])\)", r"\1", text)
    literalish = re.sub(r"\bNaT\b", "None", literalish)
    try:
        return ast.literal_eval(literalish)
    except Exception:
        return None


_INTERNAL_CASE_TERMS = (
    "priceeye",
    "3vdev",
    "3vprod",
    "atpco",
    "prod.",
    "redshift",
    "mysql",
    "s3-atp-",
    "search_kb",
    "knowledge base",
    "schema",
    "codebase",
)
_PUBLIC_WEB_TERMS = ("web", "internet", "online", "public", "external")


def _case_text(case: dict[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in (
            case.get("name"),
            case.get("question"),
            json.dumps(case.get("assertions") or {}, sort_keys=True),
        )
    ).lower()


def _is_internal_case(case: dict[str, Any]) -> bool:
    text = _case_text(case)
    return any(term in text for term in _INTERNAL_CASE_TERMS)


def _is_bounded_case(case: dict[str, Any]) -> bool:
    text = _case_text(case)
    return any(term in text for term in ("bounded", "smoke", "use search_kb", "do not run", "do not inspect", "exactly one"))


def _asks_for_public_web(case: dict[str, Any]) -> bool:
    text = " ".join(str(value or "") for value in (case.get("name"), case.get("question"))).lower()
    return "web_search" in text or any(term in text for term in _PUBLIC_WEB_TERMS)


def _cases_need_web_search(cases: list[dict[str, Any]]) -> bool:
    return any(_asks_for_public_web(case) for case in cases)


def _tool_outputs(tool_calls: list[dict[str, Any]], tool_name: str) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for tc in tool_calls:
        if str(tc.get("tool", "")) != tool_name:
            continue
        parsed = _parse_tool_output(str(tc.get("output", "")))
        if isinstance(parsed, dict):
            outputs.append(parsed)
    return outputs


def _source_reference_present(answer: str, search_outputs: list[dict[str, Any]]) -> bool:
    candidates: set[str] = set()

    def add_candidate(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        candidates.add(text)
        candidates.add(text.split("#", 1)[0])
        if ":" in text:
            candidates.add(text.split(":", 1)[1])

    for output in search_outputs:
        for citation in output.get("citations", []) or []:
            if isinstance(citation, dict):
                add_candidate(citation.get("source"))
        for bucket in ("items", "verified_items", "hints"):
            for item in output.get(bucket, []) or []:
                if isinstance(item, dict):
                    add_candidate(item.get("source_path"))
                    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                    for key in ("provenance", "git_path", "path", "config_file", "module_path", "template"):
                        add_candidate(metadata.get(key))
    answer_l = answer.lower()
    return any(candidate and candidate.lower() in answer_l for candidate in candidates)


def _has_structured_kb_evidence(search_outputs: list[dict[str, Any]]) -> bool:
    for output in search_outputs:
        for item in output.get("verified_items", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("source_type") in {"structured_snapshot", "code_verified", "live_verified"}:
                return True
        for table in output.get("tables", []) or []:
            if isinstance(table, dict) and table.get("source_type") == "structured_snapshot":
                return True
    return False


def _number_present(answer: str, value: int) -> bool:
    plain = str(value)
    with_commas = f"{value:,}"
    return (
        re.search(rf"(?<![\d,]){re.escape(plain)}(?![\d,])", answer) is not None
        or re.search(rf"(?<![\d,]){re.escape(with_commas)}(?![\d,])", answer) is not None
    )


def _value_present(answer: str, value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, int):
        return _number_present(answer, value)
    if isinstance(value, float):
        if value.is_integer():
            return _number_present(answer, int(value))
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return bool(text and text in answer)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return False
    return text in answer


def _empty_result_reflected(answer: str) -> bool:
    return re.search(
        r"\b(?:no rows?|zero rows?|0 rows?|empty result|empty dataset|no data|nothing returned)\b",
        answer,
        flags=re.IGNORECASE,
    ) is not None


def _preview_values(output: dict[str, Any], limit: int = 30) -> list[Any]:
    preview = output.get("preview")
    if not isinstance(preview, list):
        return []
    values: list[Any] = []
    for row in preview[:5]:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and abs(float(value)) < 10:
                continue
            if isinstance(value, str) and len(value.strip()) < 2:
                continue
            values.append(value)
            if len(values) >= limit:
                return values
    return values


def _data_result_reflected(answer: str, tool_calls: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    data_tools = {"execute_sql", "fetch_s3", "list_s3"}
    outputs: list[tuple[str, dict[str, Any]]] = []
    for tc in tool_calls:
        tool = str(tc.get("tool", ""))
        if tool not in data_tools:
            continue
        parsed = _parse_tool_output(str(tc.get("output", "")))
        if isinstance(parsed, dict):
            outputs.append((tool, parsed))

    if not outputs:
        return False, {"error": "No parseable SQL/S3 output found."}

    for tool, output in outputs:
        row_count = output.get("row_count")
        if isinstance(row_count, int):
            if row_count == 0 and _empty_result_reflected(answer):
                return True, {"matched": "empty_result", "tool": tool, "row_count": row_count}
            if _number_present(answer, row_count):
                return True, {"matched": "row_count", "tool": tool, "row_count": row_count}

        object_count = output.get("object_count")
        if isinstance(object_count, int) and _number_present(answer, object_count):
            return True, {"matched": "object_count", "tool": tool, "object_count": object_count}

        latest = output.get("latest")
        if isinstance(latest, dict):
            for key in ("s3_uri", "key", "last_modified"):
                value = latest.get(key)
                if _value_present(answer, value):
                    return True, {"matched": f"latest.{key}", "tool": tool, "value": value}

        for value in _preview_values(output):
            if _value_present(answer, value):
                return True, {"matched": "preview_value", "tool": tool, "value": value}

    return False, {"checked_outputs": len(outputs)}


def _published_image_ok(answer: str, tool_calls: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    publish_calls = [tc for tc in tool_calls if str(tc.get("tool", "")) == "publish_image"]
    if not publish_calls:
        return False, {"error": "No publish_image call found."}
    if "chart" not in answer.lower():
        return False, {"error": "Final answer does not mention chart."}
    if not _has_evidence_line(answer):
        return False, {"error": "Final answer is missing a Source/Evidence line."}

    answer_l = answer.lower()
    for tc in publish_calls:
        output = str(tc.get("output", ""))
        published = bool(re.search(r"['\"]published['\"]\s*:\s*(?:True|true)", output))
        if not published:
            parsed = _parse_tool_output(output)
            published = isinstance(parsed, dict) and parsed.get("published") is True
        if not published:
            continue

        candidates: set[str] = set()
        args = _parse_tool_output(str(tc.get("arguments", "")))
        if isinstance(args, dict):
            for key in ("path", "file_path"):
                value = str(args.get(key) or "").strip()
                if value:
                    candidates.add(value)
                    candidates.add(Path(value).name)
        for match in re.finditer(r"(/[^'\"\s,}]+\.(?:png|jpg|jpeg|gif|webp|svg)|https?://[^'\"\s,}]+)", output, re.I):
            value = match.group(1)
            candidates.add(value)
            candidates.add(Path(value).name)
        if any(candidate and candidate.lower() in answer_l for candidate in candidates):
            return True, {"matched": "published_image_source"}
    return False, {"error": "No published image path/url is referenced in the final answer."}


def _has_followup_offer(answer: str) -> bool:
    return re.search(
        r"\b(?:"
        r"would you like me|"
        r"do you want me to|"
        r"if you want[, ]|"
        r"let me know if|"
        r"i can also|"
        r"i can help (?:with|you)|"
        r"want me to"
        r")",
        answer,
        flags=re.IGNORECASE,
    ) is not None


def _has_evidence_line(answer: str) -> bool:
    return re.search(
        r"(?im)^\s*(?:\*\*)?(source|sources|evidence|verified from|based on)(?:\*\*)?\s*:",
        answer,
    ) is not None


def _bullet_count(answer: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*]\s+|\d+[.)]\s+)", answer))


def _s3_freshness_wording_ok(answer: str, list_outputs: list[dict[str, Any]]) -> bool:
    if not list_outputs:
        return False
    answer_l = answer.lower()
    if re.search(r"\b(after|newer than)\s+today\b|\btoday'?s date\b", answer_l):
        return False
    for output in list_outputs:
        latest = output.get("latest") if isinstance(output, dict) else None
        if not isinstance(latest, dict):
            continue
        last_modified = str(latest.get("last_modified") or "")
        if last_modified and last_modified in answer:
            return re.search(r"\b(as of|last modified|timestamp|latest visible object)\b", answer_l) is not None
    return False


def _code_source_reference_present(answer: str, tool_calls: list[dict[str, Any]]) -> bool:
    candidates: set[str] = set()
    for tc in tool_calls:
        tool = str(tc.get("tool") or "")
        if tool not in {"read_file", "list_dir", "bash"}:
            continue
        parsed = _parse_tool_output(str(tc.get("arguments") or ""))
        if isinstance(parsed, dict):
            for key in ("file_path", "path", "cwd"):
                value = str(parsed.get(key) or "").strip()
                if value:
                    candidates.add(value)
                    candidates.add(Path(value).name)
        args = str(tc.get("arguments") or "")
        for match in re.finditer(r"((?:~|/)[^\s'\"`]+|[\w.-]+/[\w./-]+\.(?:py|java|scala|js|ts|yaml|yml|json|properties))", args):
            value = match.group(1).strip()
            candidates.add(value)
            candidates.add(Path(value).name)
    answer_l = answer.lower()
    return any(candidate and candidate.lower() in answer_l for candidate in candidates)


def _check_assertions(
    case: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    answer: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
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

    max_elapsed_seconds = assertions.get("max_elapsed_seconds")
    if max_elapsed_seconds is not None and elapsed_seconds is not None:
        add(
            "max_elapsed_seconds",
            elapsed_seconds <= float(max_elapsed_seconds),
            expected=max_elapsed_seconds,
            actual=round(elapsed_seconds, 1),
        )

    exact_tool_sequence = assertions.get("exact_tool_sequence")
    if exact_tool_sequence is not None:
        expected_sequence = [str(item) for item in exact_tool_sequence]
        add("exact_tool_sequence", tool_names == expected_sequence, expected=expected_sequence, actual=tool_names)

    for tool_name in assertions.get("required_tools", []) or []:
        add("required_tool", str(tool_name) in tool_set, expected=tool_name, actual=tool_names)

    for tool_name in assertions.get("forbidden_tools", []) or []:
        add("forbidden_tool", str(tool_name) not in tool_set, expected_absent=tool_name, actual=tool_names)

    if _is_internal_case(case) and _is_bounded_case(case) and not _asks_for_public_web(case):
        add("internal_bounded_no_web_search", "web_search" not in tool_set, actual=tool_names)

    answer_lower = answer.lower()
    for keyword in assertions.get("answer_contains", []) or []:
        add("answer_contains", str(keyword).lower() in answer_lower, keyword=keyword)

    for keyword in assertions.get("answer_not_contains", []) or []:
        add("answer_not_contains", str(keyword).lower() not in answer_lower, keyword=keyword)

    for pattern in assertions.get("answer_regex", []) or []:
        add("answer_regex", re.search(str(pattern), answer, flags=re.IGNORECASE | re.MULTILINE) is not None, pattern=pattern)

    for pattern in assertions.get("answer_not_regex", []) or []:
        add("answer_not_regex", re.search(str(pattern), answer, flags=re.IGNORECASE | re.MULTILINE) is None, pattern=pattern)

    if assertions.get("no_followup_offer") or _is_internal_case(case):
        add("no_followup_offer", not _has_followup_offer(answer))

    if assertions.get("evidence_line_present"):
        add("evidence_line_present", _has_evidence_line(answer))

    max_answer_chars = assertions.get("max_answer_chars")
    if max_answer_chars is not None:
        add("max_answer_chars", len(answer) <= int(max_answer_chars), expected=max_answer_chars, actual=len(answer))

    max_bullets = assertions.get("max_bullets")
    if max_bullets is not None:
        bullets = _bullet_count(answer)
        add("max_bullets", bullets <= int(max_bullets), expected=max_bullets, actual=bullets)

    for spec in assertions.get("tool_output_contains", []) or []:
        if not isinstance(spec, dict):
            continue
        tool = str(spec.get("tool") or "")
        text = str(spec.get("text") or "")
        matching_outputs = [str(tc.get("output", "")) for tc in tool_calls if str(tc.get("tool", "")) == tool]
        add(
            "tool_output_contains",
            any(text in output for output in matching_outputs),
            tool=tool,
            text=text,
            matching_tool_calls=len(matching_outputs),
        )

    for spec in assertions.get("tool_output_not_contains", []) or []:
        if not isinstance(spec, dict):
            continue
        tool = str(spec.get("tool") or "")
        text = str(spec.get("text") or "")
        matching_outputs = [str(tc.get("output", "")) for tc in tool_calls if not tool or str(tc.get("tool", "")) == tool]
        add(
            "tool_output_not_contains",
            all(text not in output for output in matching_outputs),
            tool=tool or "*",
            text=text,
            matching_tool_calls=len(matching_outputs),
        )

    if assertions.get("s3_actual_count_wording") or "list_s3" in tool_set:
        list_outputs = _tool_outputs(tool_calls, "list_s3")
        checked_any = False
        for output in list_outputs:
            object_count = output.get("object_count")
            max_keys_scanned = output.get("max_keys_scanned")
            if not isinstance(object_count, int) or not isinstance(max_keys_scanned, int):
                continue
            checked_any = True
            object_count_present = _number_present(answer, object_count)
            max_keys_pattern = rf"(?:{max_keys_scanned}|{max_keys_scanned:,})"
            scanned_cap_match = re.search(
                rf"\b(scan(?:ned)?|scanning)\s+(?:key\s+)?(?:count|keys?|objects?|cap)\b[^\n.;:]*[:=]?\s*`?{max_keys_pattern}`?",
                answer,
                flags=re.IGNORECASE,
            )
            misleading_scanned_cap = bool(
                object_count != max_keys_scanned
                and scanned_cap_match
                and not re.search(r"\b(cap|limit|requested|max-?keys?)\b", scanned_cap_match.group(0), flags=re.IGNORECASE)
            )
            add(
                "s3_actual_count_wording",
                object_count_present and not misleading_scanned_cap,
                object_count=object_count,
                max_keys_scanned=max_keys_scanned,
                object_count_present=object_count_present,
                misleading_scanned_cap=misleading_scanned_cap,
            )
        if not checked_any:
            add("s3_actual_count_wording", False, error="No parseable list_s3 output found.")

        for output in list_outputs:
            latest = output.get("latest") if isinstance(output, dict) else None
            if not isinstance(latest, dict):
                continue
            s3_uri = str(latest.get("s3_uri") or "")
            key = str(latest.get("key") or "")
            add(
                "s3_latest_path_present",
                bool((s3_uri and s3_uri in answer) or (key and key in answer)),
                s3_uri=s3_uri,
                key=key,
            )

        if assertions.get("s3_freshness_wording"):
            add("s3_freshness_wording", _s3_freshness_wording_ok(answer, list_outputs))

    search_outputs = _tool_outputs(tool_calls, "search_kb")
    if search_outputs and _is_internal_case(case) and not "bounded documentation answer" in _case_text(case):
        add(
            "internal_kb_has_structured_evidence",
            _has_structured_kb_evidence(search_outputs),
            search_calls=len(search_outputs),
        )

    if search_outputs and "quote at least one specific source file" in _case_text(case):
        add(
            "kb_source_reference_present",
            _source_reference_present(answer, search_outputs),
            search_calls=len(search_outputs),
        )

    if assertions.get("source_reference_present"):
        add(
            "source_reference_present",
            _source_reference_present(answer, search_outputs) or _code_source_reference_present(answer, tool_calls),
            search_calls=len(search_outputs),
        )

    if assertions.get("data_result_reflected"):
        ok, extra = _data_result_reflected(answer, tool_calls)
        add("data_result_reflected", ok, **extra)

    if assertions.get("published_image_ok"):
        ok, extra = _published_image_ok(answer, tool_calls)
        add("published_image_ok", ok, **extra)

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
    profile: str = "",
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
                            run_config=RunConfig(
                                workflow_name="DS Chat E2E smoke case",
                                trace_id=trace_id,
                                group_id=thread_id,
                                trace_metadata={
                                    "case": str(case.get("name") or ""),
                                    "thread_id": thread_id,
                                    "attempt": str(attempt),
                                    "profile": profile,
                                },
                                trace_include_sensitive_data=os.getenv("DS_CHAT_TRACE_SENSITIVE", "").lower()
                                in {"1", "true", "yes"},
                            ),
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
        elapsed_for_assertions = (datetime.now(timezone.utc) - started).total_seconds()
        assertion_result = _check_assertions(case, tool_calls, answer, elapsed_seconds=elapsed_for_assertions)
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
    elapsed_values = [
        float(report["elapsed_seconds"])
        for report in payload.get("reports", [])
        if isinstance(report.get("elapsed_seconds"), (int, float))
    ]
    if elapsed_values:
        avg_elapsed = sum(elapsed_values) / len(elapsed_values)
        lines.append(
            f"- Timing: min {min(elapsed_values):.1f}s, max {max(elapsed_values):.1f}s, avg {avg_elapsed:.1f}s"
        )
        lines.append("")
    lines.append("| # | Case | Status | Failure | Failed Assertions | Elapsed | Tools |")
    lines.append("|---|------|--------|---------|-------------------|---------|-------|")
    for idx, report in enumerate(payload.get("reports", []), 1):
        name = report.get("name", "unknown")
        status = "FAIL" if report.get("failed") else "PASS"
        failure = report.get("failure_kind", "")
        failed_assertions = ", ".join(
            str(detail.get("assertion"))
            for detail in (report.get("assertions", {}) or {}).get("details", [])
            if not detail.get("passed")
        )
        elapsed = report.get("elapsed_seconds", "?")
        tc_count = report.get("tool_call_count", 0)
        lines.append(f"| {idx} | {name} | {status} | {failure} | {failed_assertions} | {elapsed}s | {tc_count} |")
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
                    lines.append("**Output:**")
                    lines.append("```")
                    lines.append(str(output))
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

    include_web_search = bool(args.include_web_search or _cases_need_web_search(cases))
    agent = build_investigation_agent(args.model, include_web_search=include_web_search)
    print(f"Agent: {agent.name}, tools: {len(agent.tools)}, model: {args.model}")
    print(f"Web search tool: {'enabled' if include_web_search else 'disabled for selected internal smoke cases'}")
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
                profile=args.profile,
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
    parser.add_argument("--model", default="gpt-5-mini", help="Model to use for the agent (default: gpt-5-mini)")
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
    parser.add_argument("--include-web-search", action="store_true", help="Force-enable hosted web_search during E2E runs")
    parser.add_argument("--master", action="store_true", help="Run only master-tagged long-running cases with gpt-5.2 and 100 max turns")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip AWS credential bootstrap")
    args = parser.parse_args()

    return asyncio.run(run_all(args))


if __name__ == "__main__":
    raise SystemExit(main())
