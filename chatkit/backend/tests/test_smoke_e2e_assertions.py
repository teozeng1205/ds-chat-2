from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_e2e import _check_assertions, _render_markdown_report, _tool_error_type


def test_internal_bounded_cases_forbid_web_search_by_default() -> None:
    case = {
        "name": "schema_check",
        "question": "What tables are in prod.monitoring? Use search_kb. This is a bounded KB lookup.",
        "assertions": {"required_tools": ["search_kb"]},
    }
    tool_calls = [
        {"tool": "web_search"},
        {"tool": "search_kb", "output": "{'verified_items': [{'source_type': 'structured_snapshot'}]}"},
    ]

    result = _check_assertions(case, tool_calls, "provider_combined_audit")

    assert result["passed"] is False
    assert any(d["assertion"] == "internal_bounded_no_web_search" and not d["passed"] for d in result["details"])


def test_s3_count_wording_rejects_scan_cap_as_actual_count() -> None:
    case = {
        "name": "s3_freshness",
        "question": "Use list_s3 against s3-atp-3victors-3vdev-use1-pe-common-output and show the latest path.",
        "assertions": {},
    }
    tool_calls = [
        {
            "tool": "list_s3",
            "output": "{'object_count': 39275, 'max_keys_scanned': 50000, 'latest': {'s3_uri': 's3://bucket/latest.parquet', 'key': 'latest.parquet'}}",
        }
    ]
    answer = "Latest: s3://bucket/latest.parquet\nScanned key count: 50000"

    result = _check_assertions(case, tool_calls, answer)

    assert result["passed"] is False
    assert any(d["assertion"] == "s3_actual_count_wording" and not d["passed"] for d in result["details"])


def test_s3_count_wording_accepts_thousands_separator() -> None:
    case = {
        "name": "s3_freshness",
        "question": "Use list_s3 against s3-atp-3victors-3vdev-use1-pe-common-output and show the latest path.",
        "assertions": {},
    }
    tool_calls = [
        {
            "tool": "list_s3",
            "output": "{'object_count': 39275, 'max_keys_scanned': 50000, 'latest': {'s3_uri': 's3://bucket/latest.parquet', 'key': 'latest.parquet'}}",
        }
    ]
    answer = "Latest: s3://bucket/latest.parquet. The scan returned 39,275 visible objects under the 50,000 max-keys cap."

    result = _check_assertions(case, tool_calls, answer)

    assert result["passed"] is True


def test_s3_count_wording_accepts_labeled_scan_cap() -> None:
    case = {
        "name": "s3_freshness",
        "question": "Use list_s3 against s3-atp-3victors-3vdev-use1-pe-common-output and show the latest path.",
        "assertions": {},
    }
    tool_calls = [
        {
            "tool": "list_s3",
            "output": "{'object_count': 39275, 'max_keys_scanned': 50000, 'latest': {'s3_uri': 's3://bucket/latest.parquet', 'key': 'latest.parquet'}}",
        }
    ]
    answer = "Latest: s3://bucket/latest.parquet. Scanned key cap: 50000. Visible objects returned: 39275."

    result = _check_assertions(case, tool_calls, answer)

    assert result["passed"] is True


def test_internal_kb_requires_structured_evidence() -> None:
    case = {
        "name": "monitoring_schema",
        "question": "What tables are in the prod.monitoring schema? Use search_kb. This is a bounded KB lookup.",
        "assertions": {},
    }
    tool_calls = [
        {
            "tool": "search_kb",
            "output": "{'verified_items': [], 'hints': [{'source_type': 'doc_hint'}], 'tables': []}",
        }
    ]

    result = _check_assertions(case, tool_calls, "According to docs, provider_combined_audit exists.")

    assert result["passed"] is False
    assert any(d["assertion"] == "internal_kb_has_structured_evidence" and not d["passed"] for d in result["details"])


def test_source_reference_requirement_accepts_any_kb_source_path() -> None:
    case = {
        "name": "docs_answer",
        "question": "Use search_kb and quote at least one specific source file. This is a bounded documentation answer.",
        "assertions": {},
    }
    tool_calls = [
        {
            "tool": "search_kb",
            "output": "{'citations': [{'source': 'docs/priceeye_system.md#Overview'}], 'items': []}",
        }
    ]

    result = _check_assertions(case, tool_calls, "Source: docs/priceeye_system.md")

    assert result["passed"] is True


def test_source_reference_requirement_accepts_item_metadata_paths() -> None:
    case = {
        "name": "docs_answer",
        "question": "Use search_kb and quote at least one specific source file. This is a bounded documentation answer.",
        "assertions": {},
    }
    tool_calls = [
        {
            "tool": "search_kb",
            "output": "{'citations': [], 'items': [{'metadata': {'git_path': 'src/jobs/handler.py'}}]}",
        }
    ]

    result = _check_assertions(case, tool_calls, "Source file: src/jobs/handler.py")

    assert result["passed"] is True


def test_no_followup_offer_rejects_bounded_internal_answer() -> None:
    case = {
        "name": "bounded_priceeye",
        "question": "Bounded PriceEye KB lookup. Use search_kb and finish directly.",
        "assertions": {"required_tools": ["search_kb"]},
    }
    tool_calls = [
        {"tool": "search_kb", "output": "{'verified_items': [{'source_type': 'structured_snapshot'}]}"}
    ]

    result = _check_assertions(case, tool_calls, "PriceEye uses provider collection. I can also check the code.")

    assert result["passed"] is False
    assert any(d["assertion"] == "no_followup_offer" and not d["passed"] for d in result["details"])


def test_no_followup_offer_rejects_internal_codebase_answer() -> None:
    case = {
        "name": "auto_scheduler_codebase_lookup",
        "question": "How does the auto-scheduler work in priceeye-scheduling? Check the actual codebase.",
        "assertions": {"required_tools": ["search_kb"]},
    }
    tool_calls = [
        {"tool": "search_kb", "output": "{'verified_items': [{'source_type': 'structured_snapshot'}]}"},
        {"tool": "read_file", "arguments": '{"file_path": "/repo/PEAutoScheduler.java"}'},
    ]

    result = _check_assertions(case, tool_calls, "PEAutoScheduler is the entry point. If you want, I can go deeper.")

    assert result["passed"] is False
    assert any(d["assertion"] == "no_followup_offer" and not d["passed"] for d in result["details"])


def test_evidence_line_requirement() -> None:
    case = {
        "name": "docs_answer",
        "question": "Use search_kb and quote at least one specific source file.",
        "assertions": {"evidence_line_present": True},
    }

    result = _check_assertions(case, [], "Answer.\nSource: common_table_live_metadata.json")

    assert result["passed"] is True


def test_evidence_line_requirement_accepts_markdown_bold_label() -> None:
    case = {
        "name": "docs_answer",
        "question": "Use search_kb and quote at least one specific source file.",
        "assertions": {"evidence_line_present": True},
    }

    result = _check_assertions(case, [], "Answer.\n**Source:** common_table_live_metadata.json")

    assert result["passed"] is True


def test_evidence_line_requirement_rejects_unlabeled_source() -> None:
    case = {
        "name": "docs_answer",
        "question": "Use search_kb and quote at least one specific source file.",
        "assertions": {"evidence_line_present": True},
    }

    result = _check_assertions(case, [], "Answer from common_table_live_metadata.json")

    assert result["passed"] is False
    assert any(d["assertion"] == "evidence_line_present" and not d["passed"] for d in result["details"])


def test_s3_freshness_wording_rejects_today_date_confusion() -> None:
    case = {
        "name": "s3_freshness",
        "question": "Use list_s3 and say whether the latest object is stale.",
        "assertions": {"s3_freshness_wording": True},
    }
    tool_calls = [
        {
            "tool": "list_s3",
            "output": "{'object_count': 5, 'max_keys_scanned': 50000, 'latest': {'s3_uri': 's3://bucket/latest.parquet', 'last_modified': '2026-05-05T00:09:26+00:00'}}",
        }
    ]

    answer = "Latest: s3://bucket/latest.parquet. It is fresh because it is newer than today (`2026-05-04`)."
    result = _check_assertions(case, tool_calls, answer)

    assert result["passed"] is False
    assert any(d["assertion"] == "s3_freshness_wording" and not d["passed"] for d in result["details"])


def test_s3_freshness_wording_accepts_last_modified_anchor() -> None:
    case = {
        "name": "s3_freshness",
        "question": "Use list_s3 and say whether the latest object is stale.",
        "assertions": {"s3_freshness_wording": True},
    }
    tool_calls = [
        {
            "tool": "list_s3",
            "output": "{'object_count': 5, 'max_keys_scanned': 50000, 'latest': {'s3_uri': 's3://bucket/latest.parquet', 'last_modified': '2026-05-05T00:09:26+00:00'}}",
        }
    ]

    answer = "Latest: s3://bucket/latest.parquet. Fresh as of latest visible object timestamp `2026-05-05T00:09:26+00:00`. Visible objects returned: 5."
    result = _check_assertions(case, tool_calls, answer)

    assert result["passed"] is True


def test_answer_length_and_bullet_limits() -> None:
    case = {"name": "bounded", "question": "Bounded answer.", "assertions": {"max_answer_chars": 20, "max_bullets": 1}}

    result = _check_assertions(case, [], "- One\n- Two and this is too long")

    assert result["passed"] is False
    assert any(d["assertion"] == "max_answer_chars" and not d["passed"] for d in result["details"])
    assert any(d["assertion"] == "max_bullets" and not d["passed"] for d in result["details"])


def test_tool_error_type_detects_graph_empty_and_timeout() -> None:
    assert _tool_error_type("{'ok': False, 'error_type': 'GraphEmpty'}") == "GraphEmpty"
    assert _tool_error_type("Command timed out after 120 seconds") == "ToolTimeout"


def test_fail_on_tool_error_types_flags_graph_hydration_error() -> None:
    case = {
        "name": "code_lookup",
        "question": "Look up this codebase component.",
        "assertions": {"fail_on_tool_error_types": ["GraphHydrationError"]},
    }
    tool_calls = [{"tool": "explain_pipeline", "output": "{'ok': False, 'error_type': 'GraphHydrationError'}"}]

    result = _check_assertions(case, tool_calls, "The graph was unavailable.")

    assert result["passed"] is False
    assert any(d["assertion"] == "fail_on_tool_error_types" and not d["passed"] for d in result["details"])


def test_max_elapsed_seconds_assertion() -> None:
    case = {"name": "slow_case", "question": "Bounded check.", "assertions": {"max_elapsed_seconds": 10}}

    result = _check_assertions(case, [], "Done.", elapsed_seconds=12.4)

    assert result["passed"] is False
    assert any(d["assertion"] == "max_elapsed_seconds" and not d["passed"] for d in result["details"])


def test_exact_tool_sequence_assertion() -> None:
    case = {
        "name": "bounded_sequence",
        "question": "Run exactly these tools.",
        "assertions": {"exact_tool_sequence": ["fetch_s3", "run_python", "execute_sql"]},
    }
    tool_calls = [{"tool": "fetch_s3"}, {"tool": "execute_sql"}]

    result = _check_assertions(case, tool_calls, "Done.")

    assert result["passed"] is False
    assert any(d["assertion"] == "exact_tool_sequence" and not d["passed"] for d in result["details"])


def test_data_result_reflected_accepts_empty_sql_result() -> None:
    case = {
        "name": "empty_sql",
        "question": "Use execute_sql.",
        "assertions": {"data_result_reflected": True},
    }
    tool_calls = [{"tool": "execute_sql", "output": "{'row_count': 0, 'preview': []}"}]

    result = _check_assertions(case, tool_calls, "No rows were returned for QL2 today.")

    assert result["passed"] is True


def test_data_result_reflected_accepts_preview_value() -> None:
    case = {
        "name": "sql_preview",
        "question": "Use execute_sql.",
        "assertions": {"data_result_reflected": True},
    }
    tool_calls = [
        {
            "tool": "execute_sql",
            "output": "{'row_count': 2, 'preview': [{'provider': 'QL2', 'issue_count': 42}]}",
        }
    ]

    result = _check_assertions(case, tool_calls, "QL2 had 42 issue rows in the sampled result.")

    assert result["passed"] is True


def test_data_result_reflected_rejects_unanchored_answer() -> None:
    case = {
        "name": "sql_preview",
        "question": "Use execute_sql.",
        "assertions": {"data_result_reflected": True},
    }
    tool_calls = [
        {
            "tool": "execute_sql",
            "output": "{'row_count': 2, 'preview': [{'provider': 'QL2', 'issue_count': 42}]}",
        }
    ]

    result = _check_assertions(case, tool_calls, "The query completed successfully.")

    assert result["passed"] is False
    assert any(d["assertion"] == "data_result_reflected" and not d["passed"] for d in result["details"])


def test_data_result_reflected_accepts_list_s3_latest_path() -> None:
    case = {
        "name": "s3_latest",
        "question": "Use list_s3.",
        "assertions": {"data_result_reflected": True},
    }
    tool_calls = [
        {
            "tool": "list_s3",
            "output": "{'object_count': 3, 'max_keys_scanned': 50, 'latest': {'s3_uri': 's3://bucket/latest.parquet', 'last_modified': '2026-05-05T00:00:00+00:00'}}",
        }
    ]

    result = _check_assertions(
        case,
        tool_calls,
        "Latest visible object: s3://bucket/latest.parquet. Visible objects returned: 3.",
    )

    assert result["passed"] is True


def test_published_image_ok_accepts_chart_with_source_path() -> None:
    case = {
        "name": "plot",
        "question": "Publish a chart.",
        "assertions": {"published_image_ok": True},
    }
    tool_calls = [
        {
            "tool": "publish_image",
            "arguments": '{"path": "/tmp/chart.png", "display_name": "Chart"}',
            "output": "{'published': True, 'path': '/private/tmp/chart.png', 'image_url': AnyUrl('http://localhost/chart')}",
        }
    ]

    result = _check_assertions(case, tool_calls, "Published the chart.\nSource: /tmp/chart.png")

    assert result["passed"] is True


def test_published_image_ok_rejects_missing_source_path() -> None:
    case = {
        "name": "plot",
        "question": "Publish a chart.",
        "assertions": {"published_image_ok": True},
    }
    tool_calls = [
        {
            "tool": "publish_image",
            "arguments": '{"path": "/tmp/chart.png", "display_name": "Chart"}',
            "output": "{'published': True, 'path': '/private/tmp/chart.png'}",
        }
    ]

    result = _check_assertions(case, tool_calls, "Published the chart.\nSource: generated plot")

    assert result["passed"] is False
    assert any(d["assertion"] == "published_image_ok" and not d["passed"] for d in result["details"])


def test_source_reference_present_accepts_read_file_argument_path() -> None:
    case = {
        "name": "code_lookup",
        "question": "Check the actual codebase.",
        "assertions": {"source_reference_present": True},
    }
    tool_calls = [
        {"tool": "read_file", "arguments": '{"file_path": "/home/ec2-user/git/app/src/scheduler.py"}'}
    ]

    result = _check_assertions(case, tool_calls, "The entry point is in scheduler.py.")

    assert result["passed"] is True


def test_markdown_summary_includes_failed_assertions_and_timing() -> None:
    report = _render_markdown_report({
        "generated_at": "2026-05-05T00:00:00+00:00",
        "model": "test-model",
        "max_turns": 10,
        "case_timeout_seconds": 30,
        "reports": [
            {
                "name": "case_a",
                "failed": True,
                "failure_kind": "assertion",
                "elapsed_seconds": 2.0,
                "tool_call_count": 1,
                "question": "Q",
                "assertions": {
                    "checked": True,
                    "passed": False,
                    "details": [{"assertion": "data_result_reflected", "passed": False}],
                },
            },
            {
                "name": "case_b",
                "failed": False,
                "elapsed_seconds": 4.0,
                "tool_call_count": 0,
                "question": "Q",
            },
        ],
    })

    assert "Timing: min 2.0s, max 4.0s, avg 3.0s" in report
    assert "Failed Assertions" in report
    assert "| 1 | case_a | FAIL | assertion | data_result_reflected | 2.0s | 1 |" in report
