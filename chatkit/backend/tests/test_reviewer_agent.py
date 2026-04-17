"""Tests for the reviewer sub-agent: builder + verdict parser."""

from __future__ import annotations

from app.agents.reviewer import (
    DEFAULT_REVIEWER_MAX_TURNS,
    DEFAULT_REVIEWER_MODEL,
    REVIEWER_INSTRUCTIONS,
    REVIEWER_TOOL_NAME,
    build_reviewer_agent,
    parse_verdict,
    reviewer_tool_description,
)


def test_reviewer_builds_with_no_tools() -> None:
    r = build_reviewer_agent()
    assert r.name == "DS Chat Reviewer"
    assert REVIEWER_INSTRUCTIONS.strip() in (r.instructions or "")
    assert r.tools == []


def test_reviewer_defaults() -> None:
    assert DEFAULT_REVIEWER_MODEL == "gpt-5.4-mini"
    assert DEFAULT_REVIEWER_MAX_TURNS == 2
    assert REVIEWER_TOOL_NAME == "review_answer"


def test_reviewer_tool_description_mentions_verdict() -> None:
    desc = reviewer_tool_description()
    assert "verdict" in desc and "JSON" in desc


# ── parse_verdict ──


def test_parse_verdict_happy_path() -> None:
    raw = '{"verdict":"pass","confidence":0.92,"concerns":[]}'
    v = parse_verdict(raw)
    assert v == {"verdict": "pass", "confidence": 0.92, "concerns": []}


def test_parse_verdict_extracts_json_from_surrounding_text() -> None:
    raw = "here is my verdict:\n{\"verdict\":\"hard_fail\",\"confidence\":0.3,\"concerns\":[{\"kind\":\"math_error\",\"text\":\"5+5 = 11?\"}]}\nthanks"
    v = parse_verdict(raw)
    assert v["verdict"] == "hard_fail"
    assert v["concerns"] == [{"kind": "math_error", "text": "5+5 = 11?"}]


def test_parse_verdict_clamps_confidence() -> None:
    assert parse_verdict('{"verdict":"pass","confidence":1.7,"concerns":[]}')["confidence"] == 1.0
    assert parse_verdict('{"verdict":"pass","confidence":-0.1,"concerns":[]}')["confidence"] == 0.0


def test_parse_verdict_unknown_verdict_becomes_soft_fail() -> None:
    v = parse_verdict('{"verdict":"superb","confidence":0.9}')
    assert v["verdict"] == "soft_fail"


def test_parse_verdict_no_json_returns_soft_fail() -> None:
    v = parse_verdict("no json here at all")
    assert v["verdict"] == "soft_fail"
    assert v["concerns"][0]["kind"] == "other"


def test_parse_verdict_invalid_json_returns_soft_fail() -> None:
    v = parse_verdict("{verdict: not_json}")
    assert v["verdict"] == "soft_fail"
    assert v["concerns"][0]["text"].startswith("reviewer returned")


def test_parse_verdict_filters_malformed_concerns() -> None:
    raw = '{"verdict":"soft_fail","confidence":0.5,"concerns":[{"kind":"missing_partition","text":"no sales_date"}, "not-a-dict", {"text":"no kind"}]}'
    v = parse_verdict(raw)
    kinds = [c["kind"] for c in v["concerns"]]
    # dict-without-kind becomes "other"; string entries are dropped
    assert kinds == ["missing_partition", "other"]
