"""Unit tests for app.config."""

from __future__ import annotations

import pytest

from app.config import PlannerReviewerMode, load_config


def _clear(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for n in names:
        monkeypatch.delenv(n, raising=False)


ALL_ENV = (
    "PLANNER_REVIEWER_MODE",
    "QUERY_CACHE_ENABLED",
    "TRACING_ENABLED",
    "COST_TELEMETRY_ENABLED",
    "SEMANTIC_KB_ENABLED",
    "GLUE_PARTITION_GUARD_ENABLED",
)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch, *ALL_ENV)
    cfg = load_config()
    assert cfg.planner_reviewer_mode is PlannerReviewerMode.OFF
    assert cfg.query_cache_enabled is True
    assert cfg.tracing_enabled is True
    assert cfg.cost_telemetry_enabled is True
    assert cfg.semantic_kb_enabled is False
    assert cfg.glue_partition_guard_enabled is False


def test_planner_reviewer_mode_parses_known_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw, expected in [
        ("off", PlannerReviewerMode.OFF),
        ("Shadow", PlannerReviewerMode.SHADOW),
        ("ON", PlannerReviewerMode.ON),
        ("bogus", PlannerReviewerMode.OFF),
        ("", PlannerReviewerMode.OFF),
        (None, PlannerReviewerMode.OFF),
    ]:
        assert PlannerReviewerMode.parse(raw) is expected


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_REVIEWER_MODE", "shadow")
    monkeypatch.setenv("QUERY_CACHE_ENABLED", "false")
    monkeypatch.setenv("SEMANTIC_KB_ENABLED", "1")
    monkeypatch.setenv("GLUE_PARTITION_GUARD_ENABLED", "yes")
    cfg = load_config()
    assert cfg.planner_reviewer_mode is PlannerReviewerMode.SHADOW
    assert cfg.query_cache_enabled is False
    assert cfg.semantic_kb_enabled is True
    assert cfg.glue_partition_guard_enabled is True


def test_bool_truthy_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw, expected in [("1", True), ("true", True), ("yes", True), ("on", True),
                           ("0", False), ("false", False), ("no", False), ("off", False),
                           ("", True),  # empty → treated like unset in _env_str; _env_bool handles empty as NOT truthy
                           ]:
        monkeypatch.setenv("QUERY_CACHE_ENABLED", raw)
        cfg = load_config()
        # For empty string the loader falls back to default (True); for any clearly-falsy value returns False
        if raw in {"1", "true", "yes", "on"}:
            assert cfg.query_cache_enabled is True, f"{raw} should be True"
        elif raw == "":
            # empty string is NOT in the truthy set → False under _env_bool
            assert cfg.query_cache_enabled is False
        else:
            assert cfg.query_cache_enabled is False, f"{raw} should be False"
