"""Typed feature-flag + runtime-config loader.

Central home for env-driven switches used across the app. Each flag
has a typed accessor so callers get validation and a sensible
default.

Philosophy:
  - Flags default to SAFE / OFF for new features (so a fresh clone
    matches today's behavior).
  - Reads are done lazily on each call — tests can monkeypatch the
    env or the cached Config directly.

Flags:
  - PLANNER_REVIEWER_MODE    off | shadow | on     (default: off)
  - QUERY_CACHE_ENABLED      bool                  (default: true)
  - TRACING_ENABLED          bool                  (default: true)
  - COST_TELEMETRY_ENABLED   bool                  (default: true)
  - SEMANTIC_KB_ENABLED      bool                  (default: false)
  - GLUE_PARTITION_GUARD     bool                  (default: false)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class PlannerReviewerMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ON = "on"

    @classmethod
    def parse(cls, raw: str | None) -> "PlannerReviewerMode":
        if not raw:
            return cls.OFF
        value = raw.strip().lower()
        for member in cls:
            if member.value == value:
                return member
        return cls.OFF


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw and raw.strip() else default


@dataclass(frozen=True)
class Config:
    planner_reviewer_mode: PlannerReviewerMode
    query_cache_enabled: bool
    tracing_enabled: bool
    cost_telemetry_enabled: bool
    semantic_kb_enabled: bool
    glue_partition_guard_enabled: bool

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            planner_reviewer_mode=PlannerReviewerMode.parse(os.environ.get("PLANNER_REVIEWER_MODE")),
            query_cache_enabled=_env_bool("QUERY_CACHE_ENABLED", True),
            tracing_enabled=_env_bool("TRACING_ENABLED", True),
            cost_telemetry_enabled=_env_bool("COST_TELEMETRY_ENABLED", True),
            semantic_kb_enabled=_env_bool("SEMANTIC_KB_ENABLED", False),
            glue_partition_guard_enabled=_env_bool("GLUE_PARTITION_GUARD_ENABLED", False),
        )


def load_config() -> Config:
    """Re-read env vars every call.

    Cheap (a handful of os.environ lookups) and keeps tests simple:
    they can monkeypatch env and call load_config() without touching
    a cache.
    """
    return Config.from_env()


__all__ = [
    "Config",
    "PlannerReviewerMode",
    "load_config",
]
