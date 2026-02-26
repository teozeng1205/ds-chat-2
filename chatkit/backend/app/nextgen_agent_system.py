"""Compatibility shim for historical nextgen agent system imports."""

from __future__ import annotations

import warnings

from .agents.orchestrator import build_agent

warnings.warn(
    "app.nextgen_agent_system is deprecated; use app.agents.orchestrator.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["build_agent"]
