"""Shared helpers for our @function_tool decorators.

Centralises the timeout + failure-text pattern recommended in the
OpenAI Agents SDK cookbook so every tool returns a deterministic
error string the model can parse, and no single tool can hang
longer than its class-level budget.
"""

from __future__ import annotations

from typing import Any


def tool_error(ctx: Any, error: Exception) -> str:
    """Default `failure_error_function` for tools that already
    catch-and-return errors but might still raise during schema
    validation. Keeps the string short and scannable."""
    name = type(error).__name__
    msg = str(error) or "(no detail)"
    # Strip tracebacks / paths that leak the CWD so the model sees
    # only the actionable piece.
    return f"Error ({name}): {msg}"


# Class-level timeouts, in seconds. Referenced by each tool module.
TIMEOUT_FAST = 30.0          # pure-local filesystem / in-memory
TIMEOUT_SHORT_NET = 60.0     # httpx / boto3 small calls
TIMEOUT_AWS = 120.0          # AWS control-plane reads
TIMEOUT_DB_QUERY = 600.0     # Redshift / MySQL queries
TIMEOUT_S3_FETCH = 300.0     # S3 multi-object fetch
TIMEOUT_LONG_SHELL = 1800.0  # bash can be legitimately slow


__all__ = [
    "tool_error",
    "TIMEOUT_FAST",
    "TIMEOUT_SHORT_NET",
    "TIMEOUT_AWS",
    "TIMEOUT_DB_QUERY",
    "TIMEOUT_S3_FETCH",
    "TIMEOUT_LONG_SHELL",
]
