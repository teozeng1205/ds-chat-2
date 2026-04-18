"""@function_tool wrappers for user/thread memory.

Gives the agent:
  - remember(key, value, scope)
  - recall(key, scope)
  - list_memories(scope)
  - forget(key, scope)

Scopes: "user" (cross-thread, defaults to DEFAULT_USER_ID until auth
lands) or "thread" (this conversation only).

No background reader yet — a later commit can auto-inject
`list_memories(scope='user')` into the agent context at turn start
so preferences are in play without being recalled explicitly.
"""

from __future__ import annotations

import logging
from typing import Any

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent

from ..memory import DEFAULT_USER_ID, Scope, get_memory_store

log = logging.getLogger(__name__)

MAX_KEY_LEN = 120
MAX_VALUE_LEN = 4000


def _thread_id(ctx: RunContextWrapper[AgentContext]) -> str:  # type: ignore[type-arg]
    thread = getattr(getattr(ctx, "context", None), "thread", None)
    tid = getattr(thread, "id", None)
    return str(tid) if tid else "default-thread"


def _user_id(ctx: RunContextWrapper[AgentContext]) -> str:  # type: ignore[type-arg]
    # Once auth lands this flips to pulling the user identity from the
    # request context. For now every user shares DEFAULT_USER_ID.
    _ = ctx
    return DEFAULT_USER_ID


def _scope_id(ctx: RunContextWrapper[AgentContext], scope: Scope) -> str:  # type: ignore[type-arg]
    return _thread_id(ctx) if scope == "thread" else _user_id(ctx)


async def _stream(ctx: RunContextWrapper[AgentContext], icon: str, text: str) -> None:  # type: ignore[type-arg]
    try:
        await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))
    except Exception:
        pass


def _err(exc: Exception) -> dict[str, Any]:
    log.exception("memory tool failed: %s", exc)
    return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}


def _validate(key: str, value: str | None = None) -> str | None:
    if not key or len(key) > MAX_KEY_LEN:
        return f"key must be 1..{MAX_KEY_LEN} chars"
    if value is not None and len(value) > MAX_VALUE_LEN:
        return f"value must be <= {MAX_VALUE_LEN} chars"
    return None


@function_tool
async def remember(
    ctx: RunContextWrapper[AgentContext],
    key: str,
    value: str,
    scope: str = "user",
) -> dict[str, Any]:
    """Persist a small fact for future turns.

    Examples:
      remember("default_customer", "B6")
      remember("team", "priceeye-scheduling")
      remember("preferred_datasource", "redshift_analytics")
      remember("current_investigation", "QL2 spike on 2026-04-17", scope="thread")

    Args:
        key: short identifier, max 120 chars, [a-zA-Z0-9_-] style.
        value: value to store, max 4000 chars.
        scope: "user" (cross-thread, default) or "thread" (this conversation only).
    """
    try:
        s: Scope = "thread" if scope == "thread" else "user"
        err = _validate(key, value)
        if err:
            return {"ok": False, "error": err, "error_type": "ValidationError"}

        await _stream(ctx, "bookmark", f"Remembering {key!r} ({s}).")
        get_memory_store().put(scope=s, scope_id=_scope_id(ctx, s), key=key, value=value)
        await _stream(ctx, "check-circle", f"Saved to {s} memory.")
        return {"ok": True, "scope": s, "key": key}
    except Exception as exc:
        return _err(exc)


@function_tool
async def recall(
    ctx: RunContextWrapper[AgentContext],
    key: str,
    scope: str = "user",
) -> dict[str, Any]:
    """Return a previously-remembered value. Missing keys return value=None."""
    try:
        s: Scope = "thread" if scope == "thread" else "user"
        err = _validate(key)
        if err:
            return {"ok": False, "error": err, "error_type": "ValidationError"}
        val = get_memory_store().get(scope=s, scope_id=_scope_id(ctx, s), key=key)
        return {"ok": True, "scope": s, "key": key, "value": val, "found": val is not None}
    except Exception as exc:
        return _err(exc)


@function_tool
async def list_memories(
    ctx: RunContextWrapper[AgentContext],
    scope: str = "user",
) -> dict[str, Any]:
    """List everything stored in the given scope. Useful on session start
    to recall user preferences."""
    try:
        s: Scope = "thread" if scope == "thread" else "user"
        items = get_memory_store().list(scope=s, scope_id=_scope_id(ctx, s))
        return {"ok": True, "scope": s, "count": len(items), "items": items}
    except Exception as exc:
        return _err(exc)


@function_tool
async def forget(
    ctx: RunContextWrapper[AgentContext],
    key: str,
    scope: str = "user",
) -> dict[str, Any]:
    """Delete a remembered fact. Idempotent — absent key returns ok with deleted=False."""
    try:
        s: Scope = "thread" if scope == "thread" else "user"
        err = _validate(key)
        if err:
            return {"ok": False, "error": err, "error_type": "ValidationError"}
        deleted = get_memory_store().delete(scope=s, scope_id=_scope_id(ctx, s), key=key)
        return {"ok": True, "scope": s, "key": key, "deleted": deleted}
    except Exception as exc:
        return _err(exc)


def memory_tools() -> list[Any]:
    return [remember, recall, list_memories, forget]


__all__ = ["remember", "recall", "list_memories", "forget", "memory_tools"]
