"""Shared helpers for our @function_tool decorators.

Centralises the timeout + failure-text pattern recommended in the
OpenAI Agents SDK cookbook so every tool returns a deterministic
error string the model can parse, and no single tool can hang
longer than its class-level budget.
"""

from __future__ import annotations

from typing import Any


# ── Workflow tool-call traces ───────────────────────────────────────────────
# Render each tool call as a persistent ChatKit workflow task so the full trace
# (e.g. the exact SQL, the bash command) stays visible in the thread and in
# history. trace_begin() adds a "loading" task and returns its index;
# trace_finish() flips it to "complete". trace_done() is the one-shot form for
# fast tools. All are best-effort and never raise into the calling tool.

def _clip(content: Any, limit: int = 2000) -> str | None:
    if not content:
        return None
    text = str(content).strip()
    if not text:
        return None
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


async def _emit_task(ctx: Any, *, title: str, content: Any, icon: str | None, status: str, index: int | None):
    """Add (index is None) or update (index given) a CustomTask. Returns the
    task index for adds, else None. Retries without the icon if rejected."""
    try:
        from chatkit.types import CustomTask  # local import: keep _common light
    except Exception:
        return None
    agent_ctx = getattr(ctx, "context", None)
    if agent_ctx is None:
        return None
    for kwargs in ({"icon": icon}, {}):
        try:
            task = CustomTask(title=str(title)[:200], content=_clip(content), status_indicator=status, **kwargs)  # type: ignore[arg-type]
        except Exception:
            continue
        try:
            if index is None:
                await agent_ctx.add_workflow_task(task)
                wf = getattr(agent_ctx, "workflow_item", None)
                if not wf:
                    return None
                for i, t in enumerate(wf.workflow.tasks):
                    if t is task:
                        return i
                return None
            await agent_ctx.update_workflow_task(task, index)
            return index
        except Exception:
            continue
    return None


async def trace_begin(ctx: Any, *, title: str, content: Any = None, icon: str | None = None) -> int | None:
    """Add a 'loading' workflow task; returns its index (or None)."""
    return await _emit_task(ctx, title=title, content=content, icon=icon, status="loading", index=None)


async def trace_finish(ctx: Any, index: int | None, *, title: str, content: Any = None, icon: str | None = None) -> None:
    """Flip the task at `index` to 'complete' with its final title/content."""
    if index is None:
        return
    await _emit_task(ctx, title=title, content=content, icon=icon, status="complete", index=index)


async def trace_done(ctx: Any, *, title: str, content: Any = None, icon: str | None = None) -> None:
    """One-shot 'complete' task for fast tools (no loading phase)."""
    await _emit_task(ctx, title=title, content=content, icon=icon, status="complete", index=None)


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
    "trace_begin",
    "trace_finish",
    "trace_done",
    "TIMEOUT_FAST",
    "TIMEOUT_SHORT_NET",
    "TIMEOUT_AWS",
    "TIMEOUT_DB_QUERY",
    "TIMEOUT_S3_FETCH",
    "TIMEOUT_LONG_SHELL",
]
