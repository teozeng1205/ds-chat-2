"""Seven shell/filesystem tools for the DS Chat coding agent.

All tools are stateless @function_tool functions; PTY session state
lives in shell_session._registry keyed by thread_id.
"""

from __future__ import annotations

import asyncio
import fnmatch
import glob as _glob
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent
from chatkit.widgets import Card
from pydantic import BaseModel

from ..investigation.shell_session import get_session

log = logging.getLogger(__name__)

# Blocked git subcommands / flags
_GIT_BLOCKED = {"push --force", "push -f", "reset --hard", "clean -f", "clean -fd"}

# Default base for relative paths
_GIT_BASE = Path("~/git").expanduser()


def _thread_id(ctx: RunContextWrapper[AgentContext]) -> str:  # type: ignore[type-arg]
    thread = getattr(ctx.context, "thread", None)
    tid = getattr(thread, "id", None)
    return str(tid) if tid else "default-thread"


async def _stream_progress(ctx: RunContextWrapper[AgentContext], icon: str, text: str) -> None:  # type: ignore[type-arg]
    try:
        await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))
    except Exception:
        pass  # Progress streaming is best-effort; never crash the tool over it


def _resolve_path(file_path: str) -> Path:
    """Resolve a file path; relative paths are anchored at ~/git/."""
    p = Path(file_path).expanduser()
    if not p.is_absolute():
        p = _GIT_BASE / p
    return p.resolve()


# ── Tool 1: bash ──

@function_tool
async def bash(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    command: str,
    timeout: int = 120,
) -> str:
    """Run a bash command in a persistent shell session for this conversation.

    CWD, environment variables, and background processes persist across calls.
    Supports: cd (persists), export (persists), background jobs (&), pipes, etc.
    Returns combined stdout+stderr (last 8192 chars). On non-zero exit, output
    includes a retry hint.

    Args:
        command: The bash command to run.
        timeout: Max seconds to wait (default 120, max 1800).
    """
    thread_id = _thread_id(ctx)
    await _stream_progress(ctx, "square-code", f"$ {command[:80]}")

    shell = await get_session(thread_id)

    # Stream live chunks for long-running commands
    output_chunks: list[str] = []
    start_time = time.monotonic()
    last_streamed = time.monotonic()
    last_activity = time.monotonic()
    async for chunk in shell.run_streaming(command, timeout=min(timeout, 1800)):
        now = time.monotonic()
        if chunk:
            output_chunks.append(chunk)
            last_activity = now
        # Relay a progress update every ~2 seconds when there's output
        if chunk and now - last_streamed >= 2.0:
            preview = "".join(output_chunks)[-300:]
            await _stream_progress(ctx, "square-code", f"$ {command[:40]}\n…{preview}")
            last_streamed = now
        # Heartbeat every 30s when silent — keeps nginx SSE connection alive
        elif not chunk and now - last_activity >= 30.0:
            last_activity = now
            elapsed = int(now - start_time)
            await _stream_progress(ctx, "clock", f"Still running… ({elapsed}s elapsed)")

    output = "".join(output_chunks)
    if not output:
        output = ""

    elapsed_total = int(time.monotonic() - start_time)
    line_count = output.count("\n") + (1 if output and not output.endswith("\n") else 0)

    exit_ok = not any(
        output.rstrip().endswith(marker)
        for marker in ("Error", "error", "not found", "failed", "No such")
    )

    # Build Card title with elapsed time
    elapsed_str = f" ({elapsed_total}s)" if elapsed_total >= 2 else ""
    card_title = f"$ {command[:60]}{elapsed_str}"

    # Build subtitle for large output
    subtitle = f"{line_count} lines" if line_count > 50 else None

    # "View Full Output" button for large outputs via data URL
    import base64 as _base64
    full_output_url = (
        "data:text/plain;base64,"
        + _base64.b64encode(output.encode("utf-8", errors="replace")).decode("ascii")
    )

    # Publish a Terminal Card with copy + view-full-output buttons
    status_type = "success" if exit_ok else "error"
    card_status: dict = {"type": status_type, "title": card_title, "text": subtitle or ""}

    card_children: list = [
        {"type": "Markdown", "value": f"```\n{output}\n```"},
        {
            "type": "Button",
            "label": "Copy",
            "style": "secondary",
            "onClickAction": {
                "type": "copy_to_clipboard",
                "handler": "client",
                "loadingBehavior": "none",
                "payload": {"text": output},
            },
        },
    ]
    if line_count > 50:
        card_children.append({
            "type": "Button",
            "label": "View Full Output",
            "style": "secondary",
            "onClickAction": {
                "type": "open_url",
                "handler": "client",
                "loadingBehavior": "none",
                "payload": {"url": full_output_url},
            },
        })

    try:
        await ctx.context.stream_widget(
            Card(
                size="lg",
                status=card_status,
                children=card_children,
            )
        )
    except Exception:
        pass  # Widget publishing is best-effort

    if not exit_ok:
        output += "\n[Non-zero exit detected. Review output, fix command, and retry.]"
    return output


# ── Tool 2: read_file ──

@function_tool
async def read_file(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    file_path: str,
    offset: int = 1,
    limit: int = 400,
) -> str:
    """Read a file with line numbers (cat -n format: '   {n}\\t{line}').

    Relative paths resolved under ~/git/. offset is 1-indexed start line.
    limit is max lines to return (cap 2000).

    Args:
        file_path: Absolute or ~/git/-relative path.
        offset: 1-indexed line to start reading from (default 1).
        limit: Max lines to return (default 400, cap 2000).
    """
    try:
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: file not found: {path}"
        if not path.is_file():
            return f"Error: not a file: {path}"

        limit = min(limit, 2000)
        offset = max(1, offset)

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total = len(lines)
        slice_lines = lines[offset - 1: offset - 1 + limit]
        numbered = "".join(
            f"{i + offset:6d}\t{line}"
            for i, line in enumerate(slice_lines)
        )
        suffix = f"\n[total_lines={total}, showing {offset}–{offset + len(slice_lines) - 1}]"
        return numbered + suffix
    except Exception as exc:
        return f"Error reading {file_path}: {exc}"


# ── Tool 3: list_dir ──

@function_tool
async def list_dir(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    path: str,
    pattern: str = "*",
) -> str:
    """List directory contents with optional glob filter (e.g. '*.py', '**/*.ts').

    Relative paths resolved under ~/git/. Returns up to 200 entries:
    type|size|modified|name

    Args:
        path: Absolute or ~/git/-relative directory path.
        pattern: Glob pattern to filter (default '*', use '**/*.py' for recursive).
    """
    try:
        base = _resolve_path(path)
        if not base.exists():
            return f"Error: path not found: {base}"
        if not base.is_dir():
            return f"Error: not a directory: {base}"

        recursive = "**" in pattern
        if recursive:
            matches = sorted(base.glob(pattern))
        else:
            matches = sorted(base.glob(pattern))

        entries: list[str] = []
        for p in matches[:200]:
            stat = p.stat()
            kind = "d" if p.is_dir() else "f"
            size = stat.st_size
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
            try:
                rel = p.relative_to(base)
            except ValueError:
                rel = p
            entries.append(f"{kind}|{size:>10}|{mtime}|{rel}")

        header = f"# {base} ({len(entries)} entries)\ntype|size|modified|name"
        return header + "\n" + "\n".join(entries)
    except Exception as exc:
        return f"Error listing {path}: {exc}"


# ── Tool 4: edit_file ──

@function_tool
async def edit_file(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    file_path: str,
    old_string: str,
    new_string: str,
) -> str:
    """Replace old_string with new_string in a file.

    old_string must appear exactly once. This enforces read-before-edit:
    0 matches → read the file first to get exact content;
    2+ matches → include more surrounding context to make it unique.

    A diff card is published on success.

    Args:
        file_path: Path to file (absolute or ~/git/-relative).
        old_string: Exact string to replace (must appear exactly once).
        new_string: Replacement string.
    """
    try:
        import difflib

        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: file not found: {path}. Read the file first with read_file."
        if not path.is_file():
            return f"Error: not a regular file: {path}"

        old_content = path.read_text(encoding="utf-8")
        count = old_content.count(old_string)
        if count == 0:
            return (
                f"Error: old_string not found in {file_path}. "
                "Use read_file to view exact content before editing."
            )
        if count > 1:
            return (
                f"Error: old_string found {count} times in {file_path}. "
                "Add more surrounding context to make it unique."
            )

        new_content = old_content.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")

        # Publish a diff card
        try:
            diff_lines = list(difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
                lineterm="",
            ))
            diff_text = "\n".join(diff_lines)
            await ctx.context.stream_widget(
                Card(
                    size="lg",
                    status={"type": "success", "title": f"Edited {path.name}", "text": ""},
                    children=[
                        {"type": "Markdown", "value": f"```diff\n{diff_text}\n```"},
                    ],
                )
            )
        except Exception:
            pass  # Widget publishing is best-effort

        return f"OK: edited {path} ({abs(len(new_content) - len(old_content)):+d} chars)"
    except Exception as exc:
        return f"Error editing {file_path}: {exc}"


# ── Tool 5: git ──

@function_tool
async def git(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    args: str,
    working_dir: str | None = None,
) -> str:
    """Run any git command (args as CLI string: 'log --oneline -5', 'status', 'diff HEAD~1').

    working_dir defaults to ~/git/. Blocked: push --force, reset --hard, clean -f.

    Args:
        args: Git subcommand and flags (e.g. 'log --oneline -5').
        working_dir: Directory to run in (defaults to ~/git/).
    """
    # Safety check for blocked operations
    args_lower = args.strip().lower()
    for blocked in _GIT_BLOCKED:
        if blocked in args_lower:
            return f"Error: blocked git operation '{blocked}'. Use safer alternatives."

    cwd = _resolve_path(working_dir) if working_dir else _GIT_BASE
    if not cwd.exists():
        cwd = Path.home()

    try:
        await _stream_progress(ctx, "square-code", f"git {args[:60]}")
        proc = await asyncio.create_subprocess_exec(
            "git", *args.split(),
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
        return output.strip() or "(no output)"
    except asyncio.TimeoutError:
        return "Error: git command timed out after 30s"
    except Exception as exc:
        return f"Error running git {args}: {exc}"


# ── Tool 6: fetch_url ──

@function_tool
async def fetch_url(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    url: str,
    max_chars: int = 8000,
) -> str:
    """Fetch text content from a URL (docs, arxiv, Stack Overflow, GitHub).

    Strips HTML tags. Returns up to max_chars (cap 32000).

    Args:
        url: URL to fetch.
        max_chars: Max characters to return (default 8000, cap 32000).
    """
    import re

    max_chars = min(max_chars, 32000)
    try:
        await _stream_progress(ctx, "globe", f"Fetching {url[:80]}")
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            text = response.text

        # Strip HTML if applicable
        if "html" in content_type:
            # Remove scripts, styles, and tags
            text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n[...truncated at {max_chars} chars]"
        return text.strip()
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


# ── Tool 7: render_image ──

@function_tool
async def render_image(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    file_path: str,
    title: str = "Plot",
) -> str:
    """Display an image file inline in the chat as a widget card.

    Use this after generating any plot or image file to show it to the user.
    Supports PNG, JPEG, GIF, WebP, SVG files.

    Args:
        file_path: Absolute path to the image file (e.g. /tmp/plot.png).
        title: Optional title shown above the image (default 'Plot').
    """
    import base64
    import mimetypes

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return f"Error: file not found: {path}"
    if not path.is_file():
        return f"Error: not a file: {path}"

    mime = mimetypes.guess_type(path.name)[0] or ""
    if not mime.startswith("image/"):
        return f"Error: {path.name} does not appear to be an image (mime={mime})"

    file_bytes = path.read_bytes()
    if not file_bytes:
        return f"Error: image file is empty: {path}"

    inline_data_url = f"data:{mime};base64,{base64.b64encode(file_bytes).decode('ascii')}"

    try:
        await ctx.context.stream_widget(
            Card(
                size="lg",
                status={"type": "success", "title": title, "text": title},
                children=[
                    {"type": "Image", "src": inline_data_url, "alt": title, "fit": "contain", "maxHeight": 500},
                    {
                        "type": "Button",
                        "label": "Download",
                        "style": "secondary",
                        "onClickAction": {
                            "type": "download_url",
                            "handler": "client",
                            "loadingBehavior": "none",
                            "payload": {"url": inline_data_url, "filename": path.name},
                        },
                    },
                ],
            )
        )
    except Exception as exc:
        return f"Image saved to {path}. Could not render widget: {exc}"

    return f"Image displayed inline."


# ── Tool 8: download_file ──

@function_tool
async def download_file(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    file_path: str,
    title: str = "",
    description: str = "",
) -> str:
    """Make any file downloadable from the chat as a card with a Download button.

    Use this after the agent creates a file the user may want to keep:
    CSV exports, JSON reports, Excel sheets, PDFs, text files, etc.
    Supports any file up to ~10 MB. For images, prefer render_image.

    Args:
        file_path: Absolute or ~/git/-relative path to the file.
        title: Card title (defaults to filename).
        description: Optional one-line description shown under the title.
    """
    import base64
    import mimetypes

    path = _resolve_path(file_path)
    if not path.exists():
        return f"Error: file not found: {path}"
    if not path.is_file():
        return f"Error: not a file: {path}"

    file_bytes = path.read_bytes()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > 10:
        return (
            f"Error: file is {size_mb:.1f} MB — too large for in-chat download (10 MB limit). "
            f"The file is at {path}. Use scp or rsync to retrieve it."
        )
    if not file_bytes:
        return f"Error: file is empty: {path}"

    await _stream_progress(ctx, "download", f"Preparing {path.name} for download…")

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    inline_data_url = f"data:{mime};base64,{base64.b64encode(file_bytes).decode('ascii')}"

    card_title = title or path.name
    size_str = f"{len(file_bytes):,} bytes" if size_mb < 0.1 else f"{size_mb:.1f} MB"
    subtitle = description or f"{size_str} · {path.suffix.lstrip('.').upper() or 'file'}"

    # Build preview for text-based formats
    children: list[Any] = []
    text_mimes = {"text/csv", "application/json", "text/plain", "text/markdown", "text/html"}
    if mime in text_mimes or path.suffix.lower() in {".csv", ".json", ".txt", ".md", ".log"}:
        try:
            text_preview = file_bytes.decode("utf-8", errors="replace")
            lines = text_preview.splitlines()[:20]
            preview_text = "\n".join(lines)
            if len(text_preview.splitlines()) > 20:
                preview_text += "\n…"
            lang = {"text/csv": "csv", "application/json": "json"}.get(mime, "")
            children.append({"type": "Markdown", "value": f"```{lang}\n{preview_text}\n```"})
        except Exception:
            pass

    children.append(
        {
            "type": "Button",
            "label": "Download",
            "style": "secondary",
            "onClickAction": {
                "type": "download_url",
                "handler": "client",
                "loadingBehavior": "none",
                "payload": {"url": inline_data_url, "filename": path.name},
            },
        }
    )

    try:
        await ctx.context.stream_widget(
            Card(
                size="lg",
                status={"type": "success", "title": card_title, "text": subtitle},
                children=children,
            )
        )
    except Exception as exc:
        return f"File saved at {path}. Could not render download widget: {exc}"

    return f"File available for download: {path.name}"


# ── Tool 9: run_parallel ──

class Experiment(BaseModel):
    """A single experiment for run_parallel."""
    name: str
    command: str
    timeout: int = 120


async def _run_one_shot(command: str, timeout: int = 120) -> dict[str, Any]:
    """Run a command in a throwaway subprocess (not PTY) for parallelism."""
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        output = (stdout + stderr).decode(errors="replace").strip()
        return {
            "exit": proc.returncode,
            "elapsed_ms": elapsed_ms,
            "output": output[:500],
        }
    except asyncio.TimeoutError:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {"exit": -1, "elapsed_ms": elapsed_ms, "output": "[timeout]"}
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {"exit": -1, "elapsed_ms": elapsed_ms, "output": f"[error: {exc}]"}


@function_tool
async def run_parallel(
    ctx: RunContextWrapper[AgentContext],  # type: ignore[type-arg]
    experiments: list[Experiment],
) -> str:
    """Run up to 8 bash commands concurrently and compare results.

    Returns a comparison table: name | exit | elapsed_ms | stdout_preview.

    Args:
        experiments: List of experiments, each with name, command, and optional timeout (default 120s).
    """
    if not experiments:
        return "Error: experiments list is empty"
    if len(experiments) > 8:
        return "Error: max 8 experiments"

    await _stream_progress(ctx, "square-code", f"Running {len(experiments)} experiments in parallel")

    tasks = [
        _run_one_shot(e.command, e.timeout)
        for e in experiments[:8]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    lines = ["| name | exit | elapsed_ms | stdout_preview |", "|------|------|------------|----------------|"]
    for exp, result in zip(experiments[:8], results):
        if isinstance(result, Exception):
            lines.append(f"| {exp.name} | -1 | — | [error: {result}] |")
        else:
            r: dict[str, Any] = result  # type: ignore[assignment]
            preview = r["output"].replace("\n", " ")[:80]
            lines.append(f"| {exp.name} | {r['exit']} | {r['elapsed_ms']} | {preview} |")

    return "\n".join(lines)


# ── Factory ──

def shell_tools() -> list[Any]:
    """Return all shell/filesystem tools for the coding agent."""
    return [bash, read_file, list_dir, edit_file, git, fetch_url, render_image, download_file, run_parallel]


__all__ = [
    "bash",
    "read_file",
    "list_dir",
    "edit_file",
    "git",
    "fetch_url",
    "render_image",
    "download_file",
    "run_parallel",
    "shell_tools",
]
