"""Sandboxed tools for codebase explanation and general repo exploration."""

from __future__ import annotations

import datetime
import json
import logging
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from agents import (
    RunContextWrapper,
    ShellCallOutcome,
    ShellCommandOutput,
    ShellCommandRequest,
    ShellResult,
    ShellTool,
    WebSearchTool,
    function_tool,
)
from chatkit.agents import AgentContext
from chatkit.types import (
    AttachmentCreateParams,
    GeneratedImage,
    GeneratedImageItem,
    ProgressUpdateEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
)

from .attachment_store import LocalDiskAttachmentStore, default_attachment_dir

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)
log.propagate = False

SANDBOX_ROOT = Path("~/git").expanduser().resolve()
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300
MAX_OUTPUT_CHARS = 25_000
MAX_FILE_LINES = 800
MAX_LIST_RESULTS = 1000

BLOCKED_SHELL_PATTERNS = (
    r"(^|\s)sudo(\s|$)",
    r"(^|\s)su(\s|$)",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-",
    r"git\s+push(\s|$)",
    r"git\s+pull(\s|$)",
    r"git\s+rebase(\s|$)",
    r"(^|\s)rm(\s|$)",
    r"(^|\s)dd(\s|$)",
    r"(^|\s)mkfs(\s|$)",
    r"curl\s+.*\|\s*(bash|sh)",
    r"wget\s+.*\|\s*(bash|sh)",
    r"(^|\s)brew(\s|$)",
    r"(^|\s)apt(\s|$)",
    r"(^|\s)yum(\s|$)",
    r"(^|\s)dnf(\s|$)",
)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _truncate_text(value: str | None, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...(truncated)"


def _resolve_sandbox_path(
    raw_path: str | None,
    *,
    require_exists: bool = True,
    require_directory: bool = True,
) -> Path:
    if not SANDBOX_ROOT.exists():
        raise ValueError(f"Sandbox root does not exist: {SANDBOX_ROOT}")
    if not SANDBOX_ROOT.is_dir():
        raise ValueError(f"Sandbox root is not a directory: {SANDBOX_ROOT}")

    candidate = Path(raw_path or ".").expanduser()
    if not candidate.is_absolute():
        candidate = SANDBOX_ROOT / candidate

    resolved = candidate.resolve(strict=False)
    if resolved != SANDBOX_ROOT and SANDBOX_ROOT not in resolved.parents:
        raise ValueError(f"Path must stay under sandbox root {SANDBOX_ROOT}: {resolved}")

    if require_exists and not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")
    if require_directory and require_exists and not resolved.is_dir():
        raise ValueError(f"Expected a directory path: {resolved}")

    return resolved


def _sandbox_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(SANDBOX_ROOT),
    }
    if extra:
        env.update(extra)
    return env


def _validate_safe_shell_command(command: str) -> None:
    normalized = command.strip().lower()
    if not normalized:
        raise ValueError("command must not be empty")
    for pattern in BLOCKED_SHELL_PATTERNS:
        if re.search(pattern, normalized):
            raise ValueError(f"blocked command pattern: {pattern}")


def _run_subprocess(
    *,
    command: list[str],
    cwd: Path,
    timeout_seconds: int,
    stdin_text: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    clamped_timeout = _clamp(timeout_seconds, 1, MAX_TIMEOUT_SECONDS)
    try:
        completed = subprocess.run(
            command,
            input=stdin_text,
            cwd=str(cwd),
            env=_sandbox_env(extra_env),
            capture_output=True,
            text=True,
            timeout=clamped_timeout,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout": _truncate_text(completed.stdout),
            "stderr": _truncate_text(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "timed_out": True,
            "stdout": _truncate_text(exc.stdout),
            "stderr": _truncate_text(exc.stderr),
        }


def _format_shell_output(
    *,
    command: list[str],
    cwd: Path,
    returncode: int | None,
    timed_out: bool,
    stdout: str,
    stderr: str,
) -> str:
    return "\n".join(
        [
            f"command: {shlex.join(command)}",
            f"cwd: {cwd}",
            f"timed_out: {timed_out}",
            f"return_code: {returncode}",
            "stdout:",
            stdout or "(empty)",
            "stderr:",
            stderr or "(empty)",
        ]
    )


def _build_python_wrapper(user_code: str) -> str:
    dumped = json.dumps(user_code)
    return f"""
import builtins
import os
import pathlib
import shutil
import socket
import subprocess

_SANDBOX_ROOT = pathlib.Path(os.environ["SANDBOX_ROOT"]).resolve()
_ORIG_OPEN = builtins.open

def _blocked(*args, **kwargs):
    raise PermissionError("This operation is disabled in sandbox python mode.")

def _resolve(path_like):
    path = pathlib.Path(path_like).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    path = path.resolve()
    if path != _SANDBOX_ROOT and _SANDBOX_ROOT not in path.parents:
        raise PermissionError(f"Path outside sandbox root: {{path}}")
    return path

def _safe_open(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        raise PermissionError("Write modes are disabled in sandbox python mode.")
    return _ORIG_OPEN(_resolve(file), mode, *args, **kwargs)

builtins.open = _safe_open
pathlib.Path.write_text = _blocked
pathlib.Path.write_bytes = _blocked
pathlib.Path.unlink = _blocked
pathlib.Path.rmdir = _blocked
pathlib.Path.rename = _blocked
os.remove = _blocked
os.unlink = _blocked
os.rmdir = _blocked
os.rename = _blocked
os.replace = _blocked
os.open = _blocked
shutil.rmtree = _blocked
shutil.move = _blocked
subprocess.run = _blocked
subprocess.Popen = _blocked
socket.socket = _blocked

_user_code = {dumped}
_scope = {{}}
exec(compile(_user_code, "<sandbox-python>", "exec"), _scope, _scope)
"""


def _walk_tree(
    root: Path,
    *,
    max_depth: int,
    include_hidden: bool,
    limit: int,
) -> list[dict[str, str]]:
    base_parts = len(root.parts)
    entries: list[dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.parts) - base_parts
        if depth > max_depth:
            dirnames[:] = []
            continue

        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]

        for dirname in sorted(dirnames):
            rel = (current / dirname).relative_to(root)
            entries.append({"type": "dir", "path": str(rel)})
            if len(entries) >= limit:
                return entries
        for filename in sorted(filenames):
            rel = (current / filename).relative_to(root)
            entries.append({"type": "file", "path": str(rel)})
            if len(entries) >= limit:
                return entries
    return entries


async def _stream_progress(
    ctx: RunContextWrapper[AgentContext],
    icon: str,
    text: str,
) -> None:
    await ctx.context.stream(ProgressUpdateEvent(icon=icon, text=text))


async def _sandbox_shell_executor(request: ShellCommandRequest) -> ShellResult:
    timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    if request.data.action.timeout_ms is not None:
        timeout_seconds = _clamp(max(1, request.data.action.timeout_ms // 1000), 1, MAX_TIMEOUT_SECONDS)

    outputs: list[ShellCommandOutput] = []
    for command_text in request.data.action.commands:
        command_text = (command_text or "").strip()
        if not command_text:
            continue

        try:
            _validate_safe_shell_command(command_text)
        except Exception as exc:
            outputs.append(
                ShellCommandOutput(
                    stdout="",
                    stderr=f"Blocked by sandbox policy: {exc}",
                    outcome=ShellCallOutcome(type="exit", exit_code=1),
                    command=command_text,
                )
            )
            continue

        try:
            await request.ctx_wrapper.context.stream(
                ProgressUpdateEvent(icon="search", text=f"shell: {command_text}")
            )
        except Exception:
            pass

        result = _run_subprocess(
            command=["bash", "-lc", command_text],
            cwd=SANDBOX_ROOT,
            timeout_seconds=timeout_seconds,
        )
        outcome = (
            ShellCallOutcome(type="timeout", exit_code=None)
            if result["timed_out"]
            else ShellCallOutcome(type="exit", exit_code=result["returncode"])
        )
        outputs.append(
            ShellCommandOutput(
                stdout=result["stdout"],
                stderr=result["stderr"],
                outcome=outcome,
                command=command_text,
            )
        )

    if not outputs:
        outputs.append(
            ShellCommandOutput(
                stdout="(no command provided)",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
                command=None,
            )
        )

    return ShellResult(output=outputs, max_output_length=request.data.action.max_output_length)


_SANDBOX_SHELL_TOOL = ShellTool(executor=_sandbox_shell_executor, environment={"type": "local"})
_WEB_SEARCH_TOOL = WebSearchTool(search_context_size="medium")


def codebase_explainer_instructions(include_shell: bool = True) -> str:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    shell_line = (
        "Use shell for Codex-like command execution in the sandbox.\n"
        if include_shell
        else "shell is disabled for this model; use run_sandbox_command() and run_sandbox_python() instead.\n"
    )
    return (
        f"You are a codebase explanation assistant. Today is {current_date}.\n"
        f"Sandbox root is fixed to: {SANDBOX_ROOT}\n"
        "Use tools for answers and inspect code directly.\n"
        f"{shell_line}"
        "For codebase understanding, usually start with list_sandbox_repositories(), "
        "then list_directory_tree(), search_code(), and read_code_file().\n"
        "Use run_sandbox_command() or shell for shell diagnostics.\n"
        "Use run_sandbox_python() or shell with python for custom analysis.\n"
        "Use web_search to look up external docs, libraries, APIs, and recent web information when needed.\n"
        "If you generate a plot image file, use publish_plot_image(path=...) to show it in chat.\n"
        "Never run destructive commands and never assume files outside sandbox root.\n"
        "When explaining, cite concrete file paths, symbols, and control flow."
    )


@function_tool
async def list_sandbox_repositories(
    ctx: RunContextWrapper[AgentContext],
    under_path: str = ".",
    max_depth: int = 4,
    limit: int = 200,
) -> dict[str, Any]:
    """List git repositories under the sandbox root (~/git)."""
    await _stream_progress(ctx, "search", "Scanning sandbox for git repositories.")
    base = _resolve_sandbox_path(under_path, require_exists=True, require_directory=True)
    clamped_depth = _clamp(max_depth, 1, 10)
    clamped_limit = _clamp(limit, 1, MAX_LIST_RESULTS)

    root_parts = len(base.parts)
    repos: list[str] = []
    for dirpath, dirnames, _ in os.walk(base):
        current = Path(dirpath)
        depth = len(current.parts) - root_parts
        if depth > clamped_depth:
            dirnames[:] = []
            continue
        if ".git" in dirnames:
            repos.append(str(current))
            dirnames.remove(".git")
            if len(repos) >= clamped_limit:
                break

    await _stream_progress(
        ctx,
        "check-circle",
        f"Repository scan complete: {len(repos)} repositories found.",
    )
    return {
        "sandbox_root": str(SANDBOX_ROOT),
        "search_root": str(base),
        "repositories": repos,
        "truncated": len(repos) >= clamped_limit,
    }


@function_tool
async def list_directory_tree(
    ctx: RunContextWrapper[AgentContext],
    path: str = ".",
    max_depth: int = 3,
    limit: int = 300,
    include_hidden: bool = False,
) -> dict[str, Any]:
    """List directory and file entries under a sandbox path."""
    base = _resolve_sandbox_path(path, require_exists=True, require_directory=True)
    clamped_depth = _clamp(max_depth, 0, 10)
    clamped_limit = _clamp(limit, 1, MAX_LIST_RESULTS)
    await _stream_progress(ctx, "search", f"Listing directory tree for {base}.")
    entries = _walk_tree(
        base,
        max_depth=clamped_depth,
        include_hidden=include_hidden,
        limit=clamped_limit,
    )
    await _stream_progress(
        ctx,
        "check-circle",
        f"Directory listing complete: {len(entries)} entries.",
    )
    return {
        "path": str(base),
        "entries": entries,
        "truncated": len(entries) >= clamped_limit,
    }


@function_tool
async def read_code_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    start_line: int = 1,
    end_line: int = 220,
) -> dict[str, Any]:
    """Read a line range from a file under sandbox root."""
    file_path = _resolve_sandbox_path(path, require_exists=True, require_directory=False)
    if not file_path.is_file():
        raise ValueError(f"Expected file path: {file_path}")

    clamped_start = max(1, start_line)
    max_end = clamped_start + MAX_FILE_LINES - 1
    clamped_end = _clamp(end_line, clamped_start, max_end)
    await _stream_progress(
        ctx,
        "search",
        f"Reading {file_path} lines {clamped_start}-{clamped_end}.",
    )
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    chunk = "\n".join(lines[clamped_start - 1 : clamped_end])
    await _stream_progress(ctx, "check-circle", f"Read complete from {file_path.name}.")
    return {
        "path": str(file_path),
        "start_line": clamped_start,
        "end_line": clamped_end,
        "total_lines": len(lines),
        "content": _truncate_text(chunk),
    }


@function_tool
async def search_code(
    ctx: RunContextWrapper[AgentContext],
    pattern: str,
    path: str = ".",
    max_results: int = 200,
) -> dict[str, Any]:
    """Search code using ripgrep under sandbox root."""
    if not pattern.strip():
        raise ValueError("pattern must not be empty")
    base = _resolve_sandbox_path(path, require_exists=True, require_directory=True)
    clamped_results = _clamp(max_results, 1, MAX_LIST_RESULTS)
    await _stream_progress(ctx, "search", f"Searching code for pattern: {pattern!r}")

    if not shutil.which("rg"):
        raise RuntimeError("ripgrep (`rg`) is required for search_code")

    result = _run_subprocess(
        command=[
            "rg",
            "-n",
            "--no-heading",
            "--hidden",
            "--glob",
            "!.git",
            "--glob",
            "!node_modules",
            "-m",
            str(clamped_results),
            pattern,
            str(base),
        ],
        cwd=base,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    raw_lines = [line for line in result["stdout"].splitlines() if line.strip()]
    matches = raw_lines[:clamped_results]
    await _stream_progress(ctx, "check-circle", f"Code search complete: {len(matches)} matches.")
    return {
        "path": str(base),
        "pattern": pattern,
        "matches": matches,
        "returncode": result["returncode"],
        "stderr": result["stderr"],
        "truncated": len(raw_lines) > clamped_results,
    }


@function_tool
async def run_sandbox_command(
    ctx: RunContextWrapper[AgentContext],
    command: str,
    cwd: str = ".",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run a shell command inside sandbox root."""
    _validate_safe_shell_command(command)
    workdir = _resolve_sandbox_path(cwd, require_exists=True, require_directory=True)
    clamped_timeout = _clamp(timeout_seconds, 1, MAX_TIMEOUT_SECONDS)
    await _stream_progress(ctx, "search", f"Running sandbox command in {workdir}: {command}")
    result = _run_subprocess(
        command=["bash", "-lc", command],
        cwd=workdir,
        timeout_seconds=clamped_timeout,
    )
    if result["timed_out"]:
        await _stream_progress(ctx, "clock", "Sandbox command timed out.")
    elif result["returncode"] == 0:
        await _stream_progress(ctx, "check-circle", "Sandbox command completed successfully.")
    else:
        await _stream_progress(ctx, "info", f"Sandbox command exited with code {result['returncode']}.")
    return {
        "command": shlex.join(result["command"]),
        "cwd": result["cwd"],
        "timed_out": result["timed_out"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


@function_tool
async def run_sandbox_python(
    ctx: RunContextWrapper[AgentContext],
    code: str,
    cwd: str = ".",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run read-only Python code inside sandbox root."""
    if not code.strip():
        raise ValueError("code must not be empty")
    workdir = _resolve_sandbox_path(cwd, require_exists=True, require_directory=True)
    clamped_timeout = _clamp(timeout_seconds, 1, MAX_TIMEOUT_SECONDS)
    await _stream_progress(ctx, "search", f"Running sandbox python in {workdir}.")

    wrapper = _build_python_wrapper(code)
    result = _run_subprocess(
        command=[sys.executable, "-I", "-c", wrapper],
        cwd=workdir,
        timeout_seconds=clamped_timeout,
        extra_env={"SANDBOX_ROOT": str(SANDBOX_ROOT)},
    )
    if result["timed_out"]:
        await _stream_progress(ctx, "clock", "Sandbox python execution timed out.")
    elif result["returncode"] == 0:
        await _stream_progress(ctx, "check-circle", "Sandbox python completed successfully.")
    else:
        await _stream_progress(ctx, "info", f"Sandbox python exited with code {result['returncode']}.")
    return {
        "command": shlex.join(result["command"]),
        "cwd": result["cwd"],
        "timed_out": result["timed_out"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


@function_tool
async def publish_plot_image(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Publish an existing image file from sandbox root into the chat as an inline generated image item."""
    image_path = _resolve_sandbox_path(path, require_exists=True, require_directory=False)
    if not image_path.is_file():
        raise ValueError(f"Expected file path: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(
            f"File must be an image (png/jpg/webp/gif/svg). Got mime={mime_type!r} for {image_path}"
        )

    file_bytes = image_path.read_bytes()
    if not file_bytes:
        raise ValueError(f"Image file is empty: {image_path}")

    local_attachment_store = LocalDiskAttachmentStore(default_attachment_dir())
    attachment = await local_attachment_store.create_attachment(
        AttachmentCreateParams(
            name=(display_name or image_path.name),
            size=len(file_bytes),
            mime_type=mime_type,
        ),
        context=ctx.context.request_context,
    )
    await ctx.context.store.save_attachment(attachment, context=ctx.context.request_context)
    await local_attachment_store.write_attachment_bytes(attachment.id, file_bytes)

    image_url = getattr(attachment, "preview_url", None) or (
        attachment.upload_descriptor.url if attachment.upload_descriptor else None
    )
    if not image_url:
        raise RuntimeError("Failed to build image URL for published plot.")

    generated_item = GeneratedImageItem(
        id=ctx.context.generate_id("message"),
        thread_id=ctx.context.thread.id,
        created_at=datetime.datetime.now(),
        image=GeneratedImage(id=attachment.id, url=image_url),
    )
    await ctx.context.stream(ThreadItemAddedEvent(item=generated_item))
    await ctx.context.stream(ThreadItemDoneEvent(item=generated_item))
    await _stream_progress(ctx, "check-circle", f"Published image to chat: {image_path.name}")

    return {
        "published": True,
        "attachment_id": attachment.id,
        "image_url": image_url,
        "path": str(image_path),
        "mime_type": mime_type,
    }


def codebase_explainer_tools(include_shell: bool = True) -> list[Any]:
    tools: list[Any] = [_WEB_SEARCH_TOOL]
    if include_shell:
        tools.append(_SANDBOX_SHELL_TOOL)
    tools.extend(
        [
        list_sandbox_repositories,
        list_directory_tree,
        read_code_file,
            search_code,
            run_sandbox_command,
            run_sandbox_python,
            publish_plot_image,
        ]
    )
    return tools
