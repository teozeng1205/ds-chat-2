"""Persistent PTY-backed bash session, one per conversation thread.

CWD, environment variables, and background processes survive across all
run() calls within a conversation thread — the same design as Codex's
exec_command PTY model.

Sentinel design: PROMPT_COMMAND is set to print a unique marker with CWD
after every command. This is more reliable than PS1 escape-sequence
expansion (which behaves differently across bash versions / macOS).
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import time
from pathlib import Path

SENTINEL = "__DSCHAT_READY__"
MAX_OUTPUT = 131072
SESSION_TTL = 3600  # 1 hour idle → auto-close

# Backend venv path — activate if present (safe no-op if absent)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_VENV_ACTIVATE = _BACKEND_ROOT / ".venv" / "bin" / "activate"

# PROMPT_COMMAND emits sentinel + CWD after every command; PS1 is cleared to
# avoid extra noise. Using printf to avoid `echo` portability issues.
#
# AWS metadata fast-fail: with no static creds in env, the AWS CLI falls back to
# the EC2 instance-metadata endpoint (169.254.169.254), which on a laptop hangs
# for minutes — making `aws ...` calls in the shell appear stuck. Cap it to a
# single ~1s attempt so `aws` errors immediately instead of hanging. Real creds
# are seeded into the process env before the shell spawns (see start()).
_INIT_CMD = (
    'export AWS_METADATA_SERVICE_NUM_ATTEMPTS=1 AWS_METADATA_SERVICE_TIMEOUT=1; '
    'export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"; '
    f'[ -f "{_VENV_ACTIVATE}" ] && source "{_VENV_ACTIVATE}"; '
    f'PROMPT_COMMAND=\'printf "{SENTINEL}:%s\\n" "$(pwd)"\'; '
    f'PS1=""\n'
)


class PersistentShell:
    """A PTY-backed bash session.

    CWD, env vars, and background processes persist across all run()
    calls within a conversation thread.
    """

    def __init__(self, thread_id: str, start_dir: Path | None = None) -> None:
        self.thread_id = thread_id
        self._start_dir = start_dir or Path.home()
        self._master_fd: int | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._last_used: float = time.monotonic()
        self.last_cwd: str = str(self._start_dir)

    async def start(self) -> None:
        # Seed 3VDEV credentials into the process env BEFORE spawning bash, so the
        # shell inherits working AWS creds (idempotent + cached). Without this the
        # shell's `aws` calls find no creds and hang on the IMDS fallback.
        try:
            from .runtime import get_runtime

            get_runtime().registry.ensure_credentials()
        except Exception:  # noqa: BLE001 — never block shell startup on cred bootstrap
            pass

        master_fd, slave_fd = pty.openpty()
        # Non-blocking reads on master so we can poll without blocking
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._proc = await asyncio.create_subprocess_exec(
            "bash", "--norc", "--noprofile",
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            cwd=str(self._start_dir),
            start_new_session=True,
        )
        os.close(slave_fd)
        self._master_fd = master_fd
        # Set PROMPT_COMMAND sentinel + clear PS1
        self._write(_INIT_CMD)
        await self._drain_to_sentinel(timeout=8)  # consume startup noise

    def _write(self, text: str) -> None:
        assert self._master_fd is not None
        os.write(self._master_fd, text.encode())

    def _parse_sentinel_line(self, line: str) -> str | None:
        """Extract CWD from a sentinel line. Returns CWD or None if not sentinel."""
        stripped = line.strip()
        prefix = SENTINEL + ":"
        if stripped.startswith(prefix):
            cwd = stripped[len(prefix):]
            if cwd:
                self.last_cwd = cwd
            return cwd
        return None

    async def _drain_to_sentinel(self, timeout: int) -> str:
        """Read PTY output until the SENTINEL line appears, return content before it."""
        assert self._master_fd is not None
        loop = asyncio.get_event_loop()
        buf = b""
        deadline = loop.time() + timeout
        sentinel_prefix = (SENTINEL + ":").encode()

        while loop.time() < deadline:
            try:
                buf += os.read(self._master_fd, 4096)
            except BlockingIOError:
                await asyncio.sleep(0.05)
                continue

            if sentinel_prefix in buf:
                text = buf.decode(errors="replace")
                lines = text.splitlines(keepends=True)

                # Find sentinel line index
                sentinel_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith(SENTINEL + ":"):
                        self._parse_sentinel_line(line)
                        sentinel_idx = i
                        break

                if sentinel_idx is not None:
                    # Skip first line (echoed command) + sentinel line + sentinel cmd echo
                    output_lines: list[str] = []
                    for i, line in enumerate(lines):
                        if i == 0:
                            continue  # echoed command
                        if i >= sentinel_idx:
                            break  # stop at sentinel
                        # Skip any line that is the echoed PROMPT_COMMAND setup
                        if "PROMPT_COMMAND" in line or line.strip().startswith(SENTINEL):
                            continue
                        output_lines.append(line)
                    return "".join(output_lines)

        return f"[timeout after {timeout}s — shell may be blocked]\n"

    async def run(self, command: str, timeout: int = 120) -> str:
        """Run a command and return combined stdout+stderr output."""
        self._last_used = time.monotonic()
        self._write(command + "\n")
        output = await self._drain_to_sentinel(timeout=min(timeout, 1800))
        # Tail to MAX_OUTPUT — most recent output is most relevant
        if len(output) > MAX_OUTPUT:
            output = (
                f"[...truncated, showing last {MAX_OUTPUT} chars...]\n"
                + output[-MAX_OUTPUT:]
            )
        return output

    async def run_streaming(self, command: str, timeout: int = 120):
        """Async generator that yields output chunks as they arrive.

        Completes when the SENTINEL is seen or timeout expires.
        """
        assert self._master_fd is not None
        self._last_used = time.monotonic()
        self._write(command + "\n")

        loop = asyncio.get_event_loop()
        buf = b""
        deadline = loop.time() + min(timeout, 1800)
        sentinel_prefix = (SENTINEL + ":").encode()
        first_line_consumed = False

        while loop.time() < deadline:
            try:
                chunk = os.read(self._master_fd, 4096)
                buf += chunk
            except BlockingIOError:
                await asyncio.sleep(0.05)
                continue

            if sentinel_prefix in buf:
                text = buf.decode(errors="replace")
                lines = text.splitlines(keepends=True)
                sentinel_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith(SENTINEL + ":"):
                        self._parse_sentinel_line(line)
                        sentinel_idx = i
                        break
                if sentinel_idx is not None:
                    start = 1 if not first_line_consumed else 0
                    result_lines = [
                        line for i, line in enumerate(lines)
                        if i >= start and i < sentinel_idx
                        and not ("PROMPT_COMMAND" in line or line.strip().startswith(SENTINEL))
                    ]
                    result = "".join(result_lines)
                    if result.strip():
                        yield result
                    return
            else:
                # Yield partial output
                text = buf.decode(errors="replace")
                lines = text.splitlines(keepends=True)
                if not first_line_consumed and len(lines) > 1:
                    lines = lines[1:]
                    first_line_consumed = True
                    buf = b""
                    partial = "".join(line for line in lines if not line.strip().startswith(SENTINEL))
                    if partial.strip():
                        yield partial
                        buf = b""

        yield f"[timeout after {timeout}s — shell may be blocked]\n"

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.kill()
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
        self._master_fd = None
        self._proc = None


# ── Module-level session registry ──

_registry: dict[str, PersistentShell] = {}
_lock = asyncio.Lock()


async def get_session(thread_id: str) -> PersistentShell:
    """Get or create the persistent shell session for a thread."""
    async with _lock:
        _evict_stale()
        shell = _registry.get(thread_id)
        if shell is None or not shell.is_alive():
            shell = PersistentShell(thread_id)
            await shell.start()
            _registry[thread_id] = shell
        return shell


def close_session(thread_id: str) -> None:
    """Close and remove the shell session for a thread."""
    shell = _registry.pop(thread_id, None)
    if shell:
        shell.close()


def _evict_stale() -> None:
    """Remove idle sessions older than SESSION_TTL seconds."""
    now = time.monotonic()
    stale = [
        tid for tid, s in list(_registry.items())
        if now - s._last_used > SESSION_TTL
    ]
    for tid in stale:
        shell = _registry.pop(tid, None)
        if shell:
            shell.close()


__all__ = ["PersistentShell", "get_session", "close_session"]
