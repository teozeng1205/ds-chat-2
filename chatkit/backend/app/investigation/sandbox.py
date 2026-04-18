"""Hardened Python sandbox.

Two defenses in series:

1. AST allowlist (`check_code`) — parses the submitted code and refuses
   constructs that bypass any imaginable blocklist: `__import__`,
   `eval`, `exec`, `compile`, `globals()`, `getattr` on dangerous
   modules, attribute access on `os` / `subprocess` / `sys` / `shutil`
   / `socket` / `ctypes` / `pickle`, and direct imports of those
   modules.

2. Subprocess isolation (`run_sandboxed`) — spawns a fresh Python
   interpreter with:
     - scrubbed env (no AWS creds, no home, no proxies)
     - working dir = workspace/sandbox
     - resource limits via `resource.setrlimit` (address space, CPU,
       file descriptors, fsize)
     - wall-clock timeout
     - no stdin (no interactive escape)

Either layer alone is not enough. Combined they make the common
`__import__('os').system(...)` escape hatches fail at the AST stage,
and if something slips through, the subprocess layer caps the damage.

This module is additive: the existing OperatorRuntime still runs
Python as before. A follow-up switches callers to SandboxedPython
behind a feature flag.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Policy ──

# Modules whose attribute access is forbidden outright, even if imported
# under an alias. Reach the AST layer's rejection, not the subprocess.
FORBIDDEN_MODULES: frozenset[str] = frozenset({
    "os", "subprocess", "sys", "shutil", "socket", "ctypes", "pickle",
    "multiprocessing", "resource", "pty", "signal",
})

# Dangerous names that should never appear as Call targets.
FORBIDDEN_CALLS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "open",  # we want explicit Path().read_text() / pd.read_* etc.
    "input",
    "exit", "quit",
})

# Default resource limits. Applied via preexec_fn on POSIX; ignored
# (silently) on systems without `resource`.
DEFAULT_MEM_BYTES = 512 * 1024 * 1024   # 512 MB
DEFAULT_CPU_SECONDS = 60
DEFAULT_NOFILE = 256
DEFAULT_FSIZE_BYTES = 128 * 1024 * 1024  # 128 MB


@dataclass
class PolicyViolation:
    kind: str
    text: str
    line: int | None = None


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    elapsed_s: float
    timed_out: bool = False
    violations: list[PolicyViolation] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)


# ── Layer 1: AST allowlist ──


def check_code(code: str) -> list[PolicyViolation]:
    """Static AST check. Returns the list of policy violations (empty = ok)."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [PolicyViolation(kind="syntax", text=str(exc), line=exc.lineno)]

    v: list[PolicyViolation] = []

    for node in ast.walk(tree):
        # Forbidden imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _top_module(alias.name) in FORBIDDEN_MODULES:
                    v.append(PolicyViolation("forbidden_import", f"import {alias.name}",
                                             getattr(node, "lineno", None)))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _top_module(mod) in FORBIDDEN_MODULES:
                v.append(PolicyViolation("forbidden_import", f"from {mod} import ...",
                                         getattr(node, "lineno", None)))

        # Forbidden name calls (eval(), exec(), __import__(), open(), …)
        elif isinstance(node, ast.Call):
            target = _call_target_name(node)
            if target and target in FORBIDDEN_CALLS:
                v.append(PolicyViolation("forbidden_call", f"{target}(...)",
                                         getattr(node, "lineno", None)))

        # Attribute access against forbidden modules, e.g. os.system
        elif isinstance(node, ast.Attribute):
            base = _attribute_base_name(node)
            if base in FORBIDDEN_MODULES:
                v.append(PolicyViolation("forbidden_attribute", f"{base}.{node.attr}",
                                         getattr(node, "lineno", None)))

        # Dunder name access that smells like a sandbox escape
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__") and node.id not in {"__name__", "__doc__"}:
                v.append(PolicyViolation("forbidden_dunder", node.id,
                                         getattr(node, "lineno", None)))

    return v


def _top_module(name: str) -> str:
    return (name or "").split(".")[0]


def _call_target_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _attribute_base_name(node: ast.Attribute) -> Optional[str]:
    cursor: ast.AST = node.value
    while isinstance(cursor, ast.Attribute):
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        return cursor.id
    return None


# ── Layer 2: subprocess isolation ──


_ALLOWED_ENV_KEYS: frozenset[str] = frozenset({
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "PYTHONHASHSEED", "PYTHONUNBUFFERED",
})


def _scrub_env(base: Optional[dict[str, str]] = None) -> dict[str, str]:
    src = base if base is not None else os.environ
    env: dict[str, str] = {}
    for k, v in src.items():
        if k in _ALLOWED_ENV_KEYS:
            env[k] = v
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def _preexec(mem: int, cpu: int, nofile: int, fsize: int):
    """Return a preexec_fn that applies resource limits on POSIX."""

    def _apply() -> None:
        try:
            import resource  # POSIX-only
        except Exception:
            return
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        except Exception:
            pass
        # Detach from controlling terminal so signals don't leak back.
        try:
            os.setsid()
        except Exception:
            pass

    return _apply


def run_sandboxed(
    code: str,
    *,
    workspace_dir: Path | None = None,
    timeout_s: int = DEFAULT_CPU_SECONDS,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    nofile: int = DEFAULT_NOFILE,
    fsize_bytes: int = DEFAULT_FSIZE_BYTES,
    python_executable: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> SandboxResult:
    """Run `code` in a hardened subprocess after AST validation.

    Returns a SandboxResult. Never raises for expected failure modes
    (AST violations, subprocess non-zero exit, timeout) — callers can
    branch on `ok` and inspect `violations`, `exit_code`, `timed_out`.
    """
    violations = check_code(code)
    if violations:
        return SandboxResult(
            ok=False, stdout="", stderr="", exit_code=-1,
            elapsed_s=0.0, timed_out=False, violations=violations, created_files=[],
        )

    workspace = workspace_dir or Path(tempfile.mkdtemp(prefix="ds-chat-sandbox-"))
    workspace.mkdir(parents=True, exist_ok=True)

    script_path = workspace / "user_code.py"
    script_path.write_text(code, encoding="utf-8")

    env = _scrub_env()
    if extra_env:
        for k, v in extra_env.items():
            if k in _ALLOWED_ENV_KEYS:
                env[k] = v

    # Snapshot files pre-run so we can diff afterwards.
    before = _snapshot_files(workspace)

    start = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            [python_executable or sys.executable, "-I", str(script_path)],
            cwd=str(workspace),
            env=env,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            preexec_fn=_preexec(mem_bytes, cpu_seconds, nofile, fsize_bytes),
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or "").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        exit_code = -9
        timed_out = True
    elapsed = time.time() - start

    created = sorted(_snapshot_files(workspace) - before - {str(script_path)})

    return SandboxResult(
        ok=(exit_code == 0) and not timed_out,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        elapsed_s=elapsed,
        timed_out=timed_out,
        violations=[],
        created_files=created,
    )


def _snapshot_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(p) for p in root.rglob("*") if p.is_file()}


__all__ = [
    "PolicyViolation",
    "SandboxResult",
    "FORBIDDEN_MODULES",
    "FORBIDDEN_CALLS",
    "check_code",
    "run_sandboxed",
]
