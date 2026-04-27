"""Guardrails for local command and file tools.

The first-pass policy is intentionally conservative. The current ChatKit
tooling has no user-approval resume flow for function tools, so operations
classified as ``approval_required`` are blocked until a future approval path
is added.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Decision = Literal["allow", "approval_required", "deny"]


@dataclass(frozen=True)
class GuardrailResult:
    decision: Decision
    reason: str
    matched: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


_COMMAND_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\n)\s*")
_SENSITIVE_NAME_RE = re.compile(
    r"(^|/)(\.env($|\.)|\.aws/credentials$|\.aws/config$|id_rsa$|id_ed25519$|"
    r".*secret.*|.*credential.*|.*token.*)",
    re.IGNORECASE,
)

_DENY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\brm\s+(-[^\s]*r[^\s]*f|-*[^\s]*f[^\s]*r)\s+(/|~|\$HOME)(\s|$)"), "destructive rm against a broad root"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard is destructive"),
    (re.compile(r"\bgit\s+clean\s+-[^\s]*f"), "git clean -f is destructive"),
    (re.compile(r"\bgit\s+push\b.*\s--force(?:-with-lease)?\b|\bgit\s+push\s+-f\b"), "force push is blocked"),
    (re.compile(r"\b(?:mkfs|shutdown|reboot|poweroff|halt)\b"), "host-destructive system command"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "raw device write is blocked"),
    (re.compile(r"\bchmod\s+-R\s+777\s+(/|~|\$HOME)\b"), "broad chmod is blocked"),
    (re.compile(r"\bchown\s+-R\b.*\s(/|~|\$HOME)\b"), "broad chown is blocked"),
)

_WRITE_COMMANDS = {
    "cat",
    "cp",
    "install",
    "mkdir",
    "mv",
    "npm",
    "pip",
    "python",
    "python3",
    "rm",
    "tee",
    "touch",
    "uv",
}

_AWS_READ_ONLY_PREFIXES = (
    "list",
    "describe",
    "get",
    "head",
    "batch-get",
)

_AWS_APPROVAL_PREFIXES = (
    "start",
    "run",
    "invoke",
    "put",
    "update",
    "create",
    "delete",
    "stop",
    "terminate",
    "cancel",
    "send",
    "publish",
    "execute",
    "submit",
)

_AWS_ALLOWED_SPECIALS = {
    ("logs", "start-query"),
    ("logs", "get-query-results"),
    ("s3", "ls"),
    ("s3api", "list-objects"),
    ("s3api", "list-objects-v2"),
    ("s3api", "head-object"),
    ("s3api", "get-object"),
    ("sts", "get-caller-identity"),
}


def _segments(command: str) -> list[str]:
    return [part.strip() for part in _COMMAND_SPLIT_RE.split(command or "") if part.strip()]


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _strip_env_assignments(tokens: list[str]) -> list[str]:
    idx = 0
    while idx < len(tokens) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[idx]):
        idx += 1
    return tokens[idx:]


def _aws_profile_decision(tokens: list[str]) -> GuardrailResult | None:
    for idx, token in enumerate(tokens):
        if token == "--profile" and idx + 1 < len(tokens):
            profile = tokens[idx + 1]
            if profile != "3VDEV":
                return GuardrailResult("deny", "AWS profile switching is blocked unless explicitly implemented with approval", profile)
        elif token.startswith("--profile="):
            profile = token.split("=", 1)[1]
            if profile != "3VDEV":
                return GuardrailResult("deny", "AWS profile switching is blocked unless explicitly implemented with approval", profile)
    return None


def classify_aws_command(tokens: list[str]) -> GuardrailResult:
    """Classify one tokenized AWS CLI command."""
    try:
        aws_index = tokens.index("aws")
    except ValueError:
        return GuardrailResult("allow", "not an aws command")

    aws_tokens = tokens[aws_index + 1 :]
    profile = _aws_profile_decision(aws_tokens)
    if profile is not None:
        return profile

    if len(aws_tokens) >= 2 and aws_tokens[0] == "sts" and aws_tokens[1] == "assume-role":
        return GuardrailResult("deny", "AWS role assumption is blocked by default", "sts assume-role")

    if len(aws_tokens) < 2:
        return GuardrailResult("allow", "aws command without service operation")

    service, operation = aws_tokens[0], aws_tokens[1]
    op_pair = (service, operation)
    if op_pair in _AWS_ALLOWED_SPECIALS:
        return GuardrailResult("allow", "AWS read-only inspection command", f"{service} {operation}")

    if operation.startswith(_AWS_READ_ONLY_PREFIXES):
        return GuardrailResult("allow", "AWS read-only inspection command", f"{service} {operation}")

    if operation.startswith(_AWS_APPROVAL_PREFIXES):
        return GuardrailResult(
            "approval_required",
            "AWS mutation, execution, task, or publish operations require explicit approval",
            f"{service} {operation}",
        )

    return GuardrailResult(
        "approval_required",
        "Unrecognized AWS operation requires explicit approval",
        f"{service} {operation}",
    )


def classify_shell_command(command: str) -> GuardrailResult:
    """Classify a shell command before execution."""
    normalized = command.strip()
    if not normalized:
        return GuardrailResult("allow", "empty command")

    lowered = normalized.lower()
    if re.search(r"\bassume\s+3v(?!dev\b)[a-z0-9_-]*", lowered):
        return GuardrailResult("deny", "AWS account/profile switching is blocked by default", "assume")

    for pattern, reason in _DENY_PATTERNS:
        match = pattern.search(lowered)
        if match:
            return GuardrailResult("deny", reason, match.group(0))

    for segment in _segments(normalized):
        tokens = _strip_env_assignments(_tokens(segment))
        if not tokens:
            continue
        if "aws" in tokens:
            aws_decision = classify_aws_command(tokens)
            if aws_decision.decision != "allow":
                return aws_decision
        first = Path(tokens[0]).name
        if first == "git" and len(tokens) > 1 and tokens[1] in {"commit", "push", "tag"}:
            return GuardrailResult("approval_required", "Git publishing/history operations require approval", f"git {tokens[1]}")
        if first in {"npm", "pnpm", "yarn"} and any(tok in {"install", "add", "remove", "run"} for tok in tokens[1:]):
            return GuardrailResult("approval_required", "Package install or script execution requires approval", first)
        if first in {"pip", "pip3", "uv"} and any(tok in {"install", "add", "remove", "sync"} for tok in tokens[1:]):
            return GuardrailResult("approval_required", "Python package environment mutation requires approval", first)
        if first in _WRITE_COMMANDS and _looks_like_shell_write(segment, tokens):
            return GuardrailResult("approval_required", "Shell file writes should use guarded file/patch tools or explicit approval", first)

    return GuardrailResult("allow", "command classified as read-only or low-risk")


def _looks_like_shell_write(segment: str, tokens: list[str]) -> bool:
    if re.search(r"(^|[^>])>{1,2}[^>]", segment) or "|" in segment and re.search(r"\btee\b", segment):
        return True
    first = Path(tokens[0]).name if tokens else ""
    if first in {"touch", "mkdir", "cp", "mv", "rm", "install"}:
        return True
    if first == "cat" and any(tok.startswith(">") for tok in tokens[1:]):
        return True
    return False


def classify_path_write(path: Path | str, *, operation: str = "write") -> GuardrailResult:
    """Classify a file write/edit/patch path."""
    resolved = Path(path).expanduser()
    text = resolved.as_posix()
    if _SENSITIVE_NAME_RE.search(text):
        return GuardrailResult("deny", f"{operation} to sensitive credential/secret path is blocked", text)
    return GuardrailResult("allow", f"{operation} path is allowed", text)

