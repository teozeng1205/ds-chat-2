"""Host-side editor for OpenAI Agents SDK's hosted `ApplyPatchTool`.

The hosted tool lets the model request multi-hunk file mutations via
unified diffs. The model emits `apply_patch` operations and the SDK
calls this editor's `create_file` / `update_file` / `delete_file`
methods. We apply the diff locally with the same path-safety rules
the `write_file` / `edit_file` tools use and return an
`ApplyPatchResult` so the SDK can report success / failure.

Reference: https://openai.github.io/openai-agents-python/tools/
Why it matters: per Aider's edit-format benchmark, a single-round
multi-hunk patch outperforms repeated `str_replace` round-trips on
large refactors.
"""

from __future__ import annotations

import difflib
import logging
import re
import tempfile
from pathlib import Path
from typing import Iterable

from agents import ApplyPatchOperation, ApplyPatchResult

log = logging.getLogger(__name__)


# Path allowlist — same roots that `write_file` uses.
_ALLOWED_ROOTS: tuple[Path, ...] = (
    Path("/tmp").resolve(),
    Path(tempfile.gettempdir()).resolve(),
    Path("~/git").expanduser().resolve(),
    Path("~/.work").expanduser().resolve(),
    Path.cwd().resolve(),
)


def _path_allowed(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    for root in _ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path("~/git").expanduser() / p
    return p.resolve()


# ── Diff application ───────────────────────────────────────────────────
#
# The hosted `apply_patch` tool emits a simplified unified-diff dialect
# (same one Aider / OpenAI Codex use). Rather than depending on the
# system `patch` binary we implement a minimal applier. This supports
# the common case — file with one or more @@ hunks containing `+`, `-`,
# and ` ` (context) lines — which is what the model produces.


_HUNK_HEADER = re.compile(r"^@@\s.*?@@")


def _split_hunks(diff_text: str) -> list[list[str]]:
    """Split a unified diff body into hunks (each @@ section's body)."""
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            current = []
            continue
        # Skip patch-header noise (--- a/..., +++ b/...).
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if current is None:
            continue
        current.append(line)
    if current is not None:
        hunks.append(current)
    return hunks


def _apply_hunk(original_lines: list[str], hunk_lines: list[str]) -> list[str]:
    """Apply one hunk. Matches the hunk's context+remove lines against
    the original, then substitutes in the hunk's context+add lines.

    Raises ValueError if the hunk's context can't be located uniquely.
    """
    # Build the "before" (context + removed) and "after" (context + added) blocks.
    before: list[str] = []
    after: list[str] = []
    for raw in hunk_lines:
        if not raw:
            before.append("")
            after.append("")
            continue
        marker, payload = raw[0], raw[1:]
        if marker == " ":
            before.append(payload)
            after.append(payload)
        elif marker == "-":
            before.append(payload)
        elif marker == "+":
            after.append(payload)
        else:
            # Unknown marker — treat as context for robustness
            before.append(raw)
            after.append(raw)

    original_joined = "\n".join(original_lines)
    before_joined = "\n".join(before)
    if not before:
        # Pure-addition hunk → append to end
        return original_lines + after

    count = original_joined.count(before_joined)
    if count == 0:
        raise ValueError(
            f"apply_patch hunk context not found in file (hunk_len={len(before)})"
        )
    if count > 1:
        raise ValueError(
            f"apply_patch hunk context matched {count} places — need more context"
        )

    new_joined = original_joined.replace(before_joined, "\n".join(after), 1)
    return new_joined.split("\n")


def _apply_diff(original_text: str, diff_text: str) -> str:
    hunks = _split_hunks(diff_text)
    lines = original_text.split("\n") if original_text else []
    # Drop the trailing empty split artifact when the file ended with a newline
    trailing_nl = original_text.endswith("\n")
    if trailing_nl and lines and lines[-1] == "":
        lines = lines[:-1]
    for hunk in hunks:
        lines = _apply_hunk(lines, hunk)
    body = "\n".join(lines)
    if trailing_nl and not body.endswith("\n"):
        body += "\n"
    return body


def _unified_diff(a: str, b: str, name: str) -> str:
    return "\n".join(difflib.unified_diff(
        a.splitlines(),
        b.splitlines(),
        fromfile=f"a/{name}",
        tofile=f"b/{name}",
        lineterm="",
    ))


# ── Editor ─────────────────────────────────────────────────────────────


class LocalApplyPatchEditor:
    """Protocol-compliant editor for `ApplyPatchTool`.

    Host-side implementation that:
      - enforces the same path allowlist as `write_file`,
      - applies unified-diff hunks to create / update files,
      - returns a short `ApplyPatchResult.output` the model can read
        to confirm the edit landed.
    """

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        path = _resolve(operation.path)
        if not _path_allowed(path):
            return ApplyPatchResult(
                status="failed",
                output=f"Error: path {path} is outside allowed roots.",
            )
        if path.exists():
            return ApplyPatchResult(
                status="failed",
                output=f"Error: {path} already exists — use update_file.",
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # For create, the model's diff body is the file content. We
            # accept either a raw body or an all-`+` hunk and strip the
            # leading markers if present.
            body = operation.diff or ""
            if body.lstrip().startswith("@@") or body.lstrip().startswith("---"):
                body = _apply_diff("", body)
            path.write_text(body, encoding="utf-8")
            return ApplyPatchResult(
                status="completed",
                output=f"Created {path} ({len(body)} chars).",
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("apply_patch create_file failed")
            return ApplyPatchResult(status="failed", output=f"Error: {exc}")

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        path = _resolve(operation.path)
        if not _path_allowed(path):
            return ApplyPatchResult(
                status="failed",
                output=f"Error: path {path} is outside allowed roots.",
            )
        if not path.exists() or not path.is_file():
            return ApplyPatchResult(
                status="failed",
                output=f"Error: {path} does not exist or is not a regular file.",
            )
        try:
            original = path.read_text(encoding="utf-8")
            updated = _apply_diff(original, operation.diff or "")
            path.write_text(updated, encoding="utf-8")
            short_diff = _unified_diff(original, updated, path.name)
            preview = short_diff[:1000] + ("…" if len(short_diff) > 1000 else "")
            return ApplyPatchResult(
                status="completed",
                output=f"Updated {path} ({len(updated) - len(original):+d} chars).\n{preview}",
            )
        except ValueError as exc:
            # Hunk context mismatch — surface the specific reason so
            # the model can re-read and retry with better context.
            return ApplyPatchResult(status="failed", output=f"Error: {exc}")
        except Exception as exc:  # noqa: BLE001
            log.exception("apply_patch update_file failed")
            return ApplyPatchResult(status="failed", output=f"Error: {exc}")

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        path = _resolve(operation.path)
        if not _path_allowed(path):
            return ApplyPatchResult(
                status="failed",
                output=f"Error: path {path} is outside allowed roots.",
            )
        if not path.exists():
            return ApplyPatchResult(
                status="failed",
                output=f"Error: {path} does not exist.",
            )
        try:
            path.unlink()
            return ApplyPatchResult(
                status="completed",
                output=f"Deleted {path}.",
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("apply_patch delete_file failed")
            return ApplyPatchResult(status="failed", output=f"Error: {exc}")


def apply_patch_tool() -> "Iterable":  # type: ignore[override]
    """Return the hosted ApplyPatchTool wired to LocalApplyPatchEditor."""
    from agents import ApplyPatchTool
    return [ApplyPatchTool(editor=LocalApplyPatchEditor())]


__all__ = [
    "LocalApplyPatchEditor",
    "apply_patch_tool",
]
