"""Unit tests for LocalApplyPatchEditor.

Exercise create_file / update_file / delete_file directly so the tests
don't need to boot the Agents SDK.
"""

from __future__ import annotations

from pathlib import Path

from agents import ApplyPatchOperation

from app.tools.apply_patch import LocalApplyPatchEditor


def _op(op_type: str, path: Path, diff: str | None = None) -> ApplyPatchOperation:
    return ApplyPatchOperation(type=op_type, path=str(path), diff=diff)


def test_create_file_writes_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    editor = LocalApplyPatchEditor()
    new = tmp_path / "new.py"
    result = editor.create_file(_op("create_file", new, diff="print('hello')\n"))
    assert result.status == "completed"
    assert new.read_text() == "print('hello')\n"


def test_create_file_fails_when_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    p = tmp_path / "exists.py"
    p.write_text("pass\n")
    editor = LocalApplyPatchEditor()
    result = editor.create_file(_op("create_file", p, diff="new body"))
    assert result.status == "failed"
    assert "already exists" in (result.output or "")


def test_create_file_rejects_outside_allowlist(tmp_path: Path) -> None:
    # Don't monkey-patch ALLOWED_ROOTS: /etc/notes_new must be rejected
    editor = LocalApplyPatchEditor()
    result = editor.create_file(_op("create_file", Path("/etc/notes_new.md"), diff="x"))
    assert result.status == "failed"
    assert "outside allowed roots" in (result.output or "")


def test_create_file_rejects_sensitive_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    editor = LocalApplyPatchEditor()
    result = editor.create_file(_op("create_file", tmp_path / ".env", diff="TOKEN=x\n"))
    assert result.status == "failed"
    assert "blocked by guardrails" in (result.output or "")


def test_update_file_applies_single_hunk(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    p = tmp_path / "mod.py"
    p.write_text("a\nb\nc\n")
    diff = """@@ -1,3 +1,3 @@
 a
-b
+B
 c
"""
    editor = LocalApplyPatchEditor()
    result = editor.update_file(_op("update_file", p, diff=diff))
    assert result.status == "completed", result.output
    assert p.read_text() == "a\nB\nc\n"


def test_update_file_multi_hunk(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    p = tmp_path / "multi.py"
    p.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n")
    diff = """@@ -1,2 +1,2 @@
-alpha
+ALPHA
 beta
@@ -4,2 +4,2 @@
 delta
-epsilon
+EPSILON
"""
    editor = LocalApplyPatchEditor()
    result = editor.update_file(_op("update_file", p, diff=diff))
    assert result.status == "completed", result.output
    assert p.read_text() == "ALPHA\nbeta\ngamma\ndelta\nEPSILON\n"


def test_update_file_fails_when_context_not_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    p = tmp_path / "miss.py"
    p.write_text("x\n")
    diff = """@@ -1,1 +1,1 @@
-NOT_IN_FILE
+replacement
"""
    editor = LocalApplyPatchEditor()
    result = editor.update_file(_op("update_file", p, diff=diff))
    assert result.status == "failed"
    assert "context not found" in (result.output or "")


def test_update_file_rejects_sensitive_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    p = tmp_path / "secret_token.txt"
    p.write_text("old\n")
    diff = """@@ -1,1 +1,1 @@
-old
+new
"""
    result = LocalApplyPatchEditor().update_file(_op("update_file", p, diff=diff))
    assert result.status == "failed"
    assert "blocked by guardrails" in (result.output or "")


def test_delete_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    p = tmp_path / "goodbye.py"
    p.write_text("bye\n")
    editor = LocalApplyPatchEditor()
    result = editor.delete_file(_op("delete_file", p))
    assert result.status == "failed"
    assert "requires explicit approval" in (result.output or "")
    assert p.exists()


def test_delete_file_missing_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.apply_patch._ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    result = LocalApplyPatchEditor().delete_file(
        _op("delete_file", tmp_path / "nope.py"),
    )
    assert result.status == "failed"
    assert "requires explicit approval" in (result.output or "")


def test_apply_patch_tool_registers_with_editor() -> None:
    """apply_patch_tool() returns a hosted ApplyPatchTool bound to
    LocalApplyPatchEditor — the registration path the agent uses."""
    from app.tools.apply_patch import apply_patch_tool
    [tool] = apply_patch_tool()
    # The SDK object exposes the editor as the first dataclass field. Check
    # by class name because test_shell_tools may reload this module in the
    # same pytest process while stubbing the Agents SDK.
    assert tool.editor.__class__.__name__ == LocalApplyPatchEditor.__name__
