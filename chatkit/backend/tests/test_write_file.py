"""Unit tests for shell_tools.write_file — the heredoc-bypass path for
creating scripts without going through the PTY."""

from __future__ import annotations

from pathlib import Path

from app.tools.shell_tools import _is_inside, _WRITE_FILE_ROOTS


def test_write_file_roots_include_tmp_and_git() -> None:
    roots = {str(p) for p in _WRITE_FILE_ROOTS}
    # /tmp resolves to /private/tmp on macOS — either is fine
    assert any("tmp" in r for r in roots)
    assert any("git" in r for r in roots) or any(r.endswith(str(Path.home())) for r in roots)


def test_is_inside_accepts_subpath() -> None:
    root = Path("/tmp").resolve()
    assert _is_inside(Path("/tmp/foo/bar.py").resolve(), root)


def test_is_inside_rejects_outside_path() -> None:
    root = Path("/tmp").resolve()
    assert not _is_inside(Path("/etc/passwd").resolve(), root)


def test_write_file_tool_registered() -> None:
    from app.tools.shell_tools import shell_tools
    names = {t.name for t in shell_tools()}
    assert "write_file" in names
