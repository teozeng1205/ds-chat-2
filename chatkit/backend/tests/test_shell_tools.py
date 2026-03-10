"""Unit tests for shell_session.py and shell_tools.py (15 tests)."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Minimal stubs so we can import without the full chatkit package ──
# These MUST be set before any `from app.*` imports so the real
# investigation_tools import chain (→ attachment_store → chatkit.store)
# never executes.

# Pre-stub investigation_tools module so app/tools/__init__.py succeeds
_inv_stub = types.ModuleType("app.tools.investigation_tools")
_inv_stub.investigation_tools = lambda: []  # type: ignore[attr-defined]
_inv_stub.investigation_tools_core = lambda: []  # type: ignore[attr-defined]
sys.modules["app.tools.investigation_tools"] = _inv_stub

# Stub chatkit packages if not installed
for mod_name in [
    "chatkit",
    "chatkit.agents",
    "chatkit.types",
    "chatkit.widgets",
    "chatkit.server",
    "agents",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# agents stubs
agents_mod = sys.modules["agents"]
if not hasattr(agents_mod, "RunContextWrapper"):
    agents_mod.RunContextWrapper = MagicMock  # type: ignore[attr-defined]
if not hasattr(agents_mod, "function_tool"):
    agents_mod.function_tool = lambda f: f  # type: ignore[attr-defined]
if not hasattr(agents_mod, "Agent"):
    agents_mod.Agent = MagicMock  # type: ignore[attr-defined]

# chatkit.agents stub
chatkit_agents = sys.modules["chatkit.agents"]
if not hasattr(chatkit_agents, "AgentContext"):
    chatkit_agents.AgentContext = MagicMock  # type: ignore[attr-defined]

# chatkit.types stub — add ALL names used by investigation_tools.py
chatkit_types = sys.modules["chatkit.types"]
for name in ["ProgressUpdateEvent", "AttachmentCreateParams", "ThreadMetadata",
             "ThreadStreamEvent", "UserMessageItem", "UserMessageTagContent", "Attachment"]:
    if not hasattr(chatkit_types, name):
        setattr(chatkit_types, name, MagicMock)  # type: ignore[attr-defined]

# chatkit.widgets stub
chatkit_widgets = sys.modules["chatkit.widgets"]
if not hasattr(chatkit_widgets, "Card"):
    chatkit_widgets.Card = MagicMock  # type: ignore[attr-defined]

# chatkit.server stub
chatkit_server = sys.modules["chatkit.server"]
if not hasattr(chatkit_server, "ChatKitServer"):
    chatkit_server.ChatKitServer = MagicMock  # type: ignore[attr-defined]
if not hasattr(chatkit_server, "StreamingResult"):
    chatkit_server.StreamingResult = MagicMock  # type: ignore[attr-defined]

# httpx stub (for fetch_url — real package usually installed)
if "httpx" not in sys.modules:
    httpx_mod = types.ModuleType("httpx")
    sys.modules["httpx"] = httpx_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.investigation.shell_session import (  # noqa: E402
    PersistentShell,
    close_session,
    get_session,
    _registry,
)


# ── Helpers ──

def make_ctx(thread_id: str = "test-thread") -> MagicMock:
    ctx = MagicMock()
    ctx.context.thread.id = thread_id
    ctx.context.stream = AsyncMock()
    ctx.context.stream_widget = AsyncMock()
    return ctx


# ═══════════════════════════════════════════════════════════
# PersistentShell tests
# ═══════════════════════════════════════════════════════════

class TestPersistentShell:
    def test_start_creates_alive_process(self):
        """Shell starts and is_alive() returns True."""
        async def run():
            shell = PersistentShell("t1")
            await shell.start()
            assert shell.is_alive()
            shell.close()
        asyncio.run(run())

    def test_run_echo(self):
        """Simple echo command returns expected output."""
        async def run():
            shell = PersistentShell("t2")
            await shell.start()
            output = await shell.run("echo hello_world")
            shell.close()
            return output
        output = asyncio.run(run())
        assert "hello_world" in output

    def test_cd_persists(self):
        """cd in one run() is remembered in the next run()."""
        async def run():
            shell = PersistentShell("t3")
            await shell.start()
            await shell.run("cd /tmp")
            output = await shell.run("pwd")
            shell.close()
            return output
        output = asyncio.run(run())
        assert "/tmp" in output

    def test_export_persists(self):
        """export in one run() is visible in the next run()."""
        async def run():
            shell = PersistentShell("t4")
            await shell.start()
            await shell.run("export DSCHAT_TEST_VAR=42")
            output = await shell.run("echo $DSCHAT_TEST_VAR")
            shell.close()
            return output
        output = asyncio.run(run())
        assert "42" in output

    def test_last_cwd_updated(self):
        """last_cwd reflects the current directory after cd."""
        async def run():
            shell = PersistentShell("t5")
            await shell.start()
            await shell.run("cd /tmp")
            cwd = shell.last_cwd
            shell.close()
            return cwd
        cwd = asyncio.run(run())
        assert cwd == "/tmp"

    def test_close_marks_not_alive(self):
        """close() causes is_alive() to return False."""
        async def run():
            shell = PersistentShell("t6")
            await shell.start()
            shell.close()
            return shell.is_alive()
        alive = asyncio.run(run())
        assert not alive


# ═══════════════════════════════════════════════════════════
# bash tool tests
# ═══════════════════════════════════════════════════════════

class TestBashTool:
    def _get_bash(self):
        """Import bash after stubs are set up."""
        from app.tools.shell_tools import bash
        return bash

    def test_bash_success(self):
        """bash() runs a command and returns output."""
        bash = self._get_bash()
        ctx = make_ctx("bash-success")
        async def run():
            _registry.clear()
            return await bash(ctx, "echo hi")
        out = asyncio.run(run())
        assert "hi" in out

    def test_bash_session_reuse(self):
        """Two bash() calls in the same thread reuse the same session."""
        bash = self._get_bash()
        ctx = make_ctx("bash-reuse")
        async def run():
            _registry.clear()
            await bash(ctx, "export REUSE_CHECK=yes")
            out = await bash(ctx, "echo $REUSE_CHECK")
            return out
        out = asyncio.run(run())
        assert "yes" in out

    def test_bash_cwd_persist_e2e(self):
        """cd in one bash() call persists in the next call (E2E smoke)."""
        bash = self._get_bash()
        ctx = make_ctx("bash-cwd-e2e")
        async def run():
            _registry.clear()
            await bash(ctx, "cd /tmp")
            out = await bash(ctx, "pwd")
            return out
        out = asyncio.run(run())
        assert "/tmp" in out


# ═══════════════════════════════════════════════════════════
# read_file tool tests
# ═══════════════════════════════════════════════════════════

class TestReadFile:
    def _get_read_file(self):
        from app.tools.shell_tools import read_file
        return read_file

    def test_read_file_basic(self, tmp_path):
        """read_file returns file content with line numbers."""
        read_file = self._get_read_file()
        ctx = make_ctx()
        p = tmp_path / "test.txt"
        p.write_text("line1\nline2\nline3\n")
        async def run():
            return await read_file(ctx, str(p))
        out = asyncio.run(run())
        assert "line1" in out
        assert "line2" in out
        assert "1\t" in out  # line number present

    def test_read_file_pagination(self, tmp_path):
        """read_file respects offset and limit."""
        read_file = self._get_read_file()
        ctx = make_ctx()
        p = tmp_path / "long.txt"
        p.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        async def run():
            return await read_file(ctx, str(p), offset=5, limit=3)
        out = asyncio.run(run())
        assert "line5" in out
        assert "line7" in out
        assert "line8" not in out

    def test_read_file_missing(self):
        """read_file returns an error string for missing files."""
        read_file = self._get_read_file()
        ctx = make_ctx()
        async def run():
            return await read_file(ctx, "/nonexistent/path/file.txt")
        out = asyncio.run(run())
        assert "Error" in out


# ═══════════════════════════════════════════════════════════
# list_dir tool tests
# ═══════════════════════════════════════════════════════════

class TestListDir:
    def _get_list_dir(self):
        from app.tools.shell_tools import list_dir
        return list_dir

    def test_list_dir_basic(self, tmp_path):
        """list_dir lists directory entries."""
        list_dir = self._get_list_dir()
        ctx = make_ctx()
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        async def run():
            return await list_dir(ctx, str(tmp_path))
        out = asyncio.run(run())
        assert "a.txt" in out
        assert "b.py" in out

    def test_list_dir_glob_filter(self, tmp_path):
        """list_dir respects glob pattern filter."""
        list_dir = self._get_list_dir()
        ctx = make_ctx()
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        async def run():
            return await list_dir(ctx, str(tmp_path), pattern="*.py")
        out = asyncio.run(run())
        assert "b.py" in out
        assert "a.txt" not in out


# ═══════════════════════════════════════════════════════════
# edit_file tool tests
# ═══════════════════════════════════════════════════════════

class TestEditFile:
    def _get_edit_file(self):
        from app.tools.shell_tools import edit_file
        return edit_file

    def test_edit_file_unique_match(self, tmp_path):
        """edit_file succeeds when old_string appears exactly once."""
        edit_file = self._get_edit_file()
        ctx = make_ctx()
        p = tmp_path / "edit.txt"
        p.write_text("hello world\n")
        async def run():
            return await edit_file(ctx, str(p), "hello world", "goodbye world")
        out = asyncio.run(run())
        assert "OK" in out
        assert p.read_text() == "goodbye world\n"

    def test_edit_file_zero_match_error(self, tmp_path):
        """edit_file returns error when old_string not found."""
        edit_file = self._get_edit_file()
        ctx = make_ctx()
        p = tmp_path / "edit2.txt"
        p.write_text("hello world\n")
        async def run():
            return await edit_file(ctx, str(p), "not present", "x")
        out = asyncio.run(run())
        assert "Error" in out
        assert "not found" in out.lower() or "0 matches" in out.lower() or "read" in out.lower()

    def test_edit_file_multi_match_error(self, tmp_path):
        """edit_file returns error when old_string matches multiple times."""
        edit_file = self._get_edit_file()
        ctx = make_ctx()
        p = tmp_path / "edit3.txt"
        p.write_text("foo\nfoo\n")
        async def run():
            return await edit_file(ctx, str(p), "foo", "bar")
        out = asyncio.run(run())
        assert "Error" in out
        assert "2" in out


# ═══════════════════════════════════════════════════════════
# git tool tests
# ═══════════════════════════════════════════════════════════

class TestGitTool:
    def _get_git(self):
        from app.tools.shell_tools import git
        return git

    def test_git_log(self):
        """git log runs successfully in the project repo."""
        git_tool = self._get_git()
        ctx = make_ctx()
        repo_dir = str(Path(__file__).resolve().parents[3])  # ds-chat-2 root
        async def run():
            return await git_tool(ctx, "log --oneline -3", working_dir=repo_dir)
        out = asyncio.run(run())
        # Should return some commit lines
        assert len(out.strip()) > 0

    def test_git_blocked_force_push(self):
        """git push --force is blocked."""
        git_tool = self._get_git()
        ctx = make_ctx()
        async def run():
            return await git_tool(ctx, "push --force origin main")
        out = asyncio.run(run())
        assert "Error" in out
        assert "blocked" in out.lower()


# ═══════════════════════════════════════════════════════════
# run_parallel tool tests
# ═══════════════════════════════════════════════════════════

class TestRunParallel:
    def _get_run_parallel(self):
        from app.tools.shell_tools import run_parallel
        return run_parallel

    def test_run_parallel_two_commands(self):
        """run_parallel runs two commands and returns comparison table."""
        from app.tools.shell_tools import Experiment
        run_parallel = self._get_run_parallel()
        ctx = make_ctx()
        async def run():
            return await run_parallel(ctx, [
                Experiment(name="echo_a", command="echo aaa"),
                Experiment(name="echo_b", command="echo bbb"),
            ])
        out = asyncio.run(run())
        assert "echo_a" in out
        assert "echo_b" in out
        assert "|" in out  # table format

    def test_run_parallel_too_many(self):
        """run_parallel rejects more than 8 experiments."""
        from app.tools.shell_tools import Experiment
        run_parallel = self._get_run_parallel()
        ctx = make_ctx()
        async def run():
            return await run_parallel(ctx, [
                Experiment(name=f"e{i}", command="echo x") for i in range(9)
            ])
        out = asyncio.run(run())
        assert "Error" in out
        assert "8" in out


# ═══════════════════════════════════════════════════════════
# Session registry tests
# ═══════════════════════════════════════════════════════════

class TestSessionRegistry:
    def test_get_session_creates_new(self):
        """get_session() creates a new session for a new thread_id."""
        async def run():
            _registry.clear()
            shell = await get_session("reg-test-1")
            alive = shell.is_alive()
            close_session("reg-test-1")
            return alive
        assert asyncio.run(run())

    def test_get_session_reuses_existing(self):
        """get_session() returns the same object for the same thread_id."""
        async def run():
            _registry.clear()
            s1 = await get_session("reg-test-2")
            s2 = await get_session("reg-test-2")
            result = s1 is s2
            close_session("reg-test-2")
            return result
        assert asyncio.run(run())

    def test_close_session_removes_from_registry(self):
        """close_session() removes the session from the registry."""
        async def run():
            _registry.clear()
            await get_session("reg-test-3")
            close_session("reg-test-3")
            return "reg-test-3" in _registry
        assert not asyncio.run(run())
