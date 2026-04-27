"""Unit tests for shell_session.py and shell_tools.py (15 tests)."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Save + stub + restore sys.modules ──
#
# These tests call `bash(ctx, ...)` etc. directly, which requires
# `agents.function_tool` to be a pass-through identity decorator so
# the tools aren't wrapped as FunctionTool objects.
#
# To keep the stubs scoped to THIS test module (so they don't break
# test_semantic_search_wiring / test_execute_sql_cache / etc. in the
# same pytest run), we snapshot the modules we're about to mutate and
# restore them at teardown.

_STUBBED_MODULE_NAMES = (
    "app.tools.investigation_tools",
    "app.tools.shell_tools",
    "chatkit",
    "chatkit.agents",
    "chatkit.types",
    "chatkit.widgets",
    "chatkit.server",
    "agents",
    "httpx",
)
_ORIGINAL_MODULES: dict[str, object] = {
    name: sys.modules[name] for name in _STUBBED_MODULE_NAMES if name in sys.modules
}
_NEWLY_STUBBED: set[str] = set()


def _stub_module(name: str) -> types.ModuleType:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
        _NEWLY_STUBBED.add(name)
    return sys.modules[name]  # type: ignore[return-value]


# Force a stub for app.tools.investigation_tools so app/tools/__init__.py
# can import without pulling in chatkit.store. Preserve any real module
# via _ORIGINAL_MODULES so the teardown puts it back.
_inv_stub = types.ModuleType("app.tools.investigation_tools")
_inv_stub.investigation_tools = lambda: []  # type: ignore[attr-defined]
_inv_stub.investigation_tools_core = lambda: []  # type: ignore[attr-defined]
sys.modules["app.tools.investigation_tools"] = _inv_stub

# Stub chatkit packages if not installed (most are installed in the venv —
# we only create a stub when missing, so tests can run on a lean machine).
for _n in ("chatkit", "chatkit.agents", "chatkit.types", "chatkit.widgets",
           "chatkit.server", "agents"):
    _stub_module(_n)

agents_mod = sys.modules["agents"]
_AGENTS_ORIG_FN_TOOL = getattr(agents_mod, "function_tool", None)
# Override function_tool so @function_tool(...) returns the raw callable —
# tests below call tools as plain async functions.
def _identity_function_tool(func=None, **_kwargs):
    if func is None:
        return lambda wrapped: wrapped
    return func


agents_mod.function_tool = _identity_function_tool  # type: ignore[attr-defined]
if not hasattr(agents_mod, "RunContextWrapper"):
    agents_mod.RunContextWrapper = MagicMock  # type: ignore[attr-defined]
if not hasattr(agents_mod, "Agent"):
    agents_mod.Agent = MagicMock  # type: ignore[attr-defined]

chatkit_agents = sys.modules["chatkit.agents"]
if not hasattr(chatkit_agents, "AgentContext"):
    chatkit_agents.AgentContext = MagicMock  # type: ignore[attr-defined]

chatkit_types = sys.modules["chatkit.types"]
for _name in ("ProgressUpdateEvent", "AttachmentCreateParams", "ThreadMetadata",
              "ThreadStreamEvent", "UserMessageItem", "UserMessageTagContent", "Attachment"):
    if not hasattr(chatkit_types, _name):
        setattr(chatkit_types, _name, MagicMock)  # type: ignore[attr-defined]

chatkit_widgets = sys.modules["chatkit.widgets"]
if not hasattr(chatkit_widgets, "Card"):
    chatkit_widgets.Card = MagicMock  # type: ignore[attr-defined]

chatkit_server = sys.modules["chatkit.server"]
if not hasattr(chatkit_server, "ChatKitServer"):
    chatkit_server.ChatKitServer = MagicMock  # type: ignore[attr-defined]
if not hasattr(chatkit_server, "StreamingResult"):
    chatkit_server.StreamingResult = MagicMock  # type: ignore[attr-defined]

_stub_module("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Drop any previously-cached version of shell_tools / shell_session so the
# upcoming import picks up our stubbed agents.function_tool.
for _m in ("app.tools.shell_tools", "app.investigation.shell_session"):
    sys.modules.pop(_m, None)

from app.investigation.shell_session import (  # noqa: E402
    PersistentShell,
    close_session,
    get_session,
    _registry,
)


@pytest.fixture(scope="module", autouse=True)
def _restore_real_modules():
    """Put the real modules back after this test module finishes so
    later test files see the production investigation_tools / agents /
    chatkit, not our stubs.

    Three layers of cleanup are needed:
      1. undo the in-place `agents.function_tool` monkeypatch (we
         mutated an already-loaded module),
      2. drop any module we created from scratch,
      3. drop the `app.*` cache so re-imports resolve the real modules
         instead of walking stale attributes on parent packages.
    """
    yield
    # 1. Undo the function_tool monkeypatch on `agents` (no-op if agents
    #    was freshly stubbed by us — the module will be popped below).
    agents_live = sys.modules.get("agents")
    if agents_live is not None:
        if _AGENTS_ORIG_FN_TOOL is None:
            # There was no function_tool before us; remove our stub.
            if hasattr(agents_live, "function_tool"):
                try:
                    delattr(agents_live, "function_tool")
                except AttributeError:
                    pass
        else:
            # Restore the previously-present function_tool.
            agents_live.function_tool = _AGENTS_ORIG_FN_TOOL  # type: ignore[attr-defined]
    # 2. Restore originals we had snapshotted
    for name, mod in _ORIGINAL_MODULES.items():
        sys.modules[name] = mod  # type: ignore[assignment]
    # Drop anything we created that wasn't there originally
    for name in list(_STUBBED_MODULE_NAMES):
        if name not in _ORIGINAL_MODULES:
            sys.modules.pop(name, None)
    # 3. Drop the `app.*` cache so a re-import resolves the real modules,
    # not the stubbed attribute on the parent package.
    for name in [n for n in list(sys.modules) if n == "app" or n.startswith("app.")]:
        sys.modules.pop(name, None)


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

    def test_bash_blocks_destructive_command(self):
        """bash() applies guardrails before command execution."""
        bash = self._get_bash()
        ctx = make_ctx("bash-blocked")
        async def run():
            _registry.clear()
            return await bash(ctx, "git reset --hard HEAD")
        out = asyncio.run(run())
        assert "blocked by guardrails" in out

    def test_bash_requires_approval_for_aws_run_task(self):
        bash = self._get_bash()
        ctx = make_ctx("bash-aws-approval")
        async def run():
            _registry.clear()
            return await bash(ctx, "aws ecs run-task --cluster c --task-definition td")
        out = asyncio.run(run())
        assert "requires explicit approval" in out


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

    def test_edit_file_insert_after_line(self, tmp_path):
        """edit_file mode='insert' inserts text after the given line."""
        edit_file = self._get_edit_file()
        ctx = make_ctx()
        p = tmp_path / "ins.txt"
        p.write_text("one\ntwo\nthree\n")
        async def run():
            return await edit_file(
                ctx, str(p), new_string="one-and-a-half",
                mode="insert", insert_line=1,
            )
        out = asyncio.run(run())
        assert "OK" in out
        assert p.read_text() == "one\none-and-a-half\ntwo\nthree\n"

    def test_edit_file_insert_at_top_when_line_zero(self, tmp_path):
        edit_file = self._get_edit_file()
        ctx = make_ctx()
        p = tmp_path / "ins2.txt"
        p.write_text("one\ntwo\n")
        async def run():
            return await edit_file(
                ctx, str(p), new_string="zero",
                mode="insert", insert_line=0,
            )
        out = asyncio.run(run())
        assert "OK" in out
        assert p.read_text() == "zero\none\ntwo\n"

    def test_edit_file_insert_line_out_of_range(self, tmp_path):
        edit_file = self._get_edit_file()
        ctx = make_ctx()
        p = tmp_path / "ins3.txt"
        p.write_text("one\ntwo\n")
        async def run():
            return await edit_file(
                ctx, str(p), new_string="x",
                mode="insert", insert_line=99,
            )
        out = asyncio.run(run())
        assert "Error" in out
        assert "out of range" in out

    def test_edit_file_unknown_mode(self, tmp_path):
        edit_file = self._get_edit_file()
        ctx = make_ctx()
        p = tmp_path / "uk.txt"
        p.write_text("x\n")
        async def run():
            return await edit_file(ctx, str(p), mode="delete_line")
        out = asyncio.run(run())
        assert "Error" in out
        assert "mode" in out

    def test_edit_file_returns_context_window(self, tmp_path):
        """Successful edits include ±3 context lines around the edit."""
        edit_file = self._get_edit_file()
        ctx = make_ctx()
        p = tmp_path / "ctx.txt"
        p.write_text("a\nb\nTARGET\nd\ne\n")
        async def run():
            return await edit_file(ctx, str(p), "TARGET", "REPLACED")
        out = asyncio.run(run())
        assert "REPLACED" in out
        assert "a" in out and "e" in out


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
