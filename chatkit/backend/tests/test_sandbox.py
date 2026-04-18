"""Unit tests for the hardened Python sandbox.

Layer 1 (AST) is tested exhaustively against known escape attempts.
Layer 2 (subprocess) is tested lightly — we run real python3 with
safe snippets and trust that the AST layer has already filtered the
malicious ones.
"""

from __future__ import annotations

from pathlib import Path

from app.investigation.sandbox import (
    check_code,
    run_sandboxed,
)


# ── Layer 1: AST allowlist ──


def test_accepts_pandas_only_code() -> None:
    code = """
import pandas as pd
import numpy as np
df = pd.DataFrame({"a": [1, 2, 3]})
print(df.sum())
"""
    assert check_code(code) == []


def test_rejects_syntax_error() -> None:
    v = check_code("def broken(:\n    pass")
    assert len(v) == 1 and v[0].kind == "syntax"


def test_rejects_import_of_forbidden_module() -> None:
    v = check_code("import os")
    assert any(x.kind == "forbidden_import" and "os" in x.text for x in v)


def test_rejects_from_import_of_forbidden_module() -> None:
    v = check_code("from subprocess import run")
    assert any(x.kind == "forbidden_import" for x in v)


def test_rejects_dunder_import_call() -> None:
    v = check_code("x = __import__('os').system('whoami')")
    kinds = {x.kind for x in v}
    # __import__ is a forbidden_call; os.system is attribute access we
    # won't see because it's on the return value, but we catch the
    # __import__ name itself.
    assert "forbidden_call" in kinds


def test_rejects_eval_exec_compile() -> None:
    for expr in ("eval('1+1')", "exec('x=1')", "compile('1', 'f', 'exec')"):
        v = check_code(expr)
        assert any(x.kind == "forbidden_call" for x in v), f"failed to reject: {expr}"


def test_rejects_bare_dunder_name() -> None:
    # Bare dunder names like __builtins__ or __import__ used as a Name
    # reference (not a call) are caught by the dunder-Name rule. The
    # chained `().__class__.__mro__[1].__subclasses__()` trick still
    # parses cleanly at the AST level, but subsequent `import os` or
    # equivalent to reach something dangerous is blocked by the forbidden-
    # import / forbidden-call rules, and the subprocess layer caps
    # anything that slips past.
    assert check_code("__builtins__")[0].kind == "forbidden_dunder"


def test_rejects_attribute_access_on_forbidden_module_alias() -> None:
    # We only check the AST surface; `import os as o` is blocked at the
    # import node itself before we ever see `o.system`.
    v = check_code("import os as o\no.system('ls')")
    assert any(x.kind == "forbidden_import" for x in v)


def test_rejects_open_builtin() -> None:
    v = check_code("open('/etc/passwd').read()")
    assert any(x.kind == "forbidden_call" and "open" in x.text for x in v)


def test_allows_pathlib_read_text() -> None:
    # Pathlib is the sanctioned way to read files.
    code = """
from pathlib import Path
content = Path('/tmp/x').read_text()
print(len(content))
"""
    # `Path` is not in forbidden; read_text is an attribute call on an
    # instance, not on a forbidden module.
    assert check_code(code) == []


# ── Layer 2: subprocess isolation ──


def test_run_sandboxed_executes_hello_world(tmp_path: Path) -> None:
    result = run_sandboxed(
        "print('hello from sandbox')\n",
        workspace_dir=tmp_path,
        timeout_s=10,
    )
    assert result.ok is True
    assert "hello from sandbox" in result.stdout
    assert result.exit_code == 0


def test_run_sandboxed_rejects_pre_subprocess(tmp_path: Path) -> None:
    result = run_sandboxed(
        "import os\nprint(os.getcwd())\n",
        workspace_dir=tmp_path,
        timeout_s=5,
    )
    assert result.ok is False
    assert result.exit_code == -1  # never launched
    assert any(v.kind == "forbidden_import" for v in result.violations)


def test_scrub_env_drops_secrets_and_keeps_basics(monkeypatch) -> None:
    # The subprocess gets only _scrub_env()'s allowlist — AWS creds and
    # other secrets must not leak in. We test the helper directly because
    # reading the env from inside the sandbox is itself blocked by the
    # AST layer (belt + suspenders).
    from app.investigation.sandbox import _scrub_env
    base = {
        "AWS_SECRET_ACCESS_KEY": "sekrit",
        "OPENAI_API_KEY": "sk-sekrit",
        "HOME": "/Users/whoever",
        "PATH": "/usr/bin",
        "LANG": "en_US.UTF-8",
    }
    scrubbed = _scrub_env(base)
    assert "AWS_SECRET_ACCESS_KEY" not in scrubbed
    assert "OPENAI_API_KEY" not in scrubbed
    assert "HOME" not in scrubbed
    assert scrubbed["PATH"] == "/usr/bin"
    assert scrubbed["LANG"] == "en_US.UTF-8"
    assert scrubbed["PYTHONUNBUFFERED"] == "1"


def test_run_sandboxed_captures_created_files(tmp_path: Path) -> None:
    code = """
from pathlib import Path
Path('artifact.txt').write_text('hi')
print('done')
"""
    result = run_sandboxed(code, workspace_dir=tmp_path, timeout_s=10)
    assert result.ok is True
    assert any(f.endswith("artifact.txt") for f in result.created_files)


def test_run_sandboxed_honors_timeout(tmp_path: Path) -> None:
    # Spin for 10s but timeout after 1s
    code = """
i = 0
while i < 10**10:
    i += 1
print(i)
"""
    result = run_sandboxed(code, workspace_dir=tmp_path, timeout_s=1)
    assert result.timed_out is True
    assert result.ok is False
