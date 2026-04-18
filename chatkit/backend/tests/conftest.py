"""Project-level pytest configuration.

Excludes test_shell_tools.py from default collection. That file stubs
agents / chatkit / app.tools.investigation_tools in sys.modules at
IMPORT time so it can call `@function_tool`-decorated shell tools as
plain async functions. Because pytest imports every test file during
collection, those stubs leak into later test files that rely on the
real modules.

Keeping test_shell_tools.py isolated matches the existing pattern in
backend/scripts/verify_investigation.sh which already skips it.
Developers working on shell_tools.py can run it directly:

    .venv/bin/python -m pytest tests/test_shell_tools.py
    DS_CHAT_RUN_SHELL_TOOLS=1 .venv/bin/python -m pytest   # opt-in run-everything

"""

from __future__ import annotations

import os

collect_ignore: list[str] = []

if not os.environ.get("DS_CHAT_RUN_SHELL_TOOLS"):
    collect_ignore.append("test_shell_tools.py")
