"""Compatibility shim for investigation tools.

Use `app.tools.investigation_tools` for active imports.
"""

from __future__ import annotations

import warnings

from ..tools.investigation_tools import *  # noqa: F401,F403

warnings.warn(
    "app.investigation.tools is deprecated; import app.tools.investigation_tools instead.",
    DeprecationWarning,
    stacklevel=2,
)
