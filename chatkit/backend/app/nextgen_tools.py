"""Compatibility shim for historical nextgen tool imports."""

from __future__ import annotations

import warnings

from .tools.investigation_tools import *  # noqa: F401,F403

warnings.warn(
    "app.nextgen_tools is deprecated; use app.tools.investigation_tools.",
    DeprecationWarning,
    stacklevel=2,
)
