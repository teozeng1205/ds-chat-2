"""Compatibility shim for historical nextgen type imports."""

from __future__ import annotations

import warnings

from .investigation.types import *  # noqa: F401,F403

warnings.warn(
    "app.nextgen_types is deprecated; use app.investigation.types.",
    DeprecationWarning,
    stacklevel=2,
)
