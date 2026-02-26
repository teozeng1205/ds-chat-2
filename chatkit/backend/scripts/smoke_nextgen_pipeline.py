#!/usr/bin/env python3
"""Compatibility wrapper for legacy nextgen smoke entrypoint."""

from __future__ import annotations

import runpy
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "smoke_investigation_pipeline.py"
runpy.run_path(str(TARGET), run_name="__main__")
