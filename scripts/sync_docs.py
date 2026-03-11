#!/usr/bin/env python3
"""Sync ~/git/documentations/*.md into the investigation KB docs directory.

Usage (from chatkit/backend/):
    .venv/bin/python ../../scripts/sync_docs.py
"""
from __future__ import annotations
import shutil
from pathlib import Path

SRC = Path.home() / "git" / "documentations"
DST = Path(__file__).resolve().parents[1] / "chatkit" / "backend" / \
      "app" / "investigation" / "knowledge" / "docs"
KB_DB = Path(__file__).resolve().parents[1] / "chatkit" / "backend" / \
        ".work" / "knowledge" / "knowledge.sqlite"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_file in sorted(SRC.glob("*.md")):
        shutil.copy2(src_file, DST / src_file.name)
        copied += 1
    print(f"Synced {copied} docs to {DST}")
    if KB_DB.exists():
        KB_DB.unlink()
        print(f"Deleted stale KB: {KB_DB} (will rebuild on next server start)")


if __name__ == "__main__":
    main()
