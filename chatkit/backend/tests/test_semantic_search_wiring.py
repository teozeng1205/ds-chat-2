"""Tests for _semantic_hits graceful fallback.

With the feature flags gone, semantic search is always enabled. The
only short-circuit left is "the embedding index hasn't been built
yet" — in that case `_semantic_hits` returns an empty list so
search_kb degrades to the lexical path without error.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_no_index_returns_empty() -> None:
    """If the semantic index file is missing, `_semantic_hits` returns []
    without raising and without making an OpenAI call."""
    from app.tools.investigation_tools import _semantic_hits
    backend_root = Path(__file__).resolve().parents[1]
    index_path = backend_root / "app" / ".data" / "ds-chat-semantic.sqlite"
    if index_path.exists():
        pytest.skip("semantic index already built; this test requires its absence")
    assert _semantic_hits("market anomalies") == []
