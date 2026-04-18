"""Tests for _semantic_hits graceful fallbacks.

We don't exercise the OpenAI embedding path in CI; we verify the
feature flag and the "no index present yet" short-circuits work so
the wiring doesn't break search_kb when the index hasn't been built.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_flag_off_returns_no_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_KB_ENABLED", "false")
    from app.tools.investigation_tools import _semantic_hits
    assert _semantic_hits("market anomalies") == []


def test_flag_on_but_no_index_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SEMANTIC_KB_ENABLED", "true")
    # Redirect .data away from any real index
    monkeypatch.setenv("DS_CHAT_QUERY_CACHE_DB", str(tmp_path / "cache.sqlite"))
    # Make sure there's no semantic index at the well-known path. Our
    # function resolves the path relative to the module, so we can't
    # easily redirect — but the production default path won't exist
    # inside an ephemeral test env.
    from app.tools.investigation_tools import _semantic_hits
    backend_root = Path(__file__).resolve().parents[1]
    index_path = backend_root / "app" / ".data" / "ds-chat-semantic.sqlite"
    if index_path.exists():
        pytest.skip("semantic index already built; this test requires its absence")
    assert _semantic_hits("market anomalies") == []
