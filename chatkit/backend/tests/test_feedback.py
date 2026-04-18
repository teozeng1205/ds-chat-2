"""Unit + HTTP integration tests for feedback."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.feedback import FeedbackStore


# ── Store ──


def test_record_and_summary(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "f.sqlite")
    store.record(thread_id="T1", verdict=1, message_id="m1")
    store.record(thread_id="T1", verdict=1, comment="clear!")
    store.record(thread_id="T1", verdict=-1, comment="wrong table")
    s = store.summary_by_thread("T1")
    assert s == {"total": 3, "up": 2, "down": 1}
    store.close()


def test_record_rejects_bad_verdict(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "f.sqlite")
    with pytest.raises(ValueError):
        store.record(thread_id="T", verdict=0)
    store.close()


def test_record_rejects_missing_thread(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "f.sqlite")
    with pytest.raises(ValueError):
        store.record(thread_id="", verdict=1)
    store.close()


def test_recent_is_newest_first(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "f.sqlite")
    store.record(thread_id="T", verdict=1, message_id="first")
    store.record(thread_id="T", verdict=-1, message_id="second")
    rows = store.recent(limit=2)
    assert rows[0].message_id == "second"
    assert rows[1].message_id == "first"
    store.close()


def test_comment_clipping(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "f.sqlite")
    long_comment = "x" * 5000
    store.record(thread_id="T", verdict=1, comment=long_comment)
    rows = store.recent(limit=1)
    assert rows[0].comment is not None and len(rows[0].comment) == 2000
    store.close()


# ── HTTP endpoint ──


@pytest.fixture
def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DS_CHAT_FEEDBACK_DB", str(tmp_path / "f.sqlite"))
    import app.feedback as fb
    fb._SINGLETON = None
    from app.main import app
    return TestClient(app)


def test_http_post_feedback_ok(_client: TestClient) -> None:
    r = _client.post("/chatkit/feedback", json={
        "thread_id": "T-1",
        "verdict": 1,
        "message_id": "m-1",
        "comment": "great answer",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["id"], int) and body["id"] > 0


def test_http_post_feedback_rejects_missing_thread(_client: TestClient) -> None:
    r = _client.post("/chatkit/feedback", json={"verdict": 1})
    assert r.status_code == 400


def test_http_post_feedback_rejects_bad_verdict(_client: TestClient) -> None:
    r = _client.post("/chatkit/feedback", json={"thread_id": "T", "verdict": 5})
    assert r.status_code == 400


def test_http_post_feedback_rejects_non_json(_client: TestClient) -> None:
    r = _client.post("/chatkit/feedback", data="not-json",
                     headers={"Content-Type": "text/plain"})
    assert r.status_code == 400
