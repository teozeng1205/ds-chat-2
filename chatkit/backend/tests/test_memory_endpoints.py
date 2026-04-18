"""HTTP integration tests for /chatkit/memory and /chatkit/feedback/summary."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DS_CHAT_MEMORY_DB", str(tmp_path / "mem.sqlite"))
    monkeypatch.setenv("DS_CHAT_FEEDBACK_DB", str(tmp_path / "fb.sqlite"))
    # Reset singletons so they honor the new env vars
    import app.memory as mem
    import app.feedback as fb
    mem._SINGLETON = None
    fb._SINGLETON = None
    from app.main import app
    return TestClient(app)


# ── Memory endpoints ──


def test_list_empty_memory(_client: TestClient) -> None:
    r = _client.get("/chatkit/memory")
    assert r.status_code == 200
    body = r.json()
    assert body == {"scope": "user", "items": []}


def test_put_then_list_memory(_client: TestClient) -> None:
    r = _client.put("/chatkit/memory", json={"key": "team", "value": "B6"})
    assert r.status_code == 200 and r.json()["ok"] is True

    r = _client.get("/chatkit/memory")
    items = r.json()["items"]
    assert [i["key"] for i in items] == ["team"]
    assert items[0]["value"] == "B6"


def test_put_upserts_on_same_key(_client: TestClient) -> None:
    _client.put("/chatkit/memory", json={"key": "team", "value": "B6"})
    _client.put("/chatkit/memory", json={"key": "team", "value": "B6, YY"})
    items = _client.get("/chatkit/memory").json()["items"]
    assert len(items) == 1
    assert items[0]["value"] == "B6, YY"


def test_put_rejects_missing_fields(_client: TestClient) -> None:
    assert _client.put("/chatkit/memory", json={"value": "x"}).status_code == 400
    assert _client.put("/chatkit/memory", json={"key": "k"}).status_code == 400


def test_put_rejects_oversize(_client: TestClient) -> None:
    r = _client.put("/chatkit/memory", json={"key": "k", "value": "x" * 5000})
    assert r.status_code == 400


def test_delete_memory(_client: TestClient) -> None:
    _client.put("/chatkit/memory", json={"key": "k", "value": "v"})
    r = _client.delete("/chatkit/memory/k")
    assert r.status_code == 200 and r.json() == {"ok": True, "deleted": True}
    # Second delete is idempotent
    r2 = _client.delete("/chatkit/memory/k")
    assert r2.status_code == 200 and r2.json() == {"ok": True, "deleted": False}


# ── Feedback summary ──


def test_feedback_summary_empty_thread(_client: TestClient) -> None:
    r = _client.get("/chatkit/feedback/summary/unknown-thread")
    assert r.status_code == 200
    assert r.json() == {"total": 0, "up": 0, "down": 0}


def test_feedback_summary_counts(_client: TestClient) -> None:
    _client.post("/chatkit/feedback", json={"thread_id": "T", "verdict": 1})
    _client.post("/chatkit/feedback", json={"thread_id": "T", "verdict": 1})
    _client.post("/chatkit/feedback", json={"thread_id": "T", "verdict": -1})
    _client.post("/chatkit/feedback", json={"thread_id": "other", "verdict": 1})

    r = _client.get("/chatkit/feedback/summary/T")
    assert r.json() == {"total": 3, "up": 2, "down": 1}
