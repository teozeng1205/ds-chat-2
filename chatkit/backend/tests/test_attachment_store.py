"""Tests for LocalDiskAttachmentStore._build_upload_url.

ChatKit's AttachmentUploadDescriptor requires an absolute URL. These
tests lock in every fallback branch so we never regress to emitting
a relative path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.attachment_store import LocalDiskAttachmentStore


def _store(tmp_path: Path) -> LocalDiskAttachmentStore:
    return LocalDiskAttachmentStore(tmp_path)


def _fake_request(**attrs: Any) -> Any:
    url = SimpleNamespace(scheme=attrs.get("scheme", "http"),
                          netloc=attrs.get("netloc", "127.0.0.1:8000"))
    return SimpleNamespace(url=url, headers=attrs.get("headers", {}))


def test_absolute_via_explicit_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHATKIT_PUBLIC_BASE_URL", "https://chat.atpco.internal")
    url = _store(tmp_path)._build_upload_url("atc_abc", {})
    assert url == "https://chat.atpco.internal/chatkit/uploads/atc_abc"


def test_absolute_via_x_forwarded_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHATKIT_PUBLIC_BASE_URL", raising=False)
    req = _fake_request(headers={"x-forwarded-host": "chat.example.com",
                                  "x-forwarded-proto": "https"})
    url = _store(tmp_path)._build_upload_url("atc_abc", {"request": req})
    assert url == "https://chat.example.com/chatkit/uploads/atc_abc"


def test_absolute_via_origin_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHATKIT_PUBLIC_BASE_URL", raising=False)
    req = _fake_request(headers={"origin": "https://app.local/"})
    url = _store(tmp_path)._build_upload_url("atc_abc", {"request": req})
    assert url == "https://app.local/chatkit/uploads/atc_abc"


def test_absolute_via_request_netloc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHATKIT_PUBLIC_BASE_URL", raising=False)
    req = _fake_request(headers={}, scheme="http", netloc="localhost:3000")
    url = _store(tmp_path)._build_upload_url("atc_abc", {"request": req})
    assert url == "http://localhost:3000/chatkit/uploads/atc_abc"


def test_absolute_via_localhost_fallback_when_no_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHATKIT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("CHATKIT_INTERNAL_BASE_URL", raising=False)
    url = _store(tmp_path)._build_upload_url("atc_abc", {})
    assert url == "http://localhost:8000/chatkit/uploads/atc_abc"


def test_internal_base_url_env_overrides_localhost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHATKIT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("CHATKIT_INTERNAL_BASE_URL", "http://internal-host:9000")
    url = _store(tmp_path)._build_upload_url("atc_abc", {})
    assert url == "http://internal-host:9000/chatkit/uploads/atc_abc"


def test_always_absolute_regardless_of_missing_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Every branch should produce a URL with a scheme
    monkeypatch.delenv("CHATKIT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("CHATKIT_INTERNAL_BASE_URL", raising=False)
    variants = [
        {},
        {"request": _fake_request(headers={})},
        {"request": _fake_request(headers={"x-forwarded-host": "h.example.com"})},
        {"request": _fake_request(headers={"origin": "https://o.example.com"})},
    ]
    for ctx in variants:
        url = _store(tmp_path)._build_upload_url("atc_x", ctx)
        assert "://" in url, f"relative URL returned for ctx={ctx!r}: {url}"
