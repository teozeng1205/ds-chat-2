"""Unit tests for app.investigation.query_cache."""

from __future__ import annotations

from pathlib import Path

from app.investigation.query_cache import QueryCache, make_key


def test_make_key_normalizes_whitespace_and_case() -> None:
    a = make_key("SELECT 1 FROM t", "analytics")
    b = make_key("  select    1   FROM\tT  ", "ANALYTICS")
    assert a == b


def test_make_key_depends_on_workgroup() -> None:
    a = make_key("select 1", "analytics")
    b = make_key("select 1", "core")
    assert a != b


def test_make_key_depends_on_extra_tags() -> None:
    a = make_key("select 1", "analytics", extra=["2026-04-17"])
    b = make_key("select 1", "analytics", extra=["2026-04-16"])
    assert a != b


def test_put_get_roundtrip(tmp_path: Path) -> None:
    cache = QueryCache(tmp_path / "cache.sqlite", default_ttl_s=900)
    payload = {"rows": [[1, "a"], [2, "b"]], "columns": ["id", "name"]}
    cache.put("select * from t", payload, workgroup="analytics")

    hit = cache.get("select * from t", "analytics")
    assert hit is not None
    assert hit.payload == payload
    assert hit.age_seconds >= 0
    cache.close()


def test_expired_entry_returns_none(tmp_path: Path) -> None:
    cache = QueryCache(tmp_path / "cache.sqlite", default_ttl_s=10)
    cache.put("select 1", {"ok": True}, workgroup="wg", now=1000.0)
    # 15s later → expired
    assert cache.get("select 1", "wg", now=1015.0) is None
    # purge_expired removes it
    removed = cache.purge_expired(now=1015.0)
    assert removed == 1
    cache.close()


def test_put_updates_existing_key_in_place(tmp_path: Path) -> None:
    cache = QueryCache(tmp_path / "cache.sqlite")
    cache.put("select 1", {"v": 1}, workgroup="wg")
    cache.put("select 1", {"v": 2}, workgroup="wg")
    hit = cache.get("select 1", "wg")
    assert hit is not None and hit.payload == {"v": 2}
    assert cache.stats()["entries"] == 1
    cache.close()


def test_invalidate(tmp_path: Path) -> None:
    cache = QueryCache(tmp_path / "cache.sqlite")
    cache.put("select 1", {"ok": True}, workgroup="wg")
    assert cache.invalidate("select 1", "wg") is True
    assert cache.get("select 1", "wg") is None
    assert cache.invalidate("select 1", "wg") is False
    cache.close()
