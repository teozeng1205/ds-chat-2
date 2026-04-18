"""Unit tests for MemoryStore + memory_tools factory."""

from __future__ import annotations

from pathlib import Path

from app.memory import MemoryStore


def test_put_get_roundtrip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite")
    store.put(scope="user", scope_id="u1", key="team", value="B6")
    assert store.get(scope="user", scope_id="u1", key="team") == "B6"
    store.close()


def test_get_missing_returns_none(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite")
    assert store.get(scope="user", scope_id="u1", key="unknown") is None
    store.close()


def test_put_upserts(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite")
    store.put(scope="user", scope_id="u1", key="team", value="B6")
    store.put(scope="user", scope_id="u1", key="team", value="B6,YY")
    assert store.get(scope="user", scope_id="u1", key="team") == "B6,YY"
    store.close()


def test_scopes_are_isolated(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite")
    # Same key, different scope types
    store.put(scope="user", scope_id="u1", key="focus", value="crossdb")
    store.put(scope="thread", scope_id="u1", key="focus", value="B6-spike")
    assert store.get(scope="user", scope_id="u1", key="focus") == "crossdb"
    assert store.get(scope="thread", scope_id="u1", key="focus") == "B6-spike"
    store.close()


def test_list_returns_all_for_scope(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite")
    store.put(scope="user", scope_id="u1", key="a", value="1")
    store.put(scope="user", scope_id="u1", key="b", value="2")
    store.put(scope="user", scope_id="u2", key="a", value="other")  # different user
    items = store.list(scope="user", scope_id="u1")
    keys = [i["key"] for i in items]
    assert keys == ["a", "b"]
    store.close()


def test_delete(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite")
    store.put(scope="user", scope_id="u", key="k", value="v")
    assert store.delete(scope="user", scope_id="u", key="k") is True
    assert store.delete(scope="user", scope_id="u", key="k") is False  # idempotent
    assert store.get(scope="user", scope_id="u", key="k") is None
    store.close()


def test_memory_tools_factory_registers_four() -> None:
    from app.tools.memory_tools import memory_tools
    names = {t.name for t in memory_tools()}
    assert names == {"remember", "recall", "list_memories", "forget"}
