"""Tests for PartitionGuard's Glue-backed mode.

Verifies:
  - Legacy classmethod behavior unchanged
  - from_glue(...) returns an instance
  - check_live uses Glue partition keys when the table resolves
  - check_live falls back to the static map when Glue misses
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.investigation.executor import PartitionGuard


class _FakeGlue:
    """Records lookups and returns configured partition_key_names."""

    def __init__(self, tables: dict[tuple[str, str], tuple[str, ...]]):
        self._tables = tables
        self.get_table_calls: list[tuple[str, str]] = []
        self.discover_calls: list[str] = []

    def get_table(self, db: str, name: str) -> Any | None:
        self.get_table_calls.append((db, name))
        keys = self._tables.get((db, name))
        if keys is None:
            return None
        return SimpleNamespace(partition_key_names=keys)

    def discover_table(self, ref: str) -> list[Any]:
        self.discover_calls.append(ref)
        matches = [
            SimpleNamespace(partition_key_names=keys)
            for (_db, name), keys in self._tables.items()
            if name == ref
        ]
        return matches


def test_legacy_classmethod_unchanged() -> None:
    q = "select 1 from analytics.market_level_anomalies_v3"
    warns = PartitionGuard.check(q)
    assert warns and "sales_date" in warns[0]


def test_from_glue_instance_uses_live_partition_keys() -> None:
    glue = _FakeGlue({("analytics_db", "custom_ephemeral_table"): ("sales_date", "customer")})
    g = PartitionGuard.from_glue(glue)

    warns = g.check_live(
        "select 1 from analytics_db.custom_ephemeral_table",
    )
    # Glue says sales_date + customer; query has neither
    assert len(warns) == 2
    messages = " | ".join(warns)
    assert "sales_date" in messages and "customer" in messages
    assert ("analytics_db", "custom_ephemeral_table") in glue.get_table_calls


def test_live_check_passes_when_partitions_present() -> None:
    glue = _FakeGlue({("analytics_db", "t"): ("sales_date",)})
    g = PartitionGuard.from_glue(glue)
    warns = g.check_live("select * from analytics_db.t where sales_date = 20260417")
    assert warns == []


def test_live_falls_back_to_static_map_when_glue_misses() -> None:
    glue = _FakeGlue({})  # no tables configured
    g = PartitionGuard.from_glue(glue)

    # Without Glue data, the static map still catches the classic case
    warns = g.check_live("select * from prod.monitoring.provider_combined_audit")
    assert warns and "sales_date" in warns[0]


def test_live_extracts_multiple_tables_from_join() -> None:
    glue = _FakeGlue({
        ("monitoring_db", "a"): ("sales_date",),
        ("analytics_db", "b"): ("customer",),
    })
    g = PartitionGuard.from_glue(glue)
    q = "select x from monitoring_db.a as a join analytics_db.b as b on a.k = b.k"
    warns = g.check_live(q)
    # both tables' partitions missing
    assert len(warns) == 2
