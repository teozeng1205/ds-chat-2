from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.entity_resolver import EntityResolver
from app.knowledge_base import KnowledgeBaseService


class _NoopTVClient:
    def mysql_table_columns(self, table_name: str):
        return []

    def query_mysql(self, query: str):
        raise RuntimeError("not expected")


def test_resolve_common_codes_without_mysql() -> None:
    root = Path(__file__).resolve().parent / "knowledgebase"
    kb = KnowledgeBaseService(root)
    kb.refresh_if_needed(force=True)
    resolver = EntityResolver(kb, _NoopTVClient())  # type: ignore[arg-type]

    results = resolver.resolve_codes("QL2,unknown-code")
    by_input = {item.input_code: item for item in results}

    assert by_input["QL2"].entity_type == "provider"
    assert by_input["QL2"].source == "common_codes"
    assert by_input["unknown-code"].entity_type == "unknown"


class _CustomerNameTVClient:
    def mysql_table_columns(self, table_name: str):
        if table_name == "priceeye.customer":
            return ["name", "description"]
        return []

    def query_mysql(self, query: str):
        if "FROM priceeye.customer" in query and "UPPER(name)" in query:
            return pd.DataFrame({"code": ["ZZTEST"]})
        return pd.DataFrame({"code": []})


def test_mysql_customer_resolution_supports_name_column() -> None:
    root = Path(__file__).resolve().parent / "knowledgebase"
    kb = KnowledgeBaseService(root)
    kb.refresh_if_needed(force=True)
    resolver = EntityResolver(kb, _CustomerNameTVClient())  # type: ignore[arg-type]

    results = resolver.resolve_codes("ZZTEST")
    assert len(results) == 1
    assert results[0].entity_type == "customer"
    assert results[0].canonical_value == "ZZTEST"
    assert results[0].source == "mysql_fallback"
