from __future__ import annotations

from pathlib import Path

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
