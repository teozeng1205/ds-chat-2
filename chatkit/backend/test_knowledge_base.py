from __future__ import annotations

from pathlib import Path

from app.knowledge_base import KnowledgeBaseService


def test_kb_load_and_search() -> None:
    root = Path(__file__).resolve().parent / "knowledgebase"
    kb = KnowledgeBaseService(root)
    kb.refresh_if_needed(force=True)

    table = kb.get_table("monitoring_provider_combined_audit")
    assert table.environment == "3VDEV"
    assert "sales_date" in table.partition_policy.required_predicates

    playbook = kb.match_playbook("what are the top site issues for QL2")
    assert playbook is not None
    assert playbook.get("playbook_id") == "top_site_issues"

    hits = kb.search("provider_combined_audit sales_date customer partition", top_k=5)
    assert hits
    assert any(hit["scope"] in {"table", "playbook", "doc"} for hit in hits)
