from __future__ import annotations

import json
from pathlib import Path

from app.investigation.kb import KnowledgeRetriever, KnowledgeStore
from app.investigation.kb import ingest


def _seed_kb_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    backend_root = tmp_path / "backend"
    knowledge_root = backend_root / "app" / "investigation" / "knowledge"
    docs_root = knowledge_root / "docs"
    skills_root = backend_root / "skills"
    tests_root = backend_root / "tests"
    docs_root.mkdir(parents=True)
    skills_root.mkdir(parents=True)
    tests_root.mkdir(parents=True)

    (docs_root / "priceeye_system.md").write_text(
        "# PriceEye System\n\n## Data Flow Overview\npriceeye-v2 -> ds-priceeye-analytics -> monitoring.\n\n## Processes & Their Tables\n\n## Provider Issues\nUse provider monitoring tables for collection issue debugging.\n",
        encoding="utf-8",
    )
    (docs_root / "old.md").write_text("# Legacy Thing\nlegacy only content\n", encoding="utf-8")
    (knowledge_root / "tables.md").write_text("# Tables\nprod.monitoring.provider_combined_audit\n", encoding="utf-8")
    (knowledge_root / "sql_best_practices.md").write_text("# SQL\nFilter by sales_date.\n", encoding="utf-8")
    (knowledge_root / "common_codes.json").write_text(
        json.dumps({"providers": [{"code": "QL2", "name": "QL2 Software", "aliases": ["QL2"]}], "sites": [], "customers": []}),
        encoding="utf-8",
    )
    (knowledge_root / "common_table_live_metadata.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table_name": "prod.monitoring.provider_combined_audit",
                        "datasource": "redshift_core",
                        "tier": "common",
                        "partitions": [{"column": "sales_date", "role": "recommended"}],
                        "columns": [{"column_name": "sales_date"}, {"column_name": "issue_sources"}],
                    },
                    {
                        "table_name": "prod.monitoring.combined_audit",
                        "datasource": "redshift_core",
                        "tier": "common",
                        "partitions": [{"column": "sales_date", "role": "recommended"}],
                        "columns": [{"column_name": "sales_date"}, {"column_name": "issue_source"}],
                    },
                    {"table_name": "local.metadata.foo_old", "datasource": "redshift_core"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (knowledge_root / "pipelines.json").write_text(
        json.dumps(
            {
                "nodes": {
                    "stage": [{"id": "stage:provider-monitor", "name": "provider-monitor", "metadata": {"repo": "ds-internal-monitoring"}}],
                    "redshift_table": [{"id": "redshift_table:prod.monitoring.provider_combined_audit", "name": "prod.monitoring.provider_combined_audit"}],
                },
                "edges": [
                    {"source": "stage:provider-monitor", "target": "redshift_table:prod.monitoring.provider_combined_audit", "rel": "writes"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (skills_root / "sql_investigation.md").write_text(
        "---\nname: sql_investigation\ndescription: SQL task\nkeywords: [sql, provider, issue]\ntier: high\n---\nUse search_kb then execute_sql.\n",
        encoding="utf-8",
    )
    (tests_root / "e2e_investigation_cases.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"id": "provider_issue", "question": "top provider issues", "assertions": {"required_tools": ["search_kb", "execute_sql"]}},
                    {"id": "how_does_priceeye_work", "question": "how does priceeye work? Use search_kb. This is a bounded documentation answer."},
                ]
            }
        ),
        encoding="utf-8",
    )
    return backend_root, knowledge_root, docs_root, skills_root


def test_kb_v2_ingests_typed_resources_and_filters_legacy(tmp_path: Path, monkeypatch) -> None:
    backend_root, knowledge_root, docs_root, skills_root = _seed_kb_tree(tmp_path)
    monkeypatch.setattr(ingest, "BACKEND_ROOT", backend_root)
    monkeypatch.setattr(ingest, "KNOWLEDGE_ROOT", knowledge_root)
    monkeypatch.setattr(ingest, "DOCS_ROOT", docs_root)
    monkeypatch.setattr(ingest, "SKILLS_ROOT", skills_root)
    monkeypatch.setattr(ingest, "E2E_CASES_PATH", backend_root / "tests" / "e2e_investigation_cases.json")

    store = KnowledgeStore(tmp_path / "kb.sqlite")
    summary = ingest.build_kb(store, force=True)

    assert summary["items"] >= 5
    assert summary["tasks"] >= 3
    assert summary["by_type"]["table"] == 2
    assert summary["by_type"]["schema"] >= 1
    hits = store.search_chunks("provider collection issues", top_k=5)
    assert any(hit["item"]["name"] == "prod.monitoring.provider_combined_audit" for hit in hits)
    schema_hits = store.search_chunks("prod.monitoring schema tables PriceEye collection issues", top_k=3)
    assert schema_hits[0]["item"]["id"] == "schema:prod.monitoring"
    assert "prod.monitoring.combined_audit" in schema_hits[0]["chunk"]["text"]
    overview_hits = store.search_chunks("how does priceeye work overview documentation source file", top_k=3)
    overview_doc = next(hit for hit in overview_hits if hit["item"]["id"] == "doc_overview:priceeye")
    assert "docs/priceeye_system.md" in overview_doc["chunk"]["text"]
    assert overview_doc["item"]["source_type"] == "doc_hint"
    assert overview_doc["item"]["requires_verification"] is True
    assert not any("legacy" in hit["chunk"]["text"].lower() for hit in hits)
    store.close()


def test_kb_v2_search_contract(tmp_path: Path, monkeypatch) -> None:
    backend_root, knowledge_root, docs_root, skills_root = _seed_kb_tree(tmp_path)
    monkeypatch.setattr(ingest, "BACKEND_ROOT", backend_root)
    monkeypatch.setattr(ingest, "KNOWLEDGE_ROOT", knowledge_root)
    monkeypatch.setattr(ingest, "DOCS_ROOT", docs_root)
    monkeypatch.setattr(ingest, "SKILLS_ROOT", skills_root)
    monkeypatch.setattr(ingest, "E2E_CASES_PATH", backend_root / "tests" / "e2e_investigation_cases.json")

    retriever = KnowledgeRetriever(tmp_path / "kb.sqlite")
    result = retriever.search("top provider issues").to_dict()

    assert set(result) == {
        "query",
        "task",
        "items",
        "verified_items",
        "hints",
        "tables",
        "lineage",
        "tool_plan",
        "citations",
        "confidence",
        "source_policy",
        "verification_required",
        "authority_trace",
        "retrieval_trace",
    }
    assert "candidate" + "_tables" not in result
    assert "semantic" + "_hits" not in result
    assert result["task"] is not None
    assert any(t["name"] == "prod.monitoring.provider_combined_audit" for t in result["tables"])
    assert any(item["id"] == "table:prod.monitoring.provider_combined_audit" for item in result["verified_items"])
    assert any(edge["rel"] == "writes" for edge in result["lineage"])

    overview = retriever.search("how does priceeye work? Use search_kb. This is a bounded documentation answer.").to_dict()
    assert overview["task"]["id"] == "task:e2e:how_does_priceeye_work"
    assert any(item["id"] == "doc_overview:priceeye" for item in overview["hints"])
    assert any(citation["source_type"] == "doc_hint" for citation in overview["citations"])
    retriever.close()
