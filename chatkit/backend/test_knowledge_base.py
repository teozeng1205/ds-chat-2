from __future__ import annotations

import warnings
from pathlib import Path
from shutil import copytree

import numpy as np

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


def test_kb_search_tolerates_non_finite_vector_index_values(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parent / "knowledgebase"
    temp_root = tmp_path / "knowledgebase"
    copytree(source_root, temp_root)

    kb = KnowledgeBaseService(temp_root)
    kb.refresh_if_needed(force=True)

    vec_path = kb.vector_dir / "doc_vectors.npy"
    matrix = np.load(vec_path, allow_pickle=False)
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix[0, 0] = np.inf
    matrix[0, 1] = np.nan
    matrix[0, 2] = -np.inf
    np.save(vec_path, matrix, allow_pickle=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        hits = kb.search("top site issues", top_k=5)

    assert hits
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings
