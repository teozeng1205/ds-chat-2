"""DS Chat Knowledge Base V2.

The V2 KB is a typed, task-first retrieval layer. It replaces the old
split lexical/semantic/lineage overlays with one SQLite-backed store
that ingests docs, table metadata, entity codes, skills, E2E cases, and
pipeline graph facts.
"""

from .models import Citation, KnowledgeChunk, KnowledgeEdge, KnowledgeItem, SearchResult, TaskRecipe
from .retriever import KnowledgeRetriever, default_kb_db_path
from .store import KnowledgeStore

__all__ = [
    "Citation",
    "KnowledgeChunk",
    "KnowledgeEdge",
    "KnowledgeItem",
    "KnowledgeRetriever",
    "KnowledgeStore",
    "SearchResult",
    "TaskRecipe",
    "default_kb_db_path",
]
