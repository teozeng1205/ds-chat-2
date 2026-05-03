"""Typed models for the DS Chat KB V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    type: str
    name: str
    title: str
    summary: str
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    item_id: str
    kind: str
    text: str
    source_path: str | None = None
    heading: str | None = None
    citation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeEdge:
    source_id: str
    target_id: str
    rel: str
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskRecipe:
    id: str
    name: str
    description: str
    triggers: tuple[str, ...]
    tool_plan: tuple[str, ...]
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["triggers"] = list(self.triggers)
        out["tool_plan"] = list(self.tool_plan)
        return out


@dataclass(frozen=True)
class Citation:
    source: str
    item_id: str
    chunk_id: str
    title: str | None = None
    excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    query: str
    task: dict[str, Any] | None
    items: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    lineage: list[dict[str, Any]]
    tool_plan: list[str]
    citations: list[dict[str, Any]]
    confidence: float
    retrieval_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
