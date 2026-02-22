"""Typed contracts for DS Chat Next-Gen investigation flows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SourceSystem = Literal["redshift", "mysql", "s3"]
EntityType = Literal["provider", "site", "customer", "unknown"]
StepType = Literal[
    "resolve_entities",
    "sql_extract",
    "s3_extract",
    "join",
    "python_analysis",
    "summarize",
]


class PartitionPolicy(BaseModel):
    partition_columns: list[str] = Field(default_factory=list)
    required_predicates: list[str] = Field(default_factory=list)
    notes: str | None = None


class TableSpec(BaseModel):
    table_id: str
    physical_name: str
    source_system: SourceSystem
    environment: str = "3VDEV"
    description: str = ""
    primary_keys: list[str] = Field(default_factory=list)
    join_keys: dict[str, str] = Field(default_factory=dict)
    entity_columns: dict[str, str] = Field(default_factory=dict)
    partition_policy: PartitionPolicy = Field(default_factory=PartitionPolicy)
    default_date_column: str | None = None
    default_customer_column: str | None = None
    default_limit: int = 5000
    tags: list[str] = Field(default_factory=list)


class RelationshipSpec(BaseModel):
    left_table_id: str
    right_table_id: str
    left_key: str
    right_key: str
    relationship: str = "many_to_one"
    notes: str | None = None


class PlanFilter(BaseModel):
    column: str
    operator: Literal["=", "!=", ">", ">=", "<", "<=", "IN", "LIKE"] = "="
    value: Any


class PlanStep(BaseModel):
    step_id: str
    step_type: StepType
    description: str
    table_id: str | None = None
    s3_uri: str | None = None
    input_datasets: list[str] = Field(default_factory=list)
    output_dataset_id: str | None = None
    filters: list[PlanFilter] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class InvestigationPlan(BaseModel):
    plan_id: str
    intent: str
    question: str
    requires_clarification: bool = False
    clarification_prompt: str | None = None
    missing_predicates: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EntityResolution(BaseModel):
    input_code: str
    normalized_code: str
    entity_type: EntityType = "unknown"
    canonical_value: str | None = None
    confidence: float = 0.0
    source: Literal["common_codes", "mysql_fallback", "unknown"] = "unknown"
    candidates: list[str] = Field(default_factory=list)
    ambiguous: bool = False


class DatasetHandle(BaseModel):
    dataset_id: str
    path: str
    manifest_path: str
    row_count: int
    columns: list[str]
    source_step_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: str
    source_type: Literal["sql", "s3", "join", "python"]
    source_ref: str
    query: str | None = None
    s3_keys: list[str] = Field(default_factory=list)
    partitions: dict[str, Any] = Field(default_factory=dict)
    row_count: int
    columns_schema: list[dict[str, str]] = Field(default_factory=list, alias="schema")
    lineage: list[str] = Field(default_factory=list)
    sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnalysisResult(BaseModel):
    analysis_name: str
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KBSearchHit(BaseModel):
    doc_id: str
    scope: str
    path: str
    title: str
    snippet: str
    score: float
    method: Literal["fts", "vector", "merged"]
