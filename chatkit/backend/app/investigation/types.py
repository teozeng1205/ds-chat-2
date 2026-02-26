"""Active typed contracts for generic autonomous investigation runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


ActionName = Literal[
    "resolve_entities",
    "retrieve_knowledge",
    "inspect_table_metadata",
    "extract_sql",
    "extract_s3",
    "run_python",
    "run_analysis",
    "summarize",
    "ask_clarification",
    "finish",
]


class ActionSpec(BaseModel):
    action: ActionName
    reason: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""


class TaskCardSpec(BaseModel):
    card_id: str
    title: str
    signals: list[str] = Field(default_factory=list)
    required_entities: list[str] = Field(default_factory=list)
    candidate_tables: list[str] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    analysis_mode: str = "profile_dataset"
    analysis_instructions: str = ""
    python_template: str = ""
    source_path: str = ""


class RunLineage(BaseModel):
    run_id: str
    dataset_ids: list[str] = Field(default_factory=list)
    key_queries: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class InvestigationEvent(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event: str
    payload: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ActionName",
    "ActionSpec",
    "TaskCardSpec",
    "RunLineage",
    "InvestigationEvent",
]
