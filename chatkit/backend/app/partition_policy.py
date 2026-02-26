"""Partition and SQL predicate enforcement for KB-defined tables."""

from __future__ import annotations

import re
from typing import Any

from .investigation.types import PlanFilter, TableSpec

_ALLOWED_OPERATORS = {"=", "!=", ">", ">=", "<", "<=", "IN", "LIKE"}
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_identifier(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch == "_")


def required_predicates(table_spec: TableSpec) -> list[str]:
    if table_spec.partition_policy.required_predicates:
        return list(table_spec.partition_policy.required_predicates)
    return list(table_spec.partition_policy.partition_columns)


def ensure_partition_filters(table_spec: TableSpec, filters: list[PlanFilter]) -> None:
    required = required_predicates(table_spec)
    if not required:
        return
    present = {_normalize_identifier(item.column) for item in filters}
    missing = [col for col in required if _normalize_identifier(col) not in present]
    if missing:
        raise ValueError(
            f"Missing required partition predicates for table '{table_spec.table_id}': {', '.join(missing)}"
        )


def _quote_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _condition_sql(filter_item: PlanFilter) -> str:
    if not _IDENT_RE.match(filter_item.column):
        raise ValueError(f"Unsupported column name in filter: {filter_item.column}")
    operator = filter_item.operator.upper()
    if operator not in _ALLOWED_OPERATORS:
        raise ValueError(f"Unsupported operator in filter: {filter_item.operator}")

    if operator == "IN":
        raw_values = filter_item.value if isinstance(filter_item.value, list) else [filter_item.value]
        if not raw_values:
            raise ValueError(f"IN predicate must include at least one value for {filter_item.column}")
        rendered = ", ".join(_quote_sql(v) for v in raw_values)
        return f"{filter_item.column} IN ({rendered})"

    return f"{filter_item.column} {operator} {_quote_sql(filter_item.value)}"


def build_where_clause(filters: list[PlanFilter]) -> str:
    if not filters:
        return "1=1"
    parts = [_condition_sql(item) for item in filters]
    return " AND ".join(parts)


def apply_default_required_filters(
    table_spec: TableSpec,
    filters: list[PlanFilter],
    defaults: dict[str, Any],
) -> list[PlanFilter]:
    out = list(filters)
    existing = {_normalize_identifier(f.column) for f in filters}
    for predicate in required_predicates(table_spec):
        norm = _normalize_identifier(predicate)
        if norm in existing:
            continue
        if predicate not in defaults:
            continue
        out.append(PlanFilter(column=predicate, operator="=", value=defaults[predicate]))
        existing.add(norm)
    return out
