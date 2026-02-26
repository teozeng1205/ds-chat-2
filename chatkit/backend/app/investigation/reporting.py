"""Final response synthesis for investigation runs."""

from __future__ import annotations

from typing import Any


def build_lineage(*, run_id: str, datasets: list[dict[str, Any]], analysis: dict[str, Any] | None, warnings: list[str]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset_ids": [item.get("dataset_id") for item in datasets],
        "key_queries": [
            item.get("source_metadata", {}).get("query")
            for item in datasets
            if isinstance(item.get("source_metadata"), dict) and item.get("source_metadata", {}).get("query")
        ],
        "caveats": list({*warnings, *(analysis or {}).get("caveats", [])}),
    }


def summarize_answer(
    *,
    question: str,
    strategy: str,
    datasets: list[dict[str, Any]],
    analysis: dict[str, Any] | None,
    warnings: list[str],
    clarification: str | None,
) -> str:
    if clarification:
        return clarification
    if analysis and analysis.get("report_markdown"):
        return str(analysis["report_markdown"])

    lines = [
        "## Investigation Result",
        f"- Strategy: {strategy}",
        f"- Datasets produced: {len(datasets)}",
        f"- Question: {question}",
    ]
    if warnings:
        lines.append("- Warnings:")
        lines.extend([f"  - {item}" for item in warnings])
    return "\n".join(lines)


__all__ = ["build_lineage", "summarize_answer"]
