#!/usr/bin/env python3
"""Evaluation harness for DS Chat KB V2.

This is intentionally offline and deterministic by default. It measures
whether task-shaped questions retrieve the expected table/item/citation
without relying on OpenAI calls.

Usage:
    .venv/bin/python scripts/eval_kb_recall.py
    .venv/bin/python scripts/eval_kb_recall.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.investigation.kb import KnowledgeRetriever  # noqa: E402


DEFAULT_CASES = [
    {
        "id": "provider_issues",
        "question": "top provider collection issues today",
        "expect_any_table": ["prod.monitoring.provider_combined_audit", "prod.monitoring.combined_audit"],
        "expect_tools": ["search_kb"],
    },
    {
        "id": "s3_freshness",
        "question": "market anomaly s3 freshness prefix",
        "expect_types": ["s3_prefix", "doc"],
        "expect_tools": ["list_s3"],
    },
    {
        "id": "market_anomaly_eda",
        "question": "market anomaly EDA distribution",
        "expect_any_table": ["prod.analytics.market_level_anomalies_v4", "prod.analytics.market_level_anomalies_v3"],
        "expect_tools": ["execute_sql"],
    },
    {
        "id": "codebase_explanation",
        "question": "how does priceeye scheduling auto scheduler work in code",
        "expect_types": ["doc", "pipeline_stage", "code"],
        "expect_tools": ["read_file"],
    },
    {
        "id": "schema_inventory",
        "question": "what tables are in prod monitoring schema for collection debugging",
        "expect_any_table": ["prod.monitoring.provider_combined_audit", "prod.monitoring.combined_audit"],
        "expect_types": ["schema"],
        "expect_tools": ["search_kb"],
    },
]


def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    retriever = KnowledgeRetriever()
    try:
        retriever.ensure_ready(force=False)
        results: list[dict[str, Any]] = []
        for case in cases:
            result = retriever.search(case["question"]).to_dict()
            table_names = {t.get("name") for t in result.get("tables", [])}
            item_types = {i.get("type") for i in result.get("items", [])}
            tools = set(result.get("tool_plan", []))
            citations = result.get("citations", [])

            checks = {
                "table": not case.get("expect_any_table")
                or bool(table_names.intersection(set(case["expect_any_table"]))),
                "type": not case.get("expect_types")
                or bool(item_types.intersection(set(case["expect_types"]))),
                "tools": set(case.get("expect_tools", [])).issubset(tools),
                "citations": bool(citations),
            }
            results.append(
                {
                    "id": case["id"],
                    "ok": all(checks.values()),
                    "checks": checks,
                    "task": (result.get("task") or {}).get("id"),
                    "tables": sorted(t for t in table_names if t)[:8],
                    "item_types": sorted(t for t in item_types if t),
                    "tool_plan": result.get("tool_plan", []),
                    "confidence": result.get("confidence"),
                }
            )
        passed = sum(1 for r in results if r["ok"])
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
            "results": results,
        }
    finally:
        retriever.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    report = evaluate(DEFAULT_CASES)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"KB V2 eval: {report['passed']}/{report['total']} passed ({report['pass_rate']:.1%})")
        for result in report["results"]:
            status = "PASS" if result["ok"] else "FAIL"
            print(f"{status} {result['id']} task={result['task']} tools={','.join(result['tool_plan'])}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
