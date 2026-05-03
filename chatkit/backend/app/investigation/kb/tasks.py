"""Task recipes for KB V2.

Skills and E2E scenarios enrich these recipes during ingestion, but the
base recipes are intentionally explicit so the agent has useful task
shape even before optional files are present.
"""

from __future__ import annotations

from .models import TaskRecipe


BASE_TASK_RECIPES: tuple[TaskRecipe, ...] = (
    TaskRecipe(
        id="task:provider_issue_investigation",
        name="Provider issue investigation",
        description="Debug PriceEye provider/site/customer collection issues using monitoring tables and issue fields.",
        triggers=("provider issue", "site issue", "collection issue", "combined audit", "provider_combined_audit", "top issues"),
        tool_plan=("search_kb", "resolve_codes", "execute_sql", "run_python"),
        metadata={"preferred_tables": ["prod.monitoring.provider_combined_audit", "prod.monitoring.combined_audit"]},
    ),
    TaskRecipe(
        id="task:s3_freshness",
        name="S3 freshness check",
        description="Find S3 bucket/prefix context, list latest objects, and compare freshness against expected outputs.",
        triggers=("s3", "bucket", "prefix", "freshness", "latest object", "mirror"),
        tool_plan=("search_kb", "list_s3", "fetch_s3"),
        metadata={"preferred_sources": ["s3_buckets.md"]},
    ),
    TaskRecipe(
        id="task:market_anomaly_eda",
        name="Market anomaly EDA",
        description="Analyze market/segment anomaly tables with partition-aware SQL and Python summaries or plots.",
        triggers=("market anomaly", "segment anomaly", "anomalies", "eda", "distribution", "impact"),
        tool_plan=("search_kb", "inspect_table", "execute_sql", "run_python", "publish_image"),
        metadata={"preferred_tables": ["prod.analytics.market_level_anomalies_v4", "prod.analytics.market_level_anomalies_v3"]},
    ),
    TaskRecipe(
        id="task:codebase_explanation",
        name="Codebase explanation",
        description="Explain how a named pipeline/component works by combining KB docs, lineage, and implementation files.",
        triggers=("how does", "implemented", "code", "class", "entry point", "scheduler", "pipeline work"),
        tool_plan=("search_kb", "trace_pipeline", "read_file", "bash"),
        metadata={"requires_code_verification": True},
    ),
    TaskRecipe(
        id="task:pipeline_root_cause",
        name="Pipeline root cause",
        description="Trace upstream/downstream data flow and inspect producer outputs when a table or S3 output is empty or stale.",
        triggers=("root cause", "empty", "0 rows", "missing", "stale", "upstream", "downstream", "lineage"),
        tool_plan=("search_kb", "trace_pipeline", "execute_sql", "list_s3", "fetch_s3", "bash"),
        metadata={"requires_lineage": True},
    ),
    TaskRecipe(
        id="task:schema_inventory",
        name="Schema inventory",
        description="Answer which tables/columns exist for a schema or business concept using KB metadata before live inspection.",
        triggers=("schema", "tables", "columns", "inventory", "which table", "what tables"),
        tool_plan=("search_kb", "inspect_table"),
        metadata={
            "bounded": True,
            "preferred_tables": ["prod.monitoring.provider_combined_audit", "prod.monitoring.combined_audit"],
        },
    ),
    TaskRecipe(
        id="task:bounded_kb_lookup",
        name="Bounded KB lookup",
        description="Answer a documentation or table-selection question from KB citations without broadening into live tools.",
        triggers=("use search_kb", "kb lookup", "documentation", "docs", "bounded"),
        tool_plan=("search_kb",),
        metadata={"bounded": True},
    ),
)


__all__ = ["BASE_TASK_RECIPES"]
