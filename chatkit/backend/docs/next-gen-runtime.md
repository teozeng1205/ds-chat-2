# DS Chat Next Gen Runtime

## Overview
The next-generation runtime unifies monitoring/anomaly investigations into a single shell-first, knowledge-guided operator flow.

## High-level flow
1. Resolve entities (`provider/site/customer`) from local common codes and MySQL fallback.
2. Retrieve table/relationship hints from local knowledge base (`tables.md` + indexed metadata).
3. Extract SQL/S3 data into local offline datasets.
4. Analyze datasets with pandas and optional custom Python operator code.
5. Return conclusions with lineage (`run_id`, `dataset_id`, query/source metadata).
6. Cleanup dataset artifacts after response while retaining manifests.

## Knowledge sources
- `tables.md`
- `backend/app/investigation/common_codes.json`
- `backend/app/investigation/task_recipes.json`
- `backend/app/investigation/sql_best_practices.md`

## Workspace layout
`backend/.work/sessions/<thread_id>/<run_id>/`
- `datasets/*.parquet` (csv fallback)
- `analysis/*.json`
- `manifest.json`

## Tooling surface
- `investigate_issue`
- `resolve_entities`
- `retrieve_knowledge`
- `browse_knowledge_files`
- `inspect_table_metadata`
- `extract_sql_to_dataset`
- `extract_s3_to_dataset`
- `run_dataframe_analysis`
- `operator_run_python`
- `cleanup_session_workspace`
- `refresh_knowledge_base`
