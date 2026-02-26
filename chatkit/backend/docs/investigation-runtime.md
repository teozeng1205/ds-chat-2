# DS Chat Investigation Runtime

## Overview
The runtime unifies data investigations into a single autonomous shell-first, knowledge-guided operator flow.

## High-level flow
1. Bootstrap credentials with `assume 3VDEV`.
2. Run an autonomous loop (`PLAN -> ACT -> OBSERVE -> CHECK_DONE -> FINALIZE`).
3. Resolve entities (`provider/site/customer`) from local common codes and MySQL fallback.
4. Retrieve table hints from local KB and inspect unknown tables on demand.
5. Extract SQL/S3 data into offline dataset artifacts.
6. Analyze with pandas (including deep table EDA).
7. Return conclusions with lineage (`run_id`, `dataset_ids`, key queries/sources, caveats).
8. Cleanup dataset artifacts after response while retaining manifests.

## Knowledge sources
- `backend/app/investigation/knowledge/tables.md`
- `backend/app/investigation/knowledge/common_codes.json`
- `backend/app/investigation/knowledge/task_cards/*.md`
- `backend/app/investigation/knowledge/sql_best_practices.md`
- `backend/app/investigation/knowledge/docs/*.md`

## Workspace layout
`backend/.work/sessions/<thread_id>/<run_id>/`
- `datasets/<dataset_id>/data.parquet` (csv fallback)
- `analysis/<analysis_id>.json`
- `logs/activity.jsonl`
- `manifest.json`

## Tooling surface
- `investigate_issue`
- `run_table_eda`
- `resolve_entities`
- `retrieve_knowledge`
- `browse_knowledge_files`
- `inspect_table_metadata`
- `extract_sql_to_dataset`
- `extract_s3_to_dataset`
- `run_dataframe_analysis`
- `operator_run_python`
- `cleanup_session_workspace`
