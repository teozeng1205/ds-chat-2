# DS Chat Next-Gen Knowledge Base

This local knowledge base powers table routing, relationship mapping, and partition-safe extraction.

## Core Principles

- Environment is fixed to `3VDEV`.
- Every DB extraction must use a `table_id` defined in `tables/*.yaml`.
- Required partitions come from `partition_policy.required_predicates`.
- If required partition values are missing, request clarification before querying.
- Multi-source joins are done offline on local dataset artifacts.

## Artifacts Contract

Per user turn, datasets are materialized to:

`chatkit/backend/.runtime/workspaces/{thread_id}/{turn_id}/datasets/{dataset_id}/`

Each dataset folder contains:

- `data.parquet`
- `preview.csv`
- `manifest.json`
