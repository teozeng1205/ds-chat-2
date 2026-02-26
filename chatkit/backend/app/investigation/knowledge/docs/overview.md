# DS Chat Investigation Knowledge v3

This local knowledge base supports autonomous, generic investigation tasks.

## Principles

- Environment bootstrap is always `assume 3VDEV`.
- Guidance is knowledge-driven, not intent-template-driven.
- Runtime tools are generic primitives: entity resolution, KB retrieval, metadata inspect, SQL/S3 extraction, python analysis, summarization.
- Partition filters are advisory and should be applied when useful.
- All fetched data is materialized to local dataset artifacts before downstream analysis.
- Conclusions must be backed by lineage (`run_id`, datasets, key sources/queries, caveats).

## Task Card Guidance

- Task cards in `task_cards/*.md` provide natural-language hints, candidate tables, and analysis suggestions.
- Cards are retrieval context only; they do not hardcode runtime branches.
- Unknown tables should follow discover-first flow:
  1. inspect metadata
  2. run bounded preview
  3. capture masked sample row
  4. persist discovered metadata for reuse

## Artifact Contract

Per run, artifacts are stored under:

`chatkit/backend/.work/sessions/<thread_id>/<run_id>/`

Key outputs:

- `datasets/<dataset_id>/data.parquet` (or csv fallback)
- `analysis/<analysis_id>.json`
- `logs/activity.jsonl`
- `manifest.json`
