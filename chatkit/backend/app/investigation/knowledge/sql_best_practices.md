# SQL Best Practices for DS Chat Next Gen

## Safety Baseline
- Use read-only SQL (`SELECT` or `WITH`) only.
- Run a single SQL statement per extraction call.
- Always use a bounded `LIMIT` for preview/exploration.

## Query Style
- Prefer fully qualified table names.
- Avoid `SELECT *` for production summaries; pick explicit columns after preview.
- Start with narrow previews, then iterate.

## Partition Guidance (Advisory)
- Prefer filtering by `sales_date` when available.
- Prefer filtering by `customer` when available.
- For broad scans, explicitly state scope/cost caveats.

## Investigation Flow
1. inspect table metadata.
2. run preview query.
3. materialize offline dataset.
4. analyze with pandas.
5. return answer with lineage.
