# SQL Best Practices for DS Chat

## Safety Baseline (Enforced)
- Use read-only SQL (`SELECT` or `WITH`) only. No INSERT, UPDATE, DELETE, DROP.
- Run a single SQL statement per extraction call.
- The system automatically applies LIMIT (default 1000, max 120,000).

## Partition Filters (Enforced)
- ALWAYS filter by `sales_date` on tables that have it (YYYYMMDD format).
- ALWAYS filter by `customer` on analytics tables that require it.
- Missing partition filters will generate warnings and risk full table scans.
- Known required partitions:
  - `analytics.market_level_anomalies_v3`: sales_date, customer
  - `analytics.market_level_analysis_v2`: sales_date, customer
  - `analytics.segment_level_analysis_v2`: sales_date, customer
  - `prod.monitoring.provider_combined_audit`: sales_date
  - `prod.monitoring.combined_audit`: sales_date
  - `prod.common_output.common_output_format`: sales_date

## Query Style
- Prefer fully qualified table names (schema.table or catalog.schema.table).
- Avoid `SELECT *` for production summaries; pick explicit columns after preview.
- Start with narrow previews, then iterate with focused queries.
- Use aggregations (GROUP BY, COUNT, SUM) to summarize before pulling full data.

## Investigation Flow
1. Resolve entity codes (provider, site, customer).
2. Inspect table metadata if unfamiliar.
3. Run preview query with LIMIT.
4. Run focused query with proper partition filters.
5. Analyze with Python if needed.
6. Return answer with supporting data.
