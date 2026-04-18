---
name: sql_investigation
description: When to use execute_sql, which datasource to target, and partition rules.
keywords: [sql, redshift, mysql, analytics, monitoring, priceeye, partition, query, select, table, sales_date, anomalies]
tier: high
---

## Running SQL on Redshift / MySQL

Use `execute_sql(query, datasource=None)` for read-only SELECT/WITH queries. The
datasource is auto-detected from table names; pass it explicitly to force:
- `redshift_analytics` — analytics workgroup, default for `analytics.*`, `prod.analytics.*`, `prod.monitoring.*`.
- `redshift_core` — monitoring cluster for `prod.monitoring.*` investigations on the core workgroup.
- `mysql_priceeye` — PriceEye MySQL for `priceeye.*` lookup tables.

Every query gets a LIMIT clamp (default 1000, max 120,000). Single statement only.

## Partition rules — critical for Redshift

Always include the required partition filter in WHERE, or the query will scan the
whole table and time out or throttle neighbours:

| Table | Required partitions |
|---|---|
| `prod.analytics.market_level_anomalies_v3/v4` | `sales_date`, `customer` |
| `prod.analytics.market_level_analysis_v2` | `sales_date`, `customer` |
| `prod.analytics.segment_level_analysis_v2` | `sales_date`, `customer` |
| `prod.monitoring.provider_combined_audit` | `sales_date` |
| `prod.monitoring.combined_audit` | `sales_date` |
| `prod.common_output.common_output_format` | `sales_date` |

Partition keys come live from Glue (via `glue_get_table`) when the catalog is
reachable, with the static map above as a fallback when Glue can't answer.

## Patterns

- Start narrow: one day, one customer, one provider. Expand only after you see
  the data shape.
- Prefer `prod.*` tables unless the user explicitly asks for dev / local data.
- For EDA: `execute_sql` first → save as dataset → `run_python` with
  `load_dataset(dataset_id)` for transforms and plots.

## When NOT to use execute_sql

- Raw object storage → `fetch_s3(bucket, key_or_prefix)`
- Schema/partition inspection → `inspect_table(table_name)` or
  `glue_get_table(database, name)` when you need the authoritative metadata.
- Code-resolution (provider / site / customer aliases) → `resolve_codes(text)`.
