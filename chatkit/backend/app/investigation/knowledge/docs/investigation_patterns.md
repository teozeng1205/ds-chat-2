# Investigation Patterns — Table / Partition / Key-Columns Catalog

This is the canonical "which table answers which question" reference for PriceEye.
The agent searches this doc via `search_kb` when it needs to pick a table for a given
concept (site issues, anomalies, competitive position, billing, etc.).

All tables listed below are the **prod** variants on the default reader. Substitute
`local.*` only when the user explicitly asks for dev data.

## Pattern → Table mapping

| Pattern                          | Table (datasource)                                                                   | Partition(s)             | Key columns |
|----------------------------------|---------------------------------------------------------------------------------------|--------------------------|-------------|
| Site issues                      | `prod.monitoring.provider_combined_audit` (redshift_core)                             | `sales_date`             | `issue_sources`, `issue_reasons`, `providercode`, `sitecode`, `filterreason` |
| Audit lifecycle / request trace  | `prod.monitoring.combined_audit` (redshift_core)                                      | `sales_date`             | `id`, `customer`, `providercode`, `sitecode`, `issue_source`, `issue_reason`, `filterreason`, `response_status`, `delivery_status`, `delivery_type`, `response_itinerarycount` |
| Market anomalies                 | `prod.analytics.market_level_anomalies_v3` (redshift_analytics)                        | `sales_date` + `customer`| `any_anomaly=1`, `impact_score`, `mkt`, `seg`, `top_offenders`, `cp`, `dow` |
| Segment anomalies                | `prod.analytics.segment_level_anomalies_v3` (redshift_analytics)                       | `sales_date` + `customer`| `any_anomaly=1`, `impact_score`, `mkt`, `seg`, `airline_code`, `cabin`, `anomaly_type` |
| Competitive position             | `prod.analytics.market_level_anomalies` or `segment_level_anomalies_v3` (redshift_analytics) | `sales_date` + `customer` | `competitive_position`, `comparison_type`, `cp_score`, `top_offenders` |
| Common output / DCO              | `prod.common_output.common_output_format` (redshift_analytics)                         | `sales_date`             | `customer`, `origin`, `destination`, `carrier`, `channel`, `price_inc`, `price_exc`, `tax`, `cabin`, `trip_type` |
| Provider retry rates             | `prod.site_metrics.retry_metrics_v1` (redshift_core)                                   | `sales_date`             | `providercode`, `retry_rate_pct`, `total_requests` |
| Provider cache hit rates         | `prod.site_metrics.cache_metrics_v1` (redshift_core)                                   | `sales_date`             | `providercode`, `sitecode`, `cache_hit_rate`, `cache_miss_rate` |
| Provider TPS capacity            | `prod.site_metrics.capacity_final` (redshift_core)                                     | `sales_date`             | `providercode`, `capacity_tph` (IQR-filtered; QL2 ≥180, SS ≥3600 floor patches) |
| Billing / request counts         | `billing_db.customer_daily_requests_v3` (redshift_core Spectrum)                       | `sales_date`             | `customer`, `total_reqs`, `billable_requests`, `GDS_scheduled`, `OTA_scheduled`, `MSE_scheduled`, `true_site_issues` |
| PAX/MIDT bookings                | `prod.analytics.pax_midt` (redshift_analytics)                                         | `sales_date` + `customer`| `origin`, `destination`, `carrier`, `cabin`, `ap_band`, `pax_count` |
| OAG seat supply                  | `prod.analytics.oag_score_v2` (redshift_analytics)                                     | `sales_date` + `customer`| `origin`, `destination`, `carrier`, `cabin`, `seat_count`, `market_share_pct` |
| Revenue score                    | `prod.analytics.revenue_score_v1` (redshift_analytics)                                 | `sales_date` + `customer`| `origin`, `destination`, `carrier`, `cabin`, `avg_price`, `pax_count`, `estimated_revenue` |
| Tax regression coefficients      | `prod.tax_reg.tax_reg_output_v1` (redshift_analytics)                                  | `sales_date`             | `pos`, `od`, `carrier`, `m`, `b`, `r2`, `correlation` |
| Tax regression MySQL             | `taxregression.tax_regression_v1` (mysql_priceeye)                                     | —                        | overwritten every Tuesday |
| Collection anomalies (S3)        | S3 `s3-atp-3victors-3vprod-use1-collection-anomalies`                                  | date in path             | `collection-customer/v1/YYYY/MM/DD/` — see `s3_buckets.md` |
| Customer collection health       | `billing_db.customer_daily_requests_v2` (redshift_core)                                | `sales_date`             | `customer`, `total_reqs`, `success`, `site_failed` |
| Table health / row counts        | `metadata.table_row_counts` (redshift_analytics)                                       | `checked_date`           | `table_name`, `row_count`, `last_updated` |

## Key distinctions

- `combined_audit` uses **singular** `issue_source` / `issue_reason`;
  `provider_combined_audit` uses **plural** `issue_sources` / `issue_reasons`. Do not mix.
- `billing_db` is a Glue external schema — use `billing_db` as the schema name in
  Redshift Spectrum queries.
- Billing metric definitions:
  - `GDS_scheduled` = sitecode `1G`
  - `OTA_scheduled` = sitecode IN `EXP / DES / BKG / OBZ / PLN / TCY / EDR`
  - `MSE_scheduled` = sitecode IN `SKYS / GGL / KYK`
  - `billable_requests` = `requested_by_customers − true_site_issues`

## Table fallback strategy

When a query returns 0 rows, walk this chain in order:

### Step 0 — Discover tables if uncertain

Run a catalog discovery query via `execute_sql` to list tables in a schema:

```sql
-- Redshift: list tables in a schema (redshift_analytics or redshift_core)
SELECT DISTINCT table_schema, table_name
FROM svv_columns
WHERE table_schema = 'analytics'
ORDER BY table_name;
```

Or search by keyword:
```sql
SELECT DISTINCT table_schema, table_name
FROM svv_columns
WHERE table_name LIKE '%anomal%'
ORDER BY table_schema, table_name;
```

For MySQL (`datasource=mysql_priceeye`):
```sql
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema IN ('analytics', 'priceeye')
ORDER BY table_schema, table_name;
```

### Step 1 — Verify the partition exists

Call `inspect_table(table_name)` to check which `sales_date` (and `customer`) partitions
are actually loaded. A 0-row result often means the partition simply hasn't been written
yet. Report the latest available partition to the user and offer to rerun on that date.

### Step 2 — Try an alternate table version

If the table is versioned (`_v4`, `_v3`, `_v2`), try the adjacent version:
- anomalies: v4 → v3 → v2
- analysis: v2 → v1
- billing: v3 → v2 → v1

### Step 3 — Check the S3 equivalent (applies to ALL tables)

Many Redshift tables have S3 mirrors written by the same pipelines. Consult `s3_buckets.md`
for the full table → bucket/key mapping. Use `fetch_s3` with the matching path for the
table's date + customer.

### Step 3.5 — Consult the federated MySQL config tables

When rows are missing **for a specific customer** (or only for some customers), the
most common cause is a config drop — the customer isn't onboarded, the site isn't
scheduled, or a lookup row got deleted. Redshift exposes the live MySQL config
tables directly via federated schemas, so there's no need to hop through a
separate `mysql_priceeye` connector:

```sql
-- Is DE configured / enabled?
SELECT * FROM federated_priceeye.customer_defaults WHERE customer = 'DE';

-- Which sites is DE scheduled against?
SELECT * FROM federated_priceeye.site_hierarchy WHERE customer = 'DE';

-- Global config (airport → city mappings, error messages, etc.)
SELECT * FROM federated_metadata.airportlocation_extra WHERE IATA = 'JFK';
```

All `federated_*` schemas are routed to `redshift_analytics` automatically by the
datasource router. See `federated_schemas.md` for the full catalog (priceeye,
metadata, scheduling, analytics). These tables are **small config tables**, so no
partition filter is required.

If the federated lookup shows the customer is missing / disabled / misconfigured,
that IS the answer — no deeper walk needed.

### Step 4 — Try `local.*` ONLY if the user requests dev data

`local.*` schemas are DEV copies — never the default. Use only when the user specifically
asks for dev, local, or staging.

### Step 5 — Report and explain

If all fallbacks are exhausted, tell the user clearly: which table you tried, what
partitions are available, and whether the table is dev-only / has no prod equivalent.
