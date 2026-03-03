# ds-customer-monitoring

> An hourly + daily data pipeline that unloads PriceEye audit tables from Redshift to S3/Glue, producing deduplicated audit snapshots, combined audit views, customer/provider-centric datasets, billing summaries, and response-duplication metrics for downstream monitoring and analytics.

> **Current branch**: `develop` — this document reflects the `develop` branch. The `master` branch represents what is running in production; the documented state may differ from production.

---

## Architecture Overview

```
[EventBridge cron: hourly at :10 UTC]
      │
      ▼
[Step Function: unload-monitoring-step-function]
      │
      ├──── Phase 1 (Parallel) ─────────────────────────────────────────────────────┐
      │  unload-deduped-provider-request-audit  → s3://..-deduped-datasets/v1/...  │
      │  unload-deduped-provider-request-audit-detail → s3://..-deduped-datasets/  │
      │  unload-deduped-provider-response-audit → s3://..-deduped-datasets/v1/...  │
      │  unload-deduped-retry-audit             → s3://..-deduped-datasets/v1/...  │
      │  unload-deduped-collection-run-audit    → s3://..-deduped-datasets/v1/...  │
      └─────────────────────────────────────────────────────────────────────────────┘
      │
      ├──── Phase 2 (Parallel) ─────────────────────────────────────────────────────┐
      │  unload-deduped-global-filter-audit-summary → s3://..-deduped-datasets/    │
      │  unload-deduped-enrichment-audit            → s3://..-deduped-datasets/    │
      │  unload-deduped-cache-loader-audit          → s3://..-deduped-datasets/    │
      │  unload-deduped-packager-audit              → s3://..-deduped-datasets/    │
      │  unload-deduped-delivery-audit              → s3://..-deduped-datasets/    │
      └─────────────────────────────────────────────────────────────────────────────┘
      │
      ├──── Phase 3 (Parallel) ──────────────────────────────────┐
      │  unload-combined-audit (today)   → s3://..-deduped-datasets/v1/combined_audit │
      │  unload-combined-audit (yesterday) → same table, yesterday partition          │
      └──────────────────────────────────────────────────────────┘
      │
      ▼
[ECS Fargate: monitoring-verify-dedupe]   ← validate tables & refresh Redshift views
      │
      ├──── Phase 5 (Parallel) ──────────────────────────────────────────────────────┐
      │  provider-centric-dataset-unload → s3://..-provider-monitor/v1/...           │
      │  customer-centric-dataset-unload → s3://..-customer-monitor/v2/...           │
      └──────────────────────────────────────────────────────────────────────────────┘


[EventBridge cron: hourly :30 from 17-23 UTC  +  daily 02:00 UTC]
      │
      ▼
[Step Function: CustomerCentricStepFunction]
      └──► customer-centric-dataset-unload  → s3://..-customer-monitor/v2/...


[EventBridge cron: daily 02:00 UTC]
      │
      ▼
[Step Function: ProviderCentricStepFunction]
      └──► provider-centric-dataset-unload  → s3://..-provider-monitor/v1/...


[Glue Trigger: daily 10:45 UTC]  ── billing-customer-daily-request-unload   → s3://..-billing/v1/...
[Glue Trigger: daily 10:45 UTC]  ── billing-customer-daily-request-internal-unload → s3://..-billing/v2/...
[Glue Trigger: daily 10:45 UTC]  ── billing-customer-daily-request-granular-unload → s3://..-billing/v3/...


[Glue Trigger: 06:00 UTC (today)]   ─┐
[Glue Trigger: 18:00 UTC (today)]   ─┼──► response-dupes-unload → s3://..-provider-monitor/v1/response_dupes/
[Glue Trigger: 02:00 UTC (yesterday)]┘
```

---

## Orchestration

### Step Function: `unload-monitoring-step-function`

- **Trigger**: EventBridge cron — every hour at 10 minutes past the hour (`cron(10 * * * ? *)`)
- **Pipeline**:
  1. **Parallel** — `unload-deduped-provider-request-audit`, `unload-deduped-provider-request-audit-detail`, `unload-deduped-provider-response-audit`, `unload-deduped-retry-audit`, `unload-deduped-collection-run-audit`
  2. **Parallel** — `unload-deduped-global-filter-audit-summary`, `unload-deduped-enrichment-audit`, `unload-deduped-cache-loader-audit`, `unload-deduped-packager-audit`, `unload-deduped-delivery-audit`
  3. **Parallel** — `unload-combined-audit` (today) + `unload-combined-audit` (yesterday) — both run the same job with different `--SALES_DATE` arguments
  4. **ECS Task** — `monitoring-verify-dedupe` (validate tables, refresh Redshift views)
  5. **Parallel** — `provider-centric-dataset-unload` + `customer-centric-dataset-unload`
- **Definition**: `source/deploy/definitions/unload-monitoring-step-function.asl.json`

### Step Function: `CustomerCentricStepFunction`

- **Trigger**: Two EventBridge rules:
  - Hourly at :30 from 17:00–23:00 UTC (`cron(30 17-23 * * ? *)`) with `DAYSOFFSET=-1` (next-day data)
  - Daily at 02:00 UTC (`cron(0 2 * * ? *)`) with `DAYSOFFSET=1` (yesterday's data)
- **Pipeline**: `customer-centric-dataset-unload` _(single step)_
- **Definition**: `source/deploy/definitions/customer-centric-step-function.asl.json`

### Step Function: `ProviderCentricStepFunction`

- **Trigger**: EventBridge rule — daily at 02:00 UTC (`cron(0 2 * * ? *)`) with `DAYSOFFSET=1`
- **Pipeline**: `provider-centric-dataset-unload` _(single step)_
- **Definition**: `source/deploy/definitions/provider-centric-step-function.asl.json`

---

## Components

_Ordered by when they run in the primary monitoring pipeline — earliest first._

---

### unload-deduped-provider-request-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 1 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads raw provider request audit records from the Redshift table `local.priceeye_audits.provider_request_audit` and unloads a deduplicated snapshot to S3 as Parquet, partitioned by sales date. These records represent individual shopping requests sent to airline/GDS providers. Retries up to 3 times on failure (60s interval) within the Step Function.

**Input**:
- Redshift: `local.priceeye_audits.provider_request_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/provider_request_audit/`
- Glue table: `deduped_provider_request_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-provider-request-audit-detail

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 1 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads the `provider_request_audit_detail` table from Redshift (along with `provider_request_audit` for joins) and unloads a deduplicated snapshot to S3. This table contains customer-level detail for each provider request, including customer name, collection name, reference, site category, and customer sales date. Retries up to 3 times on failure.

**Input**:
- Redshift: `local.priceeye_audits.provider_request_audit_detail`, `local.priceeye_audits.provider_request_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/provider_request_audit_detail/`
- Glue table: `deduped_provider_request_audit_detail` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-provider-response-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 1 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads the `provider_response_audit` table from Redshift, which tracks provider responses including status (success/failed), error messages, and itinerary counts. Writes a deduplicated Parquet snapshot to S3 partitioned by sales date. Retries up to 3 times on failure.

**Input**:
- Redshift: `local.priceeye_audits.provider_response_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/provider_response_audit/`
- Glue table: `deduped_provider_response_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-retry-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 1 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads retry audit records from Redshift — tracking which provider requests were retried, with which alternate provider/site, and with what outcome. Unloads deduplicated Parquet to S3. Retries up to 3 times on failure.

**Input**:
- Redshift: `local.priceeye_audits.retry_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/retry_audit/`
- Glue table: `deduped_retry_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-collection-run-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 1 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads collection run audit records from Redshift, which track scheduled collection runs — including start/expected delivery times and customer timezone metadata. Writes deduplicated Parquet to S3. Retries up to 3 times on failure.

**Input**:
- Redshift: `local.priceeye_audits.collection_run_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/collection_run_audit_test/`
- Glue table: `deduped_collection_run_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-global-filter-audit-summary

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 2 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads global filter audit summary records from Redshift, which count itineraries before and after global filtering (OAG, Market Date Blacklist, etc.) for each provider request. Writes a deduplicated Parquet snapshot to S3.

**Input**:
- Redshift: `local.priceeye_audits.global_filter_audit_summary`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/global_filter_audit_summary/`
- Glue table: `deduped_global_filter_audit_summary` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-enrichment-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 2 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads enrichment audit records from Redshift, which capture the enrichment pipeline outcomes for each itinerary — including brand, tax, OAG, booking code, and directional price enrichment success/fail counts. Writes deduplicated Parquet to S3.

**Input**:
- Redshift: `local.priceeye_audits.enrichment_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/enrichment_audit/`
- Glue table: `deduped_enrichment_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-cache-loader-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 2 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads cache loader audit records from Redshift, which track how many itineraries were loaded into cache for each provider request. Writes a deduplicated Parquet snapshot to S3.

**Input**:
- Redshift: `local.priceeye_audits.cache_loader_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/cache_loader_audit/`
- Glue table: `deduped_cache_loader_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-packager-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 2 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads packager audit records from Redshift, which track how itineraries were packaged and delivered to customers — including packaging name, file URI, record count, and whether a substitute provider was used. Writes a deduplicated Parquet snapshot to S3.

**Input**:
- Redshift: `local.priceeye_audits.packager_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/packager_audit/`
- Glue table: `deduped_packager_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-delivery-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 2 of `unload-monitoring-step-function`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads delivery audit records from Redshift, tracking file delivery status (success/failed), delivery type, archive file key, and failure reasons for each customer collection. Writes a deduplicated Parquet snapshot to S3.

**Input**:
- Redshift: `local.priceeye_audits.delivery_combiner_audit`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/delivery_audit/`
- Glue table: `deduped_delivery_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-combined-audit

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Step Function step (Phase 3 of `unload-monitoring-step-function`) — run twice in parallel, once for "today" and once for "yesterday"
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Builds the central combined audit table by joining all 10 deduped audit tables (provider request, request detail, response, retry, collection run, global filter, enrichment, cache loader, packager, delivery) from the Redshift `local.monitoring.*` views. For each provider request, it assembles a single wide row with the full lifecycle — from observation through response, retry, enrichment, cache, packaging, and delivery. Applies error mapping logic to classify issue sources and reasons. Writes Parquet to S3 partitioned by sales date. This table is the primary input for customer-centric, provider-centric, and billing jobs downstream.

**Input**:
- Redshift (local.monitoring views): `deduped_provider_request_audit`, `deduped_provider_request_audit_detail`, `deduped_provider_response_audit`, `deduped_retry_audit`, `deduped_collection_run_audit`, `deduped_global_filter_audit_summary`, `deduped_enrichment_audit`, `deduped_cache_loader_audit`, `deduped_packager_audit`, `deduped_delivery_audit`
- Redshift metadata: `local.federated_priceeye.error_mapping`
- Config: `s3://config-server-{env}/default/deduped-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/combined_audit/`
- Glue table: `combined_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### monitoring-verify-dedupe _(ECS Fargate)_

**Type**: ECS Fargate Task
**Trigger**: Step Function step (Phase 4 of `unload-monitoring-step-function`) — runs after combined audit completes
**Compute**: Configured via `ecs-3v{env}-use1-price-eye` cluster; network via VPC app subnets + FMS security group

**What it does**: Validates that all deduped Glue tables are consistent and up-to-date, then refreshes Redshift views (in `local.monitoring`) that read from those S3-backed tables. Ensures that downstream jobs (customer-centric, provider-centric, billing) read fresh, validated data.

**Input**:
- Glue catalog tables in `glue-atp-3victors-{env}-use1-monitoring_db`

**Output**:
- Refreshed Redshift external views (no S3 output)

---

### customer-centric-dataset-unload

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**:
  - Step Function step (Phase 5 of `unload-monitoring-step-function`)
  - `CustomerCentricStepFunction`: hourly at :30 from 17:00–23:00 UTC (DAYSOFFSET=-1), and daily at 02:00 UTC (DAYSOFFSET=1)
  - Standalone Glue Trigger: daily at 02:00 UTC
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads the combined audit table from Redshift (`local.monitoring.combined_audit`) and produces a customer-centric view, filtering to main/substitute site categories and applying customer-specific sales date logic (e.g., AA_UK, AA_B3/B4, Advito date adjustments). Outputs a wide fact table with request-level detail including customer, route, provider, cabin, filter reason, response/retry/packager/delivery status, and itinerary counts, partitioned by year/month/day. Scans a window of dates surrounding the target sales date (day-before, target, day-after) to capture late-arriving records.

**Input**:
- Redshift: `local.monitoring.combined_audit`
- Config: `s3://config-server-{env}/default/customer-combined-audit-config.properties`
  - Input table: `local.monitoring.combined_audit`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-customer-monitor/v2/customer-combined-audit/{YYYY}/{MM}/{DD}/`
- Glue table: `customer_combined_audit_v2` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### provider-centric-dataset-unload

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**:
  - Step Function step (Phase 5 of `unload-monitoring-step-function`)
  - `ProviderCentricStepFunction`: daily at 02:00 UTC (DAYSOFFSET=1)
  - Standalone Glue Trigger: daily at 02:00 UTC
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads from the combined audit table and produces a provider-centric view — one row per unique provider request ID, aggregating across customer dimensions using LISTAGG. Enriches each request with origin/destination city and country codes from airport and city location metadata tables. Applies response status normalization and issue classification logic. Computes advance purchase (AP) and length of stay (LOS) from schedule and departure dates. Outputs Parquet partitioned by observation date.

**Input**:
- Redshift: `local.monitoring.combined_audit`
- Redshift metadata: `local.federated_metadata.airportlocation_extra`, `local.federated_metadata.citylocation_extra`
- Config: `s3://config-server-{env}/default/provider-combined-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-provider-monitor/v1/provider-combined-audit/{YYYY}/{MM}/{DD}/`
- Glue table: `provider_combined_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### response-dupes-unload

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Three standalone Glue Triggers:
  - Daily at 06:00 UTC (`DAYSOFFSET=0` → today)
  - Daily at 18:00 UTC (`DAYSOFFSET=0` → today)
  - Daily at 02:00 UTC (`DAYSOFFSET=1` → yesterday)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Identifies duplicate provider responses — cases where the same provider request received multiple responses. Reads from `local.priceeye_audits.provider_response_audit` joined with `local.priceeye_audits.provider_request_audit` to surface response duplication patterns. Writes results to S3 for downstream analysis.

**Input**:
- Redshift: `local.priceeye_audits.provider_response_audit`, `local.priceeye_audits.provider_request_audit`
- Config: `s3://config-server-{env}/default/provider-combined-audit-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-provider-monitor/v1/response_dupes/`
- Glue table: `response_dupes` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### billing-customer-daily-request-unload

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Standalone Glue Trigger — daily at 10:45 UTC
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads the combined audit table from Redshift and produces a customer-level daily billing summary (v1). Groups by customer and sales date, counting total requests, site-category breakdowns (GDS/OTA/MSE), and operational outcomes (polled, cached, filtered, enriched, retried, success, failed, billable). This is the foundational billing table used for customer invoicing.

**Input**:
- Redshift: `prod.monitoring.combined_audit`
- Config: `s3://config-server-{env}/default/billing-config.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-billing/v1/customer_daily_requests/`
- Glue table: `customer_daily_requests_v1` in `glue-atp-3victors-{env}-use1-billing_db`

**Table Schema** (`customer_daily_requests_v1`):

| Column | Type |
|--------|------|
| customer | varchar(256) |
| cust_run_dt | varchar(256) |
| total_reqs | bigint |
| requested_by_customers | bigint |
| gds_scheduled | bigint |
| ota_scheduled | bigint |
| mse_scheduled | bigint |
| unq_scheduled | bigint |
| polled | bigint |
| cached | bigint |
| filtered | bigint |
| enrichment | bigint |
| retry_preemptive | bigint |
| success | bigint |
| failed | bigint |
| site_failed | bigint |
| bad_requests | bigint |
| true_site_issues | bigint |
| billable_requests | bigint |

_Partition key: `sales_date` (bigint, YYYYMMDD)_

---

### billing-customer-daily-request-internal-unload

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Standalone Glue Trigger — daily at 10:45 UTC
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Extends the v1 billing summary by adding `providercode`, `customer_site_code`, and `customersitetype` dimensions. Reads from `prod.monitoring.combined_audit` joined with the `customer_site_code` lookup table from federated PriceEye. Produces the v2 billing table, which enables billing breakdowns by provider and site type.

**Input**:
- Redshift: `prod.monitoring.combined_audit`, `local.federated_priceeye.customer_site_code`
- Config: `s3://config-server-{env}/default/billing-config-internal-v2.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-billing/v2/customer_daily_requests/`
- Glue table: `customer_daily_requests_v2` in `glue-atp-3victors-{env}-use1-billing_db`

**Table Schema** (`customer_daily_requests_v2`): Same columns as v1, plus:

| Column | Type |
|--------|------|
| providercode | varchar(256) |
| customer_site_code | varchar(256) |
| customersitetype | varchar(256) |

_Partition key: `sales_date` (bigint)_

---

### billing-customer-daily-request-granular-unload

**Type**: AWS Glue ETL Job (Glue 4.0)
**Trigger**: Standalone Glue Trigger — daily at 10:45 UTC
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Produces the most granular billing table (v3) by adding `customercollectionname` and `reference` as grouping dimensions on top of the v2 schema. This enables collection-level and reference-level billing breakdowns, read from `prod.monitoring.combined_audit` and the `customer_site_code` lookup.

**Input**:
- Redshift: `prod.monitoring.combined_audit`, `local.federated_priceeye.customer_site_code`
- Config: `s3://config-server-{env}/default/billing-config-granular-v3.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-billing/v3/customer_daily_requests/`
- Glue table: `customer_daily_requests_v3` in `glue-atp-3victors-{env}-use1-billing_db`

**Table Schema** (`customer_daily_requests_v3`): Same columns as v2, plus:

| Column | Type |
|--------|------|
| customercollectionname | varchar(256) |
| reference | varchar(256) |

_Partition key: `sales_date` (bigint)_

---

## Glue Databases

| Database | Tables |
|----------|--------|
| `glue-atp-3victors-{env}-use1-billing_db` | `customer_daily_requests_v1`, `customer_daily_requests_v2`, `customer_daily_requests_v3` |
| `glue-atp-3victors-{env}-use1-monitoring_db` | `deduped_provider_request_audit`, `deduped_provider_request_audit_detail`, `deduped_provider_response_audit`, `deduped_retry_audit`, `deduped_collection_run_audit`, `deduped_global_filter_audit_summary`, `deduped_enrichment_audit`, `deduped_cache_loader_audit`, `deduped_packager_audit`, `deduped_delivery_audit`, `combined_audit`, `customer_combined_audit_v2`, `provider_combined_audit`, `response_dupes` |

---

## S3 Buckets

| Bucket | Contents |
|--------|----------|
| `s3-atp-3victors-{env}-use1-billing` | Billing Parquet (v1/v2/v3 customer daily requests) |
| `s3-atp-3victors-{env}-use1-deduped-datasets` | Deduped audit snapshots + combined audit |
| `s3-atp-3victors-{env}-use1-customer-monitor` | Customer-centric dataset (`customer_combined_audit_v2`) |
| `s3-atp-3victors-{env}-use1-provider-monitor` | Provider-centric dataset + response dupes |

Both `customer-monitor` and `billing` buckets have EventBridge notifications enabled (S3 → EventBridge).

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| Glue ETL Jobs | 14 |
| ECS Fargate Tasks | 1 (`monitoring-verify-dedupe`) |
| Step Functions | 3 (`unload-monitoring-step-function`, `CustomerCentricStepFunction`, `ProviderCentricStepFunction`) |
| EventBridge Rules | 4 (1 hourly for monitoring SF, 2 for customer-centric SF, 1 daily for provider-centric SF) |
| Standalone Glue Triggers | 6 (3 billing daily, 2 response-dupes daily, 1 response-dupes nightly) |
| Glue Databases | 2 |
| Glue Tables (CloudFormation-defined) | 3 (billing v1/v2/v3; monitoring_db tables are catalog-registered by jobs) |
| S3 Buckets | 4 |
