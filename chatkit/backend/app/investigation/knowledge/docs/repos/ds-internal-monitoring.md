# ds-internal-monitoring

> Hourly pipeline that unloads deduplicated PriceEye audit data from Redshift into S3/Glue, assembles a
> combined audit table, validates integrity, and generates customer- and provider-centric monitoring
> datasets; a companion Lambda detects delivery-size anomalies and posts Slack alerts.

> **Note on branch**: The active branch at time of writing is `develop`. The `master`/`main` branch
> represents what is currently running in production; this document may reflect in-progress changes.

---

## Architecture Overview

```
[EventBridge cron: hourly at :10 past the hour]
              │
              ▼
[Step Function: unload-monitoring-step-function]
              │
              │  ── Stage 1 (Parallel) ──────────────────────────────────────────────────
              ├──► unload-deduped-provider-request-audit       → deduped_provider_request_audit
              ├──► unload-deduped-provider-request-audit-detail → deduped_provider_request_audit_detail
              ├──► unload-deduped-provider-response-audit      → deduped_provider_response_audit
              └──► unload-deduped-retry-audit                  → deduped_retry_audit
              │
              │  ── Stage 2 (Parallel) ──────────────────────────────────────────────────
              ├──► unload-deduped-global-filter-audit-summary  → deduped_global_filter_audit_summary
              ├──► unload-deduped-enrichment-audit             → deduped_enrichment_audit
              ├──► unload-deduped-cache-loader-audit           → deduped_cache_loader_audit
              ├──► unload-deduped-packager-audit               → deduped_packager_audit
              └──► unload-deduped-delivery-audit               → deduped_delivery_audit
              │
              │  ── Stage 3 (Parallel) ──────────────────────────────────────────────────
              ├──► unload-combined-audit (--SALES_DATE=today)     ─┐
              └──► unload-combined-audit (--SALES_DATE=yesterday) ─┘ → combined_audit
              │
              │  ── Stage 4 ──────────────────────────────────────────────────────────────
              └──► monitoring-verify-dedupe (ECS Fargate) — validates tables, refreshes views
              │
              │  ── Stage 5 (Parallel) ──────────────────────────────────────────────────
              ├──► provider-centric-dataset-unload  → provider_combined_audit
              └──► customer-centric-dataset-unload  → customer_combined_audit_v2

─────────────────────────────────────────────────────────────────────────────────────────────

[Glue Trigger: daily 2:00 AM UTC, daysoffset=1]        [Glue Trigger: daily 2:00 AM UTC, daysoffset=1]
              │                                                        │
              ▼                                                        ▼
  provider-centric-dataset-unload                        customer-centric-dataset-unload
    (standalone daily backfill)                            (standalone daily backfill)

[Glue Triggers: 6 AM UTC / 6 PM UTC (daysoffset=0) + 2 AM UTC (daysoffset=1)]
              │
              ▼
  response-dupes-unload

─────────────────────────────────────────────────────────────────────────────────────────────

[Lambda: delivery-anomalies]  (deployed and triggered independently)
              │
              ├── Scans S3 packager-archive for today's delivery files
              ├── Writes daily summary CSV → s3://3v-ds-pe-delivery-monitor/
              ├── Queries Redshift Spectrum for 35-day history
              └── Posts IQR anomaly alerts to Slack at 6 AM / 12 PM / 6 PM UTC
```

---

## Orchestration

### Step Function: `unload-monitoring-step-function`

- **Trigger**: EventBridge rule `unload-monitoring-stepfunction-task` — cron `cron(10 * * * ? *)` — fires every hour at 10 minutes past the hour
- **IAM**: Step Function execution role has permissions for `ecs:RunTask`, `glue:StartJobRun` / `glue:GetJobRun`, `s3:*`, and `states:*`
- **Pipeline order**:
  1. **Parallel** — 4 deduped raw-audit Glue jobs run concurrently (provider request, request detail, response, retry)
  2. **Parallel1** — 5 more deduped Glue jobs run concurrently (global filter, enrichment, cache loader, packager, delivery)
  3. **Parallel2** — 2 instances of `unload-combined-audit` run concurrently: one for today, one for yesterday
  4. **ValidateTablesAndRefreshViews** — `monitoring-verify-dedupe` ECS Fargate task validates the combined audit data and refreshes downstream views
  5. **Parallel3** — 2 derived-dataset Glue jobs run concurrently (provider-centric and customer-centric)
- **Error handling**: All Glue job steps in Stages 1–2 retry up to 3 times with a 60-second interval
- **Definition**: `source/deploy/definitions/unload-monitoring-step-function.asl.json`

---

## Components

> Ordered by pipeline stage — earliest to latest.

---

### unload-deduped-provider-request-audit

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 1 (also runnable manually via `MODULE_NAME=manual_historical_load_one_time`)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads raw provider request audit records from the Redshift table
`prod.priceeye_audits.provider_request_audit` for the current and adjacent sales dates (to handle
cross-midnight scheduling). Deduplicates and unloads the data as Snappy-compressed Parquet to S3,
then registers the new partition in the Glue Data Catalog. Uses config from
`config-server-{env}/default/deduped-audit-config.properties`.

**Input**:
- Redshift: `prod.priceeye_audits.provider_request_audit` (via "Monitoring Connection")

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/provider_request_audit/YYYY/MM/DD/`
- Glue table: `deduped_provider_request_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

**Script**: `s3://s3-atp-3victors{EnvironmentName}-use1-3v-glue-etl/unload-deduped-provider-request-audit.py`

---

### unload-deduped-provider-request-audit-detail

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 1 (parallel with above)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads provider request audit detail records from Redshift
(`prod.priceeye_audits.provider_request_audit_detail`), joining against the request audit table to
scope to the target sales date. Captures customer, collection, reference, site category, and POS
fields that are stored separately from the main request audit. Unloads as Parquet to S3 and
registers the Glue partition.

**Input**:
- Redshift: `prod.priceeye_audits.provider_request_audit_detail`
- Redshift: `prod.priceeye_audits.provider_request_audit` (join for date scoping)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/provider_request_audit_detail/YYYY/MM/DD/`
- Glue table: `deduped_provider_request_audit_detail` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-provider-response-audit

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 1 (parallel)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads provider response audit records from Redshift
(`prod.priceeye_audits.provider_response_audit`) for the current and adjacent sales dates. Captures
response status, error messages, itinerary counts, POS/site combinations, and response timestamps.
Unloads as Parquet and registers the Glue partition.

**Input**:
- Redshift: `prod.priceeye_audits.provider_response_audit`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/provider_response_audit/YYYY/MM/DD/`
- Glue table: `deduped_provider_response_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-retry-audit

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 1 (parallel)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads retry audit records from Redshift (`prod.priceeye_audits.retry_audit`),
capturing retry request IDs, retry provider and site codes, retry reasons, and retry response
statuses. Unloads as Parquet and registers the Glue partition.

**Input**:
- Redshift: `prod.priceeye_audits.retry_audit`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/retry_audit/YYYY/MM/DD/`
- Glue table: `deduped_retry_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-global-filter-audit-summary

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 2 (runs after Stage 1 completes)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads global filter audit summary records from Redshift
(`prod.priceeye_audits.global_filter_audit_summary`), capturing total itinerary counts before and
after global filtering per provider request. Used downstream to measure filtering effectiveness.
Unloads as Parquet and registers the Glue partition.

**Input**:
- Redshift: `prod.priceeye_audits.global_filter_audit_summary`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/global_filter_audit_summary/YYYY/MM/DD/`
- Glue table: `deduped_global_filter_audit_summary` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-enrichment-audit

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 2 (parallel)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads enrichment audit records from Redshift
(`prod.priceeye_audits.enrichment_audit`). Captures enrichment success and failure counts broken
down across enrichment types: brand, tax (combined/engine/regression/cache), cache, directional
price, booking code, operating carrier, fare basis code, and OAG-related fields. Unloads as Parquet
and registers the Glue partition.

**Input**:
- Redshift: `prod.priceeye_audits.enrichment_audit`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/enrichment_audit/YYYY/MM/DD/`
- Glue table: `deduped_enrichment_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-cache-loader-audit

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 2 (parallel)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads cache loader audit records from Redshift
(`prod.priceeye_audits.cache_loader_audit`), capturing itinerary counts loaded into cache per
provider request. Unloads as Parquet and registers the Glue partition.

**Input**:
- Redshift: `prod.priceeye_audits.cache_loader_audit`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/cache_loader_audit/YYYY/MM/DD/`
- Glue table: `deduped_cache_loader_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-packager-audit

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 2 (parallel)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads packager audit records from Redshift (`prod.priceeye_audits.packager_audit`),
capturing delivery timestamps, packaging names, file URIs, record counts, substitute usage, and
sub-provider/sub-site information per customer collection. Unloads as Parquet and registers the
Glue partition.

**Input**:
- Redshift: `prod.priceeye_audits.packager_audit`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/packager_audit/YYYY/MM/DD/`
- Glue table: `deduped_packager_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-deduped-delivery-audit

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 2 (parallel)
**Compute**: G.1X worker × 10, timeout 60 min

**What it does**: Reads delivery combiner audit records from Redshift
(`prod.priceeye_audits.delivery_combiner_audit`), capturing delivery IDs, types, status
(success/failed), archive file keys, failure reasons, group IDs, and delivery names per customer
collection. Unloads as Parquet and registers the Glue partition.

**Input**:
- Redshift: `prod.priceeye_audits.delivery_combiner_audit`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/delivery_audit/YYYY/MM/DD/`
- Glue table: `deduped_delivery_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### unload-combined-audit

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**: Step Function Stage 3 — called twice concurrently: once with `--SALES_DATE=today`, once with `--SALES_DATE=yesterday`
**Compute**: G.1X worker × 10, timeout 60 min, MaxConcurrentRuns: 10

**What it does**: The core assembly job. Joins all 9 deduped audit tables (already in Glue via Stages
1–2) using a large SQL query with CTEs: provider request → request detail → response → error
mapping → retry → global filter → enrichment → cache loader → packager → delivery. Produces a
wide, denormalized combined audit row per provider request, including response status, error
classification (issue_source, issue_reason from the error mapping metadata table), enrichment
breakdowns, and end-to-end pipeline status through delivery. Writes as Snappy Parquet and registers
the Glue partition. Reads config from `config-server-{env}/default/deduped-audit-config.properties`
using the connection name `Monitoring_Connection_Local`.

**Input**:
- Glue tables (via local Redshift Spectrum): `local.monitoring.deduped_provider_request_audit`, `local.monitoring.deduped_provider_request_audit_detail`, `local.monitoring.deduped_provider_response_audit`, `local.monitoring.deduped_retry_audit`, `local.monitoring.deduped_cache_loader_audit`, `local.monitoring.deduped_global_filter_audit_summary`, `local.monitoring.deduped_enrichment_audit`, `local.monitoring.deduped_packager_audit`, `local.monitoring.deduped_delivery_audit`
- Redshift: `local.monitoring_metadata.error_mapping` (for error classification)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-deduped-datasets/v1/combined_audit/YYYY/MM/DD/`
- Glue table: `combined_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### monitoring-verify-dedupe

**Type**: ECS Fargate Task
**Trigger**: Step Function Stage 4 — runs after both `unload-combined-audit` executions complete
**Compute**: Runs on the shared ECS cluster; network via VPC subnets / security groups imported from CloudFormation exports (`SubnetApp0/1/2`, `FMSSecuritygroupApp`)

**What it does**: Validates the completeness and integrity of the combined audit Glue tables
after each hourly unload cycle, and refreshes any downstream Redshift Spectrum or Glue views
that depend on the combined audit data. This is the gatekeeper before the derived-dataset jobs
run. (No source code is present in this repo; the Docker image is deployed separately.)

**Input**:
- Glue tables written by Stages 1–3 (reads via Redshift Spectrum or Glue)

**Output**:
- Validated/refreshed views in Redshift (exact targets defined in the Docker image)

---

### provider-centric-dataset-unload

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**:
  1. Step Function Stage 5 (hourly, receives `--DAYSOFFSET` from the Step Function input)
  2. Standalone Glue Trigger: daily at 2:00 AM UTC with `--DAYSOFFSET=1` (yesterday's data)
**Compute**: G.1X worker × 10, timeout 60 min, MaxConcurrentRuns: 10

**What it does**: Reads from the combined audit table (`prod.monitoring.combined_audit` via
"Monitoring Connection") and produces a provider-centric view of request/response activity.
The SQL aggregates requests by observation timestamp, filtering to rows where the observation
date matches the target sales date, then enriches with city/country codes from
`metadata.airportlocation_extra` and `metadata.citylocation_extra`. Collapses multi-valued
customer, site category, collection, and reference fields into pipe-delimited strings. Writes
as Snappy Parquet and registers the Glue partition. Config from
`config-server-{env}/default/provider-combined-audit-config.properties`.

**Input**:
- Redshift: `prod.monitoring.combined_audit` (via "Monitoring Connection")
- Redshift: `metadata.airportlocation_extra`, `metadata.citylocation_extra` (for geo enrichment)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-provider-monitor/v1/provider-combined-audit/YYYY/MM/DD/`
- Glue table: `provider_combined_audit` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### customer-centric-dataset-unload

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0)
**Trigger**:
  1. Step Function Stage 5 (hourly, receives `--DAYSOFFSET` from the Step Function input)
  2. Standalone Glue Trigger: daily at 2:00 AM UTC with `--DAYSOFFSET=1` (yesterday's data)
**Compute**: G.1X worker × 10, timeout 60 min, MaxConcurrentRuns: 10

**What it does**: Reads from the combined audit table (`prod.monitoring.combined_audit`) and
produces a customer-centric view. The SQL applies customer-specific date adjustment logic
(e.g., AA_UK, AA_B3/B4 overnight schedules, Advito same-day rules) to compute the correct
`customer_salesdate`, then filters to main and substitute site categories. Groups by customer,
reference, observation hour, provider, route (origin/destination), trip type, cabin, and
delivery/retry/packaging status — rolling up itinerary counts and request counts. Writes as
Snappy Parquet and registers the Glue partition. Config from
`config-server-{env}/default/customer-combined-audit-config.properties`.

**Input**:
- Redshift: `prod.monitoring.combined_audit` (via "Monitoring Connection")

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-customer-monitor/v2/customer-combined-audit/YYYY/MM/DD/`
- Glue table: `customer_combined_audit_v2` in `glue-atp-3victors-{env}-use1-monitoring_db`

---

### response-dupes-unload

**Type**: AWS Glue ETL Job (Python 3, Glue 4.0) — standalone (not part of the Step Function)
**Trigger**: Three standalone Glue Triggers:
  - `response-dupes-unload-twice-daily-trigger-1`: daily at 6:00 AM UTC, `DAYSOFFSET=0`
  - `response-dupes-unload-twice-daily-trigger-2`: daily at 6:00 PM UTC, `DAYSOFFSET=0`
  - `response-dupes-unload-daily-offset-trigger`: daily at 2:00 AM UTC, `DAYSOFFSET=1`
**Compute**: G.1X worker × 10, timeout 60 min, MaxConcurrentRuns: 10

**What it does**: Identifies duplicate provider responses — cases where the same provider request
received more than one response record (response_count > 1). Joins `provider_response_audit` to
`provider_request_audit` to scope by request sales date, then groups by provider request attributes
and counts occurrences. The result is a dataset of duplicate response events for quality monitoring.
Reads config from `config-server-{env}/default/provider-combined-audit-config.properties`.

**Input**:
- Redshift: `prod.priceeye_audits.provider_response_audit`
- Redshift: `prod.priceeye_audits.provider_request_audit` (join for request date scoping)

**Output**:
- S3: destination path and Glue table configured in `provider-combined-audit-config.properties`
  (under `pc_audit_*` keys; exact key not present in observed config — likely a separate key)

---

### delivery-anomalies (Lambda)

**Type**: AWS Lambda Function — standalone (not part of the Step Function)
**Trigger**: Invoked independently (event-driven or on a schedule, per deployment config not present in this repo)
**Compute**: Lambda (Python); uses Redshift Data API, S3, and STS for cross-account access

**What it does**: Monitors customer delivery file sizes to detect anomalies. On each invocation it:
(1) Assumes a cross-account role (`3VDEVDS-S3-Readonly`) to scan the production packager archive
bucket (`s3-atp-3victors-3vprod-use1-pe-packager-archive`) for all customer delivery files on the
target date using parallel prefix listing; (2) Aggregates file counts and sizes in MB per customer,
delivery name, and UTC hour; (3) Uploads the daily summary CSV to
`s3://3v-ds-pe-delivery-monitor/customer_delivery_size_tracker/v1/YYYY/MM/DD/`; (4) Ensures the
corresponding Redshift Spectrum partition exists on
`delivery_monitor.customer_delivery_size_tracker_v1`; (5) Queries 35 days of history from
Redshift Spectrum; and (6) Applies IQR-based anomaly detection (both all-day and day-of-week
windows) against each external customer's delivery size. At 6 AM, 12 PM, and 6 PM UTC, posts a
Slack alert table of anomalous customers (direction, today's value vs. historical average, IQR
range). Internal/test customers (ADB, CH, QA, Test, etc.) are excluded from alerts.

**Input**:
- S3 (cross-account): `s3://s3-atp-3victors-3vprod-use1-pe-packager-archive/{customer}/{delivery_name}/{YYYY}/{MM}/{DD}/`
- Redshift Spectrum: `delivery_monitor.customer_delivery_size_tracker_v1` (35-day history)

**Output**:
- S3: `s3://3v-ds-pe-delivery-monitor/customer_delivery_size_tracker/v1/YYYY/MM/DD/cust_dlv_size_track.csv`
- Redshift Spectrum partition: `delivery_monitor.customer_delivery_size_tracker_v1` (sales_date partition)
- Slack webhook: anomaly alerts posted to the configured channel at 6 AM, 12 PM, 6 PM UTC

---

## Placeholder / Future Components

The following source directories exist but contain only `.gitkeep` (no deployed code):

| Directory | Status |
|-----------|--------|
| `source/collection-anomalies/collection-customer/` | Placeholder |
| `source/collection-anomalies/collection-provider/src/main.py` | Empty entry point |
| `source/provider-monitoring/adc-monitoring-notification/` | Placeholder |
| `source/provider-monitoring/provider-monitoring-qs/` | Placeholder |

`source/provider-monitoring/provider-monitoring-streamlit/` contains a Streamlit dashboard
(`dashboard.py`, `rs_access_v1.py`) for ad-hoc provider monitoring queries against Redshift —
this is a local development/analysis tool, not an AWS deployed component.

---

## Glue Databases & Tables

| Database | Tables |
|----------|--------|
| `glue-atp-3victors-{env}-use1-monitoring_db` | `deduped_provider_request_audit`, `deduped_provider_request_audit_detail`, `deduped_provider_response_audit`, `deduped_retry_audit`, `deduped_global_filter_audit_summary`, `deduped_enrichment_audit`, `deduped_cache_loader_audit`, `deduped_packager_audit`, `deduped_delivery_audit`, `combined_audit`, `provider_combined_audit`, `customer_combined_audit_v2` |

**Redshift Spectrum external table** (not in Glue):
- `delivery_monitor.customer_delivery_size_tracker_v1` → `s3://3v-ds-pe-delivery-monitor/customer_delivery_size_tracker/v1/` (partitioned by `sales_date`)

---

## Configuration

Runtime configuration (S3 paths, Redshift table names, Glue database/table names) is loaded at job
start from S3 properties files. Three config files exist, one per component family:

| Config File | Used By |
|-------------|---------|
| `config-server-{env}/default/deduped-audit-config.properties` | All 9 deduped audit unload jobs + `unload-combined-audit` |
| `config-server-{env}/default/provider-combined-audit-config.properties` | `provider-centric-dataset-unload`, `response-dupes-unload` |
| `config-server-{env}/default/customer-combined-audit-config.properties` | `customer-centric-dataset-unload` |

Reference copies of these properties files live in `docs/properties/` in this repo.

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| AWS Glue ETL Jobs | 11 |
| ECS Fargate Tasks | 1 (`monitoring-verify-dedupe`) |
| Lambda Functions | 1 (`delivery-anomalies`) |
| Step Functions | 1 (`unload-monitoring-step-function`) |
| EventBridge Rules | 4 (1 hourly Step Function trigger + 3 standalone Glue triggers) |
| Glue Databases | 1 (`glue-atp-3victors-{env}-use1-monitoring_db`) |
| Glue Tables | 12 |
| S3 Buckets (written) | 4 (`deduped-datasets`, `customer-monitor`, `provider-monitor`, `3v-ds-pe-delivery-monitor`) |
| Redshift Connections | 2 (`Monitoring Connection`, `Monitoring_Connection_Local`) |
