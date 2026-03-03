# ds-priceeye-data-collection

> A suite of ECS Fargate tasks, Lambda functions, and Glue ETL jobs that collect, transform, and publish operational data — including site metrics, YQYR tax predictions, Sales POC inputs, fare TTL, and collection-optimizer SWIA data — for use across the PriceEye analytics platform.

> **Current branch**: `develop` — this document reflects the `develop` branch. The **master** branch represents what is running in production; differences may exist between this document and production state.

---

## Architecture Overview

```
═══════════════════════ DAILY PIPELINES (02:00 UTC) ═══════════════════════

[EventBridge: daily 02:00 UTC]
        │
        ├──► [cache-metrics-generator]   ──► S3: sitemetrics/cache_metrics/v2/
        │        (Lambda, 512 MB)              (Glue: cache_metrics_v1)
        │
        ├──► [import-metrics-generator]  ──► S3: sitemetrics/import_metrics/v1/
        │        (Lambda, 512 MB)              (Glue: import_metrics_v1)
        │
        ├──► [retry-metrics-generator]   ──► S3: sitemetrics/retry_metrics/v2/
        │        (Lambda, 512 MB)              (Glue: retry_metrics_v1)
        │
        └──► [capacity-metrics-generator] ──► S3: sitemetrics/capacity_metrics/v1/
                 (Lambda, 2048 MB)             (Glue: capacity_raw, capacity_hourly,
                                                capacity_daily, capacity_final,
                                                capacity_stats_by_date)

═══════════════════ SITE METRICS PIPELINE (05:00 UTC) ══════════════════════

[EventBridge: daily 05:00 UTC]
        │
        ▼
[Step Function: site-metrics-stepfunction]
        │
        ├─Step 1─► [site-metrics-input-unload]            ──► S3: sitemetrics/provider_tps_validate/v1/
        │               (Glue, G.1X × 10)                     (Glue: provider_tps_validate_v1)
        │
        └─Step 2─► [site-metrics-provider-tps-summary-interval] ──► S3: sitemetrics/provider_tps_by_intervals/v1/
                        (Glue, G.1X × 10)                               (Glue: provider_tps_by_intervals_v1)

[EventBridge: daily 05:30 UTC]
        │
        └──► [site-metrics-monitor]   ──► Slack alerts if zero-row tables detected
                 (Lambda, 512 MB)

═══════════════════ SALES POC PIPELINE (Saturdays 02:00 UTC) ═══════════════

[EventBridge: weekly Saturday 02:00 UTC]
        │
        ▼
[Step Function: DS-Sales-POC-Jobs]
        │
        ├─Step 1─► [sales-poc-market-generator]  ──► S3: ds-sales-poc/market_data/v1/
        │               (ECS Fargate, 2048 MB)        (Glue: market_data_v1)
        │               reads: Redshift flight_summary
        │
        └─Step 2─► [sales-poc-input-generator]   ──► MySQL: input_request table
                        (ECS Fargate, 2048 MB)        S3: ds-sales-poc/input_requests/v1/
                        reads: S3 market_data,         (Glue: input_requests_v1)
                               MIDT, MySQL

═══════════════ YQYR CACHE INFERENCE (Glue, no master-stack schedule) ══════

[yqyr-cache-unload] (Glue, G.1X × 40)
        │  reads: Redshift/monitoring YQYR fare data
        │
        └──► S3: ds-yqyr-cache/yqyr_cache/v1/
                 (Glue: yqyr_cache_v1)
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
[yqyr-cache-inference]    [yqyr-cache-inference-daily]
  (Glue, customer: AA)       (Glue, G.1X × 30, customer: B6)
           │                         │
           └──────────┬──────────────┘
                      ▼
             S3: ds-yqyr-cache/yqyr_predictions/
             (Glue: yqyr_predictions)

═══════════════ COLLECTION OPTIMIZER (Glue, daily 01:10 UTC) ═══════════════

[Glue Trigger: daily 01:10 UTC]
        │
        ▼
[delta-data-swia-input-unload]   reads: Redshift SWIA observation data (DL ODs)
  (Glue, G.2X × 10)
        │
        └──► S3: ds-collection-optimizer/delta-swia/v1/
                 (Glue: delta_swia_v1)

══════════════ INGEST TTL (ECS, weekly Sunday 21:30 UTC) ════════════════════

[EventBridge: weekly Sunday 21:30 UTC]
        │
        ▼
[ingest-ttl]   reads: Redshift observation/ingest data per carrier
  (ECS Fargate, 4096 MB)
        │
        └──► S3: {ttl.s3.bucket}/ingest_ttl/v1/{carrier}/{YYYY}/{MM}/{DD}/data.parquet

══════════════════ AS DASHBOARD (ECS, manually triggered) ═══════════════════

[Manual trigger with auto_schedule_id argument]
        │
        ▼
[as-dashboard-generator]   reads: Redshift scheduling.auto_schedule_output
  (ECS Fargate, 2048 MB)
        │
        └──► S3: {output.bucket}/{prefix}/{auto_schedule_id}/
                 - comparison_provider_{id}.csv
                 - comparison_customer_{id}.csv
```

---

## Orchestration

### Step Function: DS-Sales-POC-Jobs

- **Trigger**: EventBridge cron — every Saturday at 02:00 UTC
- **Pipeline**: `sales-poc-market-generator` → `sales-poc-input-generator` _(sequential)_
- **Definition**: `source/deploy/definitions/ds-sales-poc-step-function.asl.json`
- **Note**: The Step Function comment documents the overall workflow: "reads from flight_summary, writes to s3-atp-3victors-{env}-use1-ds-sales-poc and to sales_poc.input_request (MySQL table)."

### Step Function: site-metrics-stepfunction

- **Trigger**: EventBridge cron — daily at 05:00 UTC
- **Pipeline**: `site-metrics-input-unload` → `site-metrics-provider-tps-summary-interval` _(sequential; on failure → FailState)_
- **Definition**: `source/deploy/definitions/site-metrics-step-function.asl.json`

### Standalone EventBridge Rules (no Step Function)

| Rule | Schedule | Target |
|------|----------|--------|
| `cache-metrics-generator-task` | Daily 02:00 UTC | `cache-metrics-generator` Lambda |
| `import-metrics-generator-task` | Daily 02:00 UTC | `import-metrics-generator` Lambda |
| `retry-metrics-generator-task` | Daily 02:00 UTC | `retry-metrics-generator` Lambda |
| `capacity-metrics-generator-task` | Daily 02:00 UTC | `capacity-metrics-generator` Lambda |
| `site-metrics-monitor-task` | Daily 05:30 UTC | `site-metrics-monitor` Lambda |
| `ingest-ttl-task` _(standalone)_ | Weekly Sunday 21:30 UTC | `ingest-ttl` ECS task |
| `delta-data-swia-input-unload` _(Glue Trigger)_ | Daily 01:10 UTC | `delta-data-swia-input-unload` Glue Job |

---

## Components

> Ordered by pipeline, then time of execution.

---

### delta-data-swia-input-unload

**Type**: Glue ETL Job (standalone stack — not in master stack)
**Trigger**: Glue Trigger — daily at 01:10 UTC
**Compute**: G.2X × 10 workers, 60-minute timeout, Glue 5.0

**What it does**: Queries Redshift via the "Monitoring Connection" for Delta/SWIA (Search With Itineraries and Availability) observation data, filtering specifically for DL (Delta Air Lines) OD pairs observed over the previous 9 days. Deduplicates to distinct origin/destination pairs and writes the results as parquet to the `delta_swia_v1` Glue table. This output is used by the collection-optimizer to understand which markets have been actively searched.

**Input**:
- Redshift: monitoring/input table (customer_site_code filter: DL or carrier like '%DL%'), `sales_date` range of prior 9 days
- Glue Connection: `Monitoring Connection`

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-collection-optimizer/delta-swia/v1/`
- Glue table: `delta_swia_v1` from `glue-atp-3victors{env}-use1-collection_optimizer_db`

**Table Schema** (`delta_swia_v1`):

| Column | Type |
|--------|------|
| observation_date | date |
| observation_hour | int |
| validatingcarrier | varchar(256) |
| source | varchar(256) |
| origincitycode | varchar(256) |
| destinationcitycode | varchar(256) |
| origin_airportcode | varchar(256) |
| dest_airportcode | varchar(256) |
| departdate | bigint |
| returndate | bigint |
| cabin | varchar(256) |
| advancepurchase | int |
| lengthofstay | int |
| pointofsalecountrycode | varchar(256) |
| outboundroutekey | varchar(256) |
| outboundstops | int |
| outboundcarriers | varchar(256) |
| outbound_brandid | varchar(256) |
| inboundroutekey | varchar(256) |
| inboundstops | int |
| inboundcarriers | varchar(256) |
| inbound_brandid | varchar(256) |
| min_price | double |

_Partition keys: `sales_date`_

---

### cache-metrics-generator

**Type**: Lambda Function (container image, arm64)
**Trigger**: EventBridge cron — daily at 02:00 UTC (`{"daysoffset": 1}`)
**Compute**: 512 MB, 900-second timeout

**What it does**: Queries the monitoring database for cache hit/miss metrics per provider/site over the past 4 weeks up to yesterday. Aggregates cache metrics at the provider × site × hour level and writes a daily CSV snapshot to S3 under the site-metrics bucket. This data powers dashboards showing cache utilization rates (how often responses are served from cache vs. live queries) per provider/site.

**Input**:
- Monitoring DB (Redshift via VPC): table configured in `site-metrics-config.properties` (`cache_metrics_input_table`), date range: `[today-28days, yesterday]`

**Output**:
- S3: `s3://{cache_metrics_unload_bucket}/{cache_metrics_unload_prefix}/{YYYY}/{MM}/{DD}/output_{YYYYMMDD}.csv`
- Glue table: `cache_metrics_v1` from `glue-atp-3victors-{env}-use1-site-metrics-db`

**Table Schema** (`cache_metrics_v1`):

| Column | Type |
|--------|------|
| providercode | varchar(50) |
| sitecode | varchar(50) |
| hour | int |
| cache_count | int |
| total | int |
| cachepct | decimal(5,2) |
| cachesource | varchar(100) |
| last_updated | varchar(256) |

_Partition keys: `sales_date`_

---

### import-metrics-generator

**Type**: Lambda Function (container image, arm64)
**Trigger**: EventBridge cron — daily at 02:00 UTC (`{"daysoffset": 1}`)
**Compute**: 512 MB, 900-second timeout

**What it does**: Queries the monitoring database for import (schedule import / crawl completion) metrics per active customer collection ID over the past 4 weeks. Cross-references active `customercollectionid` values from a MySQL reference table, fetches TPH (transactions per hour) rates from Redshift, and uploads an aggregated daily CSV to S3. This data tracks how many results are being imported per provider/site per hour for each customer collection.

**Input**:
- Redshift (VPC): import metrics input table (configured in `site-metrics-config.properties` → `import_metrics_input_table`)
- MySQL: reference table (`import_metrics_reference_table`) for active collection IDs

**Output**:
- S3: `s3://{import_metrics_s3_bucket}/{import_metrics_s3_prefix}/{YYYY}/{MM}/{DD}/output_{YYYYMMDD}.csv`
- Glue table: `import_metrics_v1` from `glue-atp-3victors-{env}-use1-site-metrics-db`

**Table Schema** (`import_metrics_v1`):

| Column | Type |
|--------|------|
| providercode | varchar(50) |
| sitecode | varchar(50) |
| cust_collectid | varchar(1024) |
| obs_hr | int |
| tph | double |
| last_updated | timestamp |

_Partition keys: `sales_date`_

---

### retry-metrics-generator

**Type**: Lambda Function (container image, arm64)
**Trigger**: EventBridge cron — daily at 02:00 UTC (`{"daysoffset": 1}`)
**Compute**: 512 MB, 900-second timeout

**What it does**: Queries the monitoring database for retry metrics (how often requests are retried) per provider/site over the past 4 weeks. Filters out zero-retry entries and outputs a CSV with retry counts, total requests, and retry percentage by provider, site, and hour. This data reveals providers with reliability issues that require frequent retries.

**Input**:
- Redshift (VPC): retry metrics input table (configured in `site-metrics-config.properties` → `retry_metrics_input_table`)

**Output**:
- S3: `s3://{retry_metrics_s3_bucket}/{retry_metrics_s3_prefix}/{YYYY}/{MM}/{DD}/output_{YYYYMMDD}.csv`
- Glue table: `retry_metrics_v1` from `glue-atp-3victors-{env}-use1-site-metrics-db`

**Table Schema** (`retry_metrics_v1`):

| Column | Type |
|--------|------|
| providercode | varchar(50) |
| sitecode | varchar(50) |
| hour | int |
| retry_count | int |
| total | int |
| retry_pct | decimal(5,2) |
| retry_prov_site | varchar(50) |

_Partition keys: `sales_date`_

---

### capacity-metrics-generator

**Type**: Lambda Function (container image, arm64)
**Trigger**: EventBridge cron — daily at 02:00 UTC (`{"delta_days": 14}`)
**Compute**: 2048 MB, 900-second timeout

**What it does**: Computes throughput capacity (TPH — transactions per hour) for each provider/site combination by reading `provider_tps_by_intervals` data from Redshift over the past 14 days. Calculates raw, hourly, daily, and final capacity statistics using IQR-based outlier filtering and site hierarchy coverage enforcement. Applies provider-specific patches (e.g., QL2Vacation = 2× max, minimum floors for QL2/SS/Atlas). Writes multiple CSV outputs to S3 under the site-metrics bucket, and registers Glue partitions for each. This data drives the collection scheduler's understanding of how fast each provider/site can process requests.

**Input**:
- Redshift (VPC): `capacity_metrics_input_table` from `site-metrics-config.properties` — 14-day rolling window of `provider_tps_by_intervals`
- MySQL (VPC): site hierarchy and API provider rates

**Output**:
- S3: `s3://{C_S3_BUCKET}/{C_S3_PREFIX}/capacity_raw/{YYYY}/{MM}/{DD}/...`
- S3: `s3://{C_S3_BUCKET}/{C_S3_PREFIX}/capacity_hourly/{YYYY}/{MM}/{DD}/...`
- S3: `s3://{C_S3_BUCKET}/{C_S3_PREFIX}/capacity_daily/{YYYY}/{MM}/{DD}/...`
- S3: `s3://{C_S3_BUCKET}/{C_S3_PREFIX}/capacity_final/{YYYY}/{MM}/{DD}/...`
- S3: `s3://{C_S3_BUCKET}/{C_S3_PREFIX}/capacity_stats_by_date/{YYYY}/{MM}/{DD}/...`
- Glue tables: `capacity_raw`, `capacity_hourly`, `capacity_daily`, `capacity_final`, `capacity_stats_by_date` in `glue-atp-3victors-{env}-use1-site-metrics-db`

**Table Schema** (`capacity_final`):

| Column | Type |
|--------|------|
| providercode | varchar(256) |
| sitecode | varchar(256) |
| utc_hour | bigint |
| tph_median | double |
| measure | varchar(256) |
| ct_sum | double |
| avg_first_resp_delay_minute | double |
| last_updated | varchar(256) |

_Partition keys: `sales_date`_

---

### site-metrics-input-unload

**Type**: Glue ETL Job
**Trigger**: Step Function `site-metrics-stepfunction` — Step 1 (after daily EventBridge at 05:00 UTC)
**Compute**: G.1X × 10 workers, 60-minute timeout, Glue 4.0
**Connection**: `Monitoring Connection`

**What it does**: Unloads raw provider-level TPS (transactions per second) validation data from the monitoring database into S3 as parquet. The output contains per-request observations including provider/site, response statuses, retry chains, timestamps, and durations. This is the first step in the site-metrics pipeline and produces the raw dataset that `site-metrics-provider-tps-summary-interval` then aggregates.

**Input**:
- Monitoring DB (Redshift via Glue connection): raw TPS observation records for the given `SALES_DATE` or `DAYSOFFSET`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-sitemetrics/provider_tps_validate/v1/`
- Glue table: `provider_tps_validate_v1` from `glue-atp-3victors-{env}-use1-site-metrics-db`

**Table Schema** (`provider_tps_validate_v1`):

| Column | Type |
|--------|------|
| id | bigint |
| providercode | varchar(256) |
| sitecode | varchar(256) |
| sitecategories | varchar(256) |
| filterreason | varchar(256) |
| issue_sources | varchar(256) |
| issue_reasons | varchar(256) |
| retry_providers | varchar(256) |
| retry_sites | varchar(256) |
| query_ts | timestamp |
| resp_statuses | varchar(256) |
| resp_ts | timestamp |
| obs_ts | timestamp |
| sources | varchar(256) |
| itins | bigint |
| resp_dur | bigint |
| obs_dur | bigint |

_Partition keys: `sales_date`_

---

### site-metrics-provider-tps-summary-interval

**Type**: Glue ETL Job
**Trigger**: Step Function `site-metrics-stepfunction` — Step 2 (after `site-metrics-input-unload`)
**Compute**: G.1X × 10 workers, 60-minute timeout, Glue 4.0
**Connection**: `Monitoring Connection`

**What it does**: Reads the validated TPS data from the monitoring connection and aggregates it into fixed time intervals (e.g., hourly buckets), computing TPS, TPM (transactions per minute), observation durations, and completion windows per provider/site. The result is written to the `provider_tps_by_intervals_v1` Glue table, which is later read by `capacity-metrics-generator` to compute rolling capacity estimates.

**Input**:
- Monitoring DB: provider TPS validate data, cross-joined with `time_intervals` reference

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-sitemetrics/provider_tps_by_intervals/v1/`
- Glue table: `provider_tps_by_intervals_v1` from `glue-atp-3victors-{env}-use1-site-metrics-db`

**Table Schema** (`provider_tps_by_intervals_v1`):

| Column | Type |
|--------|------|
| providercode | varchar(50) |
| sitecode | varchar(50) |
| src | varchar(256) |
| time_interval | timestamp |
| obs_ts_min | timestamp |
| obs_ts_max | timestamp |
| completed_minutes | bigint |
| ct | bigint |
| obs_dur_avg | bigint |
| avg_first_resp_delay | bigint |
| tpm | bigint |
| tps | decimal(8,2) |

_Partition keys: `sales_date`_

---

### site-metrics-monitor

**Type**: Lambda Function (container image, arm64)
**Trigger**: EventBridge cron — daily at 05:30 UTC (`{"daysoffset": 1}`)
**Compute**: 512 MB, 900-second timeout

**What it does**: Monitors all site-metrics Redshift tables (configured via `smm-config.properties` → `smm_table*` keys) for the previous day's data. For each monitored table, queries Redshift for the record count on the given `sales_date`. If any table returns zero rows or has schema errors (missing columns), the monitor sends a Slack alert to the operations webhook. Runs approximately 30 minutes after the site-metrics pipeline completes, acting as a data quality gate.

**Input**:
- Redshift (VPC): all tables listed under `smm_table*` in `smm-config.properties`

**Output**:
- Slack notification (via `slackHelper` Secrets Manager secret, `operations.webhook` key) if zero-row tables or schema errors are detected

---

### sales-poc-market-generator

**Type**: ECS Fargate Task (ARM64)
**Trigger**: Step Function `DS-Sales-POC-Jobs` — Step 1 (weekly Saturday 02:00 UTC)
**Compute**: 2048 MB, 1024 CPU (1 vCPU)

**What it does**: Downloads market data (competitive seat capacity and market share by OD) from Redshift's `flight_summary` table, joining with parent airline mappings, valid carrier lists, and top routes. Computes competitor lists, market shares, missing carrier flags, and validity indicators. Writes the resulting dataset as parquet to the Sales POC S3 bucket, partitioned by date, serving as the market context for `sales-poc-input-generator`.

**Input**:
- Redshift: `flight_summary`, parent airline mapping, valid carriers, top routes (via SQL queries in `source/sales-poc-market-generator/src/sql/`)
- Config: `sales-poc.properties` → `sales.poc.bucket`

**Output**:
- S3: `s3://{SALES_POC_BUCKET}/market_data/v1/{YYYY}/{MM}/{DD}/market_data.parquet`
- Glue table: `market_data_v1` from `glue-atp-3victors{env}-use1-sales_poc_db`

**Table Schema** (`market_data_v1`):

| Column | Type |
|--------|------|
| carrier_code | varchar(32) |
| origin_airport_code | varchar(256) |
| destination_airport_code | varchar(256) |
| origin_city_code | varchar(8) |
| destination_city_code | varchar(8) |
| origin_country_code | varchar(8) |
| destination_country_code | varchar(8) |
| airport_od_seats | bigint |
| city_od_seats | double |
| city_rank | double |
| competitor_list | varchar(256) |
| airport_od_total_seats | bigint |
| market_share | double |
| missing_carriers | varchar(256) |
| missing_shares | varchar(256) |
| total_missing | double |
| is_valid | boolean |
| missing_threshold | boolean |

_Partition keys: `sales_date`_

---

### sales-poc-input-generator

**Type**: ECS Fargate Task (ARM64)
**Trigger**: Step Function `DS-Sales-POC-Jobs` — Step 2 (after `sales-poc-market-generator`)
**Compute**: 2048 MB, 1024 CPU (1 vCPU)

**What it does**: Generates input request files for each paying PriceEye customer by combining market data (from S3), trip type patterns (from MIDT/Redshift), and valid carrier lists (from MySQL). For each customer, it selects the top N city ODs, applies carrier filtering, and creates input request records covering origin, destination, cabin, trip type, length-of-stay, and other shopping parameters. Writes results to both the MySQL `input_request` table (truncating first for full refresh) and an S3 parquet archive partitioned by date and customer. Records a qualification score (perfect/working/mediocre/partial) per customer based on how many of the target city ODs were achieved.

**Input**:
- S3: `s3://{SALES_POC_BUCKET}/market_data/v1/{YYYY}/{MM}/{DD}/market_data.parquet` (from previous step)
- Redshift (MIDT): trip type distribution data
- MySQL: valid carriers, customer list, `input_request` table (read + write)
- Config: `sales-poc.properties` → `sales.poc.num_cities`, `sales.poc.bucket`

**Output**:
- MySQL: `input_request` table (truncated and repopulated for all customers)
- S3: `s3://{SALES_POC_BUCKET}/input_requests/v1/{YYYY}/{MM}/{DD}/...`
- Glue table: `input_requests_v1` from `glue-atp-3victors{env}-use1-sales_poc_db`

**Table Schema** (`input_requests_v1`):

| Column | Type |
|--------|------|
| customer_site_code | varchar(256) |
| customer_site_name | varchar(256) |
| pos | varchar(32) |
| carrier_codes | varchar(8) |
| connection_airport_codes | varchar(8) |
| max_stops | bigint |
| cabin | varchar(8) |
| origin_airport_code | varchar(8) |
| destination_airport_code | varchar(8) |
| depart_date | varchar(8) |
| trip_type | varchar(8) |
| length_of_stay | varchar(256) |
| dow_filter_depart | varchar(256) |
| dow_filter_return | varchar(256) |
| passenger_count | bigint |
| frequency | varchar(256) |
| refundable | bigint |
| reference_val | varchar(256) |
| status | varchar(256) |

_Partition keys: `sales_date`, `customer`_

---

### yqyr-cache-unload

**Type**: Glue ETL Job (standalone stack — not in master stack)
**Trigger**: Not scheduled in this repo — triggered externally or on-demand
**Compute**: G.1X × 40 workers, 240-minute timeout, Glue 5.0

**What it does**: Unloads YQYR (YQ/YR fuel surcharge) fare cache data from the source database/Redshift into S3 as parquet, partitioned by `sales_date`. This acts as the first stage of the YQYR inference pipeline — populating the raw cache that the inference jobs then read. Supports multi-customer and date-range modes via `--SALES_DATE`, `--SALES_DATE_BEGIN`, `--SALES_DATE_END` arguments.

**Input**:
- Redshift: YQYR fare observation data

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-yqyr-cache/yqyr_cache/v1/`
- Glue table: `yqyr_cache_v1` from `glue-atp-3victors{env}-use1-yqyr_cache_db`

---

### yqyr-cache-inference

**Type**: Glue ETL Job (standalone stack — not in master stack)
**Trigger**: Not scheduled in this repo — triggered externally or on-demand
**Compute**: G.1X × 30 workers, 240-minute timeout, Glue 5.0
**Default Customer**: `AA` (American Airlines)

**What it does**: Reads the YQYR cache (from `yqyr-cache-unload`) and common_output itinerary data, then predicts YQ and YR tax amounts using a multi-level hierarchical fallback strategy (route → route+AP bucket → carrier → default). Writes predictions as parquet to the `yqyr_predictions` Glue table. Supports `--CUSTOMER`, `--SALES_DATE`, date-range, and `--RUN_MODE` arguments for flexible scheduling.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-yqyr-cache/yqyr_cache/v1/` (Glue: `yqyr_cache_v1`)
- S3: common_output data (customer-specific)

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-yqyr-cache/yqyr_predictions/`
- Glue table: `yqyr_predictions` from `glue-atp-3victors{env}-use1-yqyr_cache_db`

---

### yqyr-cache-inference-daily

**Type**: Glue ETL Job (in master stack)
**Trigger**: Not scheduled via EventBridge in this repo — invoked externally or on-demand
**Compute**: G.1X × 30 workers, 240-minute timeout, Glue 5.0
**Default Customer**: `B6` (JetBlue)

**What it does**: Identical in structure to `yqyr-cache-inference` but scoped to customer `B6` (JetBlue). Reads the YQYR fare cache and performs multi-level hierarchical fallback prediction for YQ/YR tax amounts, writing parquet predictions partitioned by `sales_date`. The `-daily` suffix indicates this variant is designed for daily scheduled runs with a single sales date.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-yqyr-cache/yqyr_cache/v1/`
- S3: common_output data (customer B6)

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-yqyr-cache/yqyr_predictions/`
- Glue table: `yqyr_predictions` from `glue-atp-3victors{env}-use1-yqyr_cache_db`

**Table Schema** (`yqyr_predictions`):

| Column | Type |
|--------|------|
| carrier | varchar(8) |
| origin | varchar(8) |
| destination | varchar(8) |
| ap | int |
| outbound_travel_stop_over | varchar(512) |
| inbound_travel_stop_over | varchar(512) |
| outbound_flight_no | varchar(512) |
| inbound_flight_no | varchar(512) |
| outbound_flight_duration | varchar(256) |
| inbound_flight_duration | varchar(256) |
| price_inc | decimal(18,2) |
| outbound_total_flight_duration | int |
| inbound_total_flight_duration | int |
| cache_prediction_yq | double |
| cache_prediction_yr | double |
| cache_prediction_total | double |
| yq_cache_level | int |
| yr_cache_level | int |

_Partition keys: `sales_date`_

---

### ingest-ttl

**Type**: ECS Fargate Task (ARM64) (standalone stack — not in master stack)
**Trigger**: EventBridge cron — weekly Sunday at 21:30 UTC
**Compute**: 4096 MB, 2048 CPU (2 vCPU)

**What it does**: Computes Time-To-Live (TTL) for airline fares — i.e., how long a fare price remains stable before changing. For a given carrier and date, reads raw observation data (pos, carrier, cabin, OD, depart date, flight number, min price, observation timestamp) from Redshift. Applies forward-fill to create a complete hourly price series per flight, then detects price changes and computes the 25th-percentile duration of each price segment as the TTL. Writes the result as parquet to S3 partitioned by carrier, year, month, and day, and registers the Glue partition.

**Input**:
- Redshift: observation/ingest data for the carrier (via `dao/redshift_reader.py`)
- Config: `ingest-ttl.properties` → `ttl.s3.bucket`, `ttl.glue.database`, `ttl.glue.table`, `ttl.s3.prefix`

**Output**:
- S3: `s3://{ttl.s3.bucket}/{ttl.s3.prefix}/{carrier}/{YYYY}/{MM}/{DD}/data.parquet`
- Glue: partition registered in `{ttl.glue.database}.{ttl.glue.table}`

---

### as-dashboard-generator

**Type**: ECS Fargate Task (ARM64)
**Trigger**: Manual — invoked with an `auto_schedule_id` argument; no EventBridge schedule
**Compute**: 2048 MB, 1024 CPU (1 vCPU)

**What it does**: Generates dashboard comparison CSVs for the auto-scheduler by querying Redshift's `scheduling.auto_schedule_output` table for a specific `auto_schedule_id`. Produces two views: a provider/site-level hourly planned vs. actual comparison (`comparison_provider_{id}.csv`) and a customer distribution comparison (`comparison_customer_{id}.csv`). Uploads both files to S3 under a prefix keyed by the schedule ID. Used by the scheduling team to analyze auto-scheduler plan quality.

**Input**:
- Redshift: `local.scheduling.auto_schedule_output` (filtered by `generation_id = auto_schedule_id`)
- Config: `as-dashboard-generator.properties` → `output.bucket`, `output.prefix`

**Output**:
- S3: `s3://{output.bucket}/{output.prefix}/{auto_schedule_id}/comparison_provider_{id}.csv`
- S3: `s3://{output.bucket}/{output.prefix}/{auto_schedule_id}/comparison_customer_{id}.csv`

---

## Glue Databases

| Database | Tables |
|----------|--------|
| `glue-atp-3victors{env}-use1-sales_poc_db` | `market_data_v1`, `input_requests_v1`, `missing_carrier_markets_v1` |
| `glue-atp-3victors-{env}-use1-site-metrics-db` | `provider_tps_validate_v1`, `time_intervals`, `provider_tps_by_intervals_v1`, `capacity_raw`, `capacity_hourly`, `capacity_daily`, `capacity_final`, `capacity_stats_by_date`, `retry_metrics_v1`, `cache_metrics_v1`, `import_metrics_v1`, `runtime_metrics`, `preemptive_metrics` |
| `glue-atp-3victors{env}-use1-yqyr_cache_db` | `yqyr_cache_v1`, `yqyr_predictions` |
| `glue-atp-3victors{env}-use1-collection_optimizer_db` | `delta_swia_v1` |

---

## S3 Buckets

| Bucket | Purpose |
|--------|---------|
| `s3-atp-3victors{env}-use1-ds-sales-poc` | Sales POC market data and input requests |
| `s3-atp-3victors{env}-use1-sitemetrics` | Site metrics outputs (TPS, capacity, cache, import, retry) |
| `s3-atp-3victors{env}-use1-ds-yqyr-cache` | YQYR cache and inference predictions |
| `s3-atp-3victors{env}-use1-ds-collection-optimizer` | Delta SWIA collection optimizer data |

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| ECS Fargate Tasks | 4 (as-dashboard-generator, sales-poc-market-generator, sales-poc-input-generator, ingest-ttl) |
| Lambda Functions | 5 (cache-metrics-generator, import-metrics-generator, retry-metrics-generator, capacity-metrics-generator, site-metrics-monitor) |
| Glue ETL Jobs | 5 (site-metrics-input-unload, site-metrics-provider-tps-summary-interval, yqyr-cache-inference-daily, yqyr-cache-inference, yqyr-cache-unload, delta-data-swia-input-unload) |
| Step Functions | 2 (DS-Sales-POC-Jobs, site-metrics-stepfunction) |
| Glue Databases | 4 |
| Glue Tables | 20 |
| EventBridge Rules | 7 (5 Lambda crons + 1 Step Function per pipeline + 1 Glue Trigger) |
| S3 Buckets | 4 |

> **Note on standalone stacks**: Several CloudFormation templates in `source/deploy/commonfiles/` are not referenced by the master stack (`ds-priceeye-data-collection.yaml`) and are deployed as independent stacks. These include: `ingest-ttl`, `yqyr-cache-inference`, `yqyr-cache-unload`, `delta-data-swia-input-unload`, `yqyr-cache-db`, `yqyr-cache-inference-db`, `yqyr-cache-buckets`, `collection-optimizer-db`, and `collection-optimizer-buckets`.
