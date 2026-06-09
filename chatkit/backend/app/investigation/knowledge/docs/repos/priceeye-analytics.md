# priceeye-analytics

> Compute and publish airline pricing analytics — including normalized fares, price outlook, competitive-position anomalies, and QuickSight dashboard replication — for the PriceEye product.

> **Current branch**: `develop` _(this document was generated from the `develop` branch; `master` represents what is running in production and may differ)_

---

## Architecture Overview

```
[EventBridge cron: every hour at :05]
      │
      ▼
[derived-common-output-launcher]  ──launches──►  [derived-common-output]  (Spark, 6GB/2vCPU)
                                                        │ reads: raw pricing data + brand_equivalence (Glue)
                                                        │ writes: S3: derived-common-output/v2/ → Glue: derived_common_output_v2
                                                        │
                                       (S3 Object Created: _SUCCESS in derived-common-output)
                                                        │
                                                        ▼
                                          [derived-common-output-converter-launcher]  (Lambda)
                                                        │ launches:
                                                        ▼
                                          [derived-common-output-csv-converter]  (Spark, 4GB/2vCPU)
                                                        │ writes: S3: derived-common-output-csv/
                                                        │
                                       (S3 Object Created: _SUCCESS in derived-common-output-csv)
                                                        │
                                                        ▼
                                          [dco-csv-batch-upload-launcher]  (Lambda, up to 3 tasks)
                                                        │ launches:
                                                        ▼
                                          [dco-csv-batch-upload]  (ECS, 2GB/1vCPU)
                                                        │ uploads CSV data to downstream database

[EventBridge cron: every hour at :30]
      │
      ▼
[view-refresher]  ─── refreshes materialized views in analytics database (multi-threaded, 15 min timeout)

[EventBridge cron: every 6h at :45]
      │
      ▼
[replication-teardown-checker]  ─── checks for dashboards pending teardown
      │ (if teardown needed)
      ▼
[Step Function: Teardown-Dashboard]
      └──► [replication-dashboard-teardown]

[Step Function: Generate-Dashboard]  (triggered externally, e.g. priceeye-scheduling)
      ├──► [replication-materialized-views]
      ├──► [replication-quicksight-datasets]
      ├──► [replication-quicksight-dashboards]
      └──► [replication-dashboard-deploy]

[price-outlook]            ──► S3: price-outlook/v1/ → Glue: price_outlook
[anomalies-segment-level]  ──► S3: anomaly-datasets/segment-level/v4/ → Glue: segment_level_anomalies_v4
[anomalies-market-level]   ──► S3: anomaly-datasets/market-level/v4/ → Glue: market_level_anomalies_v4
[anomalies-competitive-position] ──► S3: competitive-position/v2/ → Glue: competitive_position_v2
  (↑ these are ECS launch tasks with no schedule or event rule in the master stack;
     trigger mechanism is external — likely driven by priceeye-scheduling or manual invocation)
```

---

## Orchestration

### Step Function: Generate-Dashboard

- **Trigger**: External invocation (e.g., from priceeye-scheduling or via the AWS console); the Step Function is not directly wired to an EventBridge cron in this repo
- **Pipeline**: `replication-materialized-views` → `replication-quicksight-datasets` → `replication-quicksight-dashboards` → `replication-dashboard-deploy` _(in order, fully synchronous)_
- **Input**: `$.Arguments` — a payload containing `dashboardId`, `customer`, and `version`, passed to each ECS task via the `ARGUMENTS` environment variable
- **Definition**: `source/deploy/commonfiles/generate-dashboard-step-function.asl.json`
- **State machine name**: `Generate-Dashboard`

### Step Function: Teardown-Dashboard

- **Trigger**: Started programmatically by `replication-teardown-checker` (which runs on a cron); can also be invoked directly
- **Pipeline**: `replication-dashboard-teardown` _(single task)_
- **Input**: `$.Arguments` — dashboard identity payload
- **Definition**: `source/deploy/commonfiles/teardown-dashboard-step-function.asl.json`
- **State machine name**: `Teardown-Dashboard`

### Standalone EventBridge Rules

| Rule | Schedule | Target |
|------|----------|--------|
| `derived-common-output-launcher-task` | Every hour at :05 (`cron(5 * * * ? *)`) | ECS task: `derived-common-output-launcher` |
| `view-refresher-task` | Every hour at :30 (`cron(30 * * * ? *)`) | ECS task: `view-refresher` |
| `replication-teardown-checker-task` | Every 6 hours at :45 (`cron(45 */6 * * ? *)`) | ECS task: `replication-teardown-checker` |

---

## Components

_(Ordered by when they run in the primary DCO pipeline, then ancillary pipelines.)_

---

### derived-common-output-launcher

**Type**: ECS Fargate Task (scheduled launcher)
**Trigger**: EventBridge cron — every hour at :05 UTC
**Compute**: 2048 MB, 1 vCPU, 4 GB Java heap, 200 GB ephemeral storage

**What it does**: Runs on a fixed hourly schedule and determines whether a new `derived-common-output` computation is needed. It reads from an input S3 bucket to identify pending work, then launches the `derived-common-output` ECS task with the appropriate customer and date/time arguments. Acts as the scheduler/gatekeeper for the core DCO pipeline.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-derived-common-output/` _(scans for pending work)_

**Output**:
- Launches ECS task: `derived-common-output`

---

### derived-common-output

**Type**: ECS Fargate Task (Spark job)
**Trigger**: Launched by `derived-common-output-launcher`
**Compute**: 6144 MB, 2 vCPU, 4 GB Java heap, 200 GB ephemeral storage

**What it does**: The core analytics Spark job. Reads raw airline pricing observations from an upstream enrichment S3 bucket and joins them with the `brand_equivalence` Glue table to normalize carrier brand groupings. Applies business logic to produce the canonical Derived Common Output (DCO) dataset — a clean, partitioned parquet dataset covering fares by origin, destination, carrier, cabin, and brand. Writes results to S3 and registers them in the Glue catalog.

**Input**:
- S3: upstream enrichment/collection bucket _(path configured via `DerivedCommonOutputSparkJob.properties`)_
- Glue table: `brand_equivalence` from `glue-atp-3victors-{env}-use1-brands_enrichment_db`

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-derived-common-output/v2/`
- Glue table: `derived_common_output_v2` (partitioned by `sales_date`, `customer`)

**Table Schema** (`derived_common_output` / `derived_common_output_v2`):

| Column | Type |
|--------|------|
| customer_observation_date | varchar(16) |
| observation_datetime | varchar(32) |
| origin | varchar(8) |
| destination | varchar(8) |
| outbound_gcm | int |
| pos | varchar(8) |
| source | varchar(32) |
| carrier | varchar(16) |
| cabin | varchar(16) |
| brand_group | varchar(128) |
| price_exc | double |
| tax | double |
| price_inc | double |
| provider_currency_exchange_rate | double |

_Partition keys: `sales_date` (int), `customer` (varchar(32))_

---

### derived-common-output-converter-launcher

**Type**: Lambda Function
**Trigger**: EventBridge S3 event — Object Created with key suffix `_SUCCESS` in `s3-atp-3victors{env}-use1-derived-common-output`
**Compute**: 2048 MB, 270 s timeout, reserved concurrency: 5

**What it does**: Responds to each completed DCO Spark job (signaled by a `_SUCCESS` marker file). Extracts the customer and date from the S3 key path, then launches the `derived-common-output-csv-converter` ECS task to convert that job's parquet output into CSV format. The reserved concurrency of 5 limits the number of concurrent conversions.

**Input**:
- S3 event: `_SUCCESS` file created in `s3://s3-atp-3victors{env}-use1-derived-common-output/`

**Output**:
- Launches ECS task: `derived-common-output-csv-converter`

---

### derived-common-output-csv-converter

**Type**: ECS Fargate Task (Spark job)
**Trigger**: Launched by `derived-common-output-converter-launcher`
**Compute**: 4096 MB, 2 vCPU, 4 GB Java heap

**What it does**: Reads the parquet output produced by `derived-common-output` and converts it to CSV format. Writes the CSV files into the `derived-common-output-csv` S3 bucket for downstream consumers. Emits a `_SUCCESS` marker file when complete to trigger the next stage.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-derived-common-output/` _(parquet)_

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-derived-common-output-csv/` _(CSV)_

---

### dco-csv-batch-upload-launcher

**Type**: Lambda Function
**Trigger**: EventBridge S3 event — Object Created with key suffix `_SUCCESS` in `s3-atp-3victors{env}-use1-derived-common-output-csv`
**Compute**: 512 MB, 600 s timeout

**What it does**: Responds to completed CSV conversion jobs (detected via `_SUCCESS` markers). Throttles ECS task launches to a configurable maximum (default: 3 concurrent tasks) by inspecting currently running `dco-csv-batch-upload` tasks before launching new ones. Passes the S3 path and table name (`derived_common_output`) as arguments to the ECS task.

**Input**:
- S3 event: `_SUCCESS` file created in `s3://s3-atp-3victors{env}-use1-derived-common-output-csv/`

**Output**:
- Launches ECS task: `dco-csv-batch-upload` (up to 3 concurrent)

---

### dco-csv-batch-upload

**Type**: ECS Fargate Task (queue-driven)
**Trigger**: Launched by `dco-csv-batch-upload-launcher`
**Compute**: 2048 MB, 1 vCPU, 1 GB Java heap

**What it does**: Reads CSV files from the `derived-common-output-csv` S3 bucket and batch-uploads them to a downstream database table (named `derived_common_output_{customer}`). Uses SQS for work-item coordination and processes records in configurable batch sizes. This is the final step in the primary DCO pipeline, writing normalized pricing data into the analytics database.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-derived-common-output-csv/`
- SQS: work queue for batch coordination

**Output**:
- Database table: `derived_common_output_{customer}` _(Timescale or equivalent)_

---

### view-refresher

**Type**: ECS Fargate Task (scheduled)
**Trigger**: EventBridge cron — every hour at :30 UTC
**Compute**: 2048 MB, 1 vCPU, 4 GB Java heap, 200 GB ephemeral storage

**What it does**: Iterates over a configurable list of database schemas and refreshes all materialized views within each schema, skipping any views matching configured skip-keywords. Uses a 5-thread executor to parallelize refreshes across views, with a 15-minute overall timeout. Keeps analytics database views current with the latest underlying data.

**Input**:
- Database schemas (configured via `ViewRefresh.properties`)

**Output**:
- Refreshed materialized views in analytics database

---

### price-outlook

**Type**: ECS Fargate Task (Spark job)
**Trigger**: No schedule or EventBridge rule defined in the master stack; trigger mechanism is external (likely via `price-outlook-launcher` deployed separately, or via priceeye-scheduling)
**Compute**: 6144 MB, 2 vCPU, 4 GB Java heap, 200 GB ephemeral storage

**What it does**: Runs a Spark job to compute forward-looking price forecasts ("price outlook") for airline itineraries. Reads pricing data from an input S3 bucket, applies forecasting logic, and writes results as parquet partitioned by `sales_date` and `customer` to the price-outlook S3 bucket, registered in the Glue catalog.

**Input**:
- S3: input bucket _(configured via `PriceOutlookSparkJob.properties`)_

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-price-outlook/v1/`
- Glue table: `price_outlook` in `glue-atp-3victors{env}-use1-analytics_db`

**Table Schema** (`price_outlook`):

| Column | Type |
|--------|------|
| customer_observation_date | varchar(16) |
| observation_datetime | varchar(32) |
| origin | varchar(8) |
| origin_city | varchar(8) |
| origin_metro | varchar(8) |
| origin_country | varchar(8) |
| destination | varchar(8) |
| destination_city | varchar(8) |
| destination_metro | varchar(8) |
| destination_country | varchar(8) |
| outbound_gcm | int |
| pos | varchar(8) |
| source | varchar(32) |
| carrier | varchar(16) |
| cabin | varchar(16) |
| outbound_departure_date | varchar(16) |
| inbound_departure_date | varchar(16) |
| advance_purchase | int |
| length_of_stay | int |
| stops | int |
| trip_type | varchar(8) |
| refundable | boolean |
| currency | varchar(16) |
| price_exc | double |
| tax | double |
| price_inc | double |
| preferred_currency_rate | double |

_Partition keys: `sales_date` (int), `customer` (varchar(32))_

---

### anomalies-segment-level

**Type**: ECS Fargate Task (Spark job)
**Trigger**: No schedule or EventBridge rule in master stack; trigger mechanism is external
**Compute**: 6144 MB, 2 vCPU, 4 GB Java heap

**What it does**: Runs a Spark anomaly-detection job at the route-segment level. Reads derived pricing data, applies an IQR-based competitive-position scoring algorithm across segment groupings (route + carrier + cabin), and flags anomalies where the carrier's pricing deviates beyond a configurable threshold (default: 7%). Writes scored results as parquet to the anomaly-datasets bucket.

**Input**:
- S3: input pricing bucket _(via `SegmentLevelSparkJob.properties`)_

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-anomaly-datasets/segment-level/v4/`
- Glue table: `segment_level_anomalies_v4` in `glue-atp-3victors{env}-use1-analytics_db`

**Table Schema** (`segment_level_anomalies`):

| Column | Type |
|--------|------|
| segment_name | varchar(128) |
| competitive_position | varchar(32) |
| region_name | varchar(64) |
| carrier_group | varchar(64) |
| cabin_group | varchar(32) |
| cp_score | double |
| cp_weight | double |
| itinerary_percentage | double |

_Partition keys: `sales_date` (int), `customer` (varchar(32))_

---

### anomalies-market-level

**Type**: ECS Fargate Task (Spark job)
**Trigger**: No schedule or EventBridge rule in master stack; trigger mechanism is external
**Compute**: 6144 MB, 2 vCPU, 4 GB Java heap

**What it does**: Runs a Spark anomaly-detection job aggregated at the metro-market level (city-pair + carrier + cabin). Applies the same IQR-based competitive-position scoring as the segment-level job but rolled up to market granularity. Writes results to the anomaly-datasets bucket for use in market-level analytics views.

**Input**:
- S3: input pricing bucket _(via `MarketLevelSparkJob.properties`)_

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-anomaly-datasets/market-level/v4/`
- Glue table: `market_level_anomalies_v4` in `glue-atp-3victors{env}-use1-analytics_db`

**Table Schema** (`market_level_anomalies`):

| Column | Type |
|--------|------|
| segment_name | varchar(128) |
| competitive_position | varchar(32) |
| metro_market | varchar(32) |
| carrier_group | varchar(64) |
| cabin_group | varchar(32) |
| cp_score | double |
| cp_weight | double |
| itinerary_percentage | double |

_Partition keys: `sales_date` (int), `customer` (varchar(32))_

---

### anomalies-competitive-position

**Type**: ECS Fargate Task (Spark job)
**Trigger**: No schedule or EventBridge rule in master stack; trigger mechanism is external
**Compute**: 6144 MB, 2 vCPU, 4 GB Java heap

**What it does**: Runs a Spark job to compute competitive-position scores — a summary metric indicating where a customer's fares sit relative to the market. Reads the derived pricing data, applies the competitive-positioning algorithm, and writes scored results as parquet to the competitive-position S3 bucket, registered in the Glue catalog.

**Input**:
- S3: input pricing bucket _(via `CompetitivePositionSparkJob.properties`)_

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-competitive-position/v2/`
- Glue table: `competitive_position_v2` in `glue-atp-3victors{env}-use1-analytics_db`

---

### replication-teardown-checker

**Type**: ECS Fargate Task (scheduled)
**Trigger**: EventBridge cron — every 6 hours at :45 UTC (`cron(45 */6 * * ? *)`)
**Compute**: 2048 MB, 1 vCPU, 4 GB Java heap, 200 GB ephemeral storage

**What it does**: Periodically checks the database for customer dashboards that are pending teardown (e.g., expired trial or churned customers). For each dashboard identified, it triggers the `Teardown-Dashboard` Step Function with the appropriate arguments. This component holds IAM permission to call `states:StartExecution` on the `Teardown-Dashboard` state machine.

**Input**:
- Database _(schema/table via `teardown-checker.properties`)_

**Output**:
- Invokes Step Function: `Teardown-Dashboard` _(for each dashboard pending teardown)_

---

### replication-materialized-views

**Type**: ECS Fargate Task (Step Function step)
**Trigger**: First step in `Generate-Dashboard` Step Function
**Compute**: 2048 MB, 1 vCPU, 4 GB Java heap

**What it does**: The first step in dashboard provisioning. Reads `DatasetTemplate` records and their associated SQL template files from the `replication-templates` S3 bucket, then creates per-customer materialized views in the analytics database using those templates. Accepts `dashboardId`, `customer`, and `version` via the `ARGUMENTS` environment variable.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-replication-templates/` _(SQL templates)_
- Database: `DatasetTemplate` table _(schema via `MaterializedViewReplicator.properties`)_
- ARGUMENTS: `dashboardId`, `customer`, `version`

**Output**:
- Database: per-customer materialized views created in the target schema

---

### replication-quicksight-datasets

**Type**: ECS Fargate Task (Step Function step)
**Trigger**: Second step in `Generate-Dashboard` Step Function (after `replication-materialized-views`)
**Compute**: 2048 MB, 1 vCPU, 4 GB Java heap

**What it does**: Replicates QuickSight dataset definitions from a source (template) dashboard to the target customer dashboard. Reads dataset configuration from the `replication-templates` bucket and the QuickSight API, creates or updates datasets pointing to the customer's materialized views, and assigns them to the appropriate QuickSight group. Uses the configured `datasource.arn` and `group.arn` from properties.

**Input**:
- QuickSight API _(source dashboard datasets)_
- S3: `s3://s3-atp-3victors{env}-use1-replication-templates/`
- ARGUMENTS: `dashboardId`, `customer`, `version`

**Output**:
- QuickSight datasets created/updated for the customer

---

### replication-quicksight-dashboards

**Type**: ECS Fargate Task (Step Function step)
**Trigger**: Third step in `Generate-Dashboard` Step Function (after `replication-quicksight-datasets`)
**Compute**: 2048 MB, 1 vCPU, 4 GB Java heap

**What it does**: Replicates QuickSight dashboard definitions from the template dashboard to the target customer. Clones the source dashboard layout and wires the replicated datasets from the previous step into the new dashboard. Handles permissions and group assignments per properties configuration.

**Input**:
- QuickSight API _(source dashboard definition)_
- ARGUMENTS: `dashboardId`, `customer`, `version`

**Output**:
- QuickSight dashboard created/updated for the customer

---

### replication-dashboard-deploy

**Type**: ECS Fargate Task (Step Function step)
**Trigger**: Fourth (final) step in `Generate-Dashboard` Step Function
**Compute**: 2048 MB, 1 vCPU, 4 GB Java heap

**What it does**: Finalizes dashboard deployment by publishing and making the replicated QuickSight dashboard available to the customer. Applies final configuration — such as embedding settings, sharing rules, and version publishing — and records the deployment status in the database.

**Input**:
- QuickSight API
- ARGUMENTS: `dashboardId`, `customer`, `version`

**Output**:
- QuickSight dashboard published and accessible to customer
- Database: deployment status updated

---

### replication-dashboard-teardown

**Type**: ECS Fargate Task (Step Function step)
**Trigger**: Single step in `Teardown-Dashboard` Step Function
**Compute**: 2048 MB, 1 vCPU, 4 GB Java heap

**What it does**: Performs the full teardown of a customer dashboard. Deletes the QuickSight dashboard and associated datasets, drops the customer's materialized views from the analytics database, and records the teardown completion. Receives `dashboardId`, `customer`, and `version` via `ARGUMENTS`.

**Input**:
- QuickSight API
- Database: customer materialized views
- ARGUMENTS: `dashboardId`, `customer`, `version`

**Output**:
- QuickSight dashboard and datasets deleted
- Database: customer materialized views dropped; teardown status recorded

---

## Components in `commonfiles` Not Deployed via Master Stack

The following CloudFormation templates exist in `source/deploy/commonfiles/` but are **not referenced in `priceeye-analytics.yaml`**. They may be deployed via separate stacks or represent standalone / legacy deployments.

### price-outlook-launcher

**Type**: ECS Fargate Task (scheduled launcher, deployed separately)
**Trigger**: EventBridge cron schedule _(default `cron(20 * * * ? *)`, every hour at :20)_
**Compute**: 2048 MB, 1 vCPU, 200 GB ephemeral storage

**What it does**: Checks for new data in the price-outlook input bucket and, when found, launches the `price-outlook` ECS task with the extracted date/time and customer arguments. Mirrors the role that `derived-common-output-launcher` plays for the DCO pipeline.

**Input**:
- S3: price-outlook input bucket _(via `PriceOutlookLauncher.properties`)_

**Output**:
- Launches ECS task: `price-outlook`

---

### price-evolution-upload

**Type**: Lambda Function (deployed separately)
**Trigger**: EventBridge S3 event — `.parquet` file created in `s3-atp-3victors{env}-use1-derived-common-output-csv`
**Compute**: 2048 MB, 270 s timeout

**What it does**: Listens for new parquet files in the CSV-conversion output bucket and uploads price evolution time-series data to a configured time-series database. Supports multiple writer backends: Timescale, Amazon Timestream, InfluxDB, and TSAnalytics. Uses a `JobLockDAO` to prevent duplicate uploads for the same file. Table name is `derived_common_output_{customer}` (configurable).

**Input**:
- S3: `.parquet` files in `s3://s3-atp-3victors{env}-use1-derived-common-output-csv/`

**Output**:
- Time-series database table: `derived_common_output_{customer}` _(Timescale / Timestream / InfluxDB)_

---

## Glue Databases

| Database | Tables |
|----------|--------|
| `glue-atp-3victors{env}-use1-analytics_db` | `derived_common_output`, `derived_common_output_v2`, `price_outlook`, `segment_level_anomalies`, `segment_level_anomalies_v2`, `segment_level_anomalies_v3`, `segment_level_anomalies_v4`, `market_level_anomalies`, `market_level_anomalies_v2`, `market_level_anomalies_v3`, `market_level_anomalies_v4`, `competitive_position`, `competitive_position_v2`, `market_analysis_v2`, `segment_analysis_v2`, `daily_itins_prices_v2`, `oag_score_v2`, `pax_midt`, `revenue_score_v1` |
| `glue-atp-3victors-{env}-use1-brands_enrichment_db` | `brand_equivalence` |

---

## S3 Buckets

| Bucket | Purpose |
|--------|---------|
| `s3-atp-3victors{env}-use1-derived-common-output` | Parquet output of the DCO Spark job; triggers converter launcher on `_SUCCESS` |
| `s3-atp-3victors{env}-use1-derived-common-output-csv` | CSV conversion of DCO; triggers batch-upload launcher and price-evolution-upload on `_SUCCESS` |
| `s3-atp-3victors{env}-use1-anomaly-datasets` | Parquet output of all anomaly detection Spark jobs (segment, market, OAG, PAX, revenue, etc.) |
| `s3-atp-3victors{env}-use1-competitive-position` | Parquet output of competitive-position Spark job |
| `s3-atp-3victors{env}-use1-price-outlook` | Parquet output of price-outlook Spark job |
| `s3-atp-3victors{env}-use1-replication-templates` | SQL templates and dataset config used during dashboard replication |

All buckets have **EventBridge notifications enabled**.

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| ECS Fargate Task Definitions | 14 |
| Lambda Functions | 3 (in master stack: 2; separately deployed: 1) |
| Step Functions | 2 |
| EventBridge Cron Rules | 3 |
| EventBridge S3 Event Rules | 2 (in master stack) |
| Glue Databases | 2 |
| Glue Tables | 19+ (including versioned variants) |
| S3 Buckets | 6 (owned by this repo) |
| CloudWatch Alarm (Lambda timeouts) | 2 |
