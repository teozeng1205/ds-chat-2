# priceeye-scheduling

> A Java-based scheduling platform that generates, converts, and activates flight-price collection schedules for PriceEye, handles preemptive and retry request dispatch, and produces Glue-backed Parquet metrics for operational visibility.

> **Current branch**: `develop` (note: documentation reflects the state of the `develop` branch at time of writing — validate against `master` before treating this as the production baseline)

---

## Architecture Overview

```
                          ┌─────────────────────────────────────────────────────────────────┐
                          │                     SCHEDULE GENERATION PIPELINE                │
                          │                                                                 │
  PEAutoScheduleRequest   │  ┌─────────────────┐   DB + S3    ┌──────────────────────────┐ │
  FIFO Queue ─────────────┼─►│  auto-scheduler  │─────────────►│ auto-schedule-converter  │ │
                          │  │ (PEAutoScheduler)│  persists    │(AutoScheduleConverter)   │ │
                          │  └─────────────────┘  to DB       └──────────┬───────────────┘ │
                          │                                               │ Parquet → S3    │
                          │                                               ▼                 │
                          │                                   s3://..-as-scheduled-         │
                          │                                   comparison/auto-schedule-     │
                          │                                   output/ (Glue: auto_schedule_ │
                          │                                   output)                       │
                          │                                                                 │
                          │  ┌──────────────────┐  DB swap                                 │
  cron(0,25 * * * ? *)────┼─►│ schedule-cutover  │────────────► Activates "scheduled"      │
                          │  │(PEScheduleCutover)│              generation in DB             │
                          │  └──────────────────┘                                          │
                          └─────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────────────────────────────────────────────────┐
                          │                     RUNTIME REQUEST DISPATCH                    │
                          │                                                                 │
  cron(0,25 * * * ? *)────┼─► ┌──────────────────┐  Provider SQS Queues                   │
                          │   │  scheduler         │──────────────────────────────────────► │
                          │   │(PESchedulerApp)    │  PERetry.fifo                          │
                          │   └──────────────────┘                                         │
                          │                                                                 │
  cron(0,25 * * * ? *)────┼─► ┌──────────────────┐  Provider SQS / PEBatchRequestRetry     │
                          │   │ preemptive-polling │──────────────────────────────────────► │
                          │   │(PEPreemptivePoll.) │  .fifo                                 │
                          │   └──────────────────┘                                         │
                          └─────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────────────────────────────────────────────────┐
                          │                       RETRY INFRASTRUCTURE                      │
                          │                                                                 │
  PERetry.fifo ───────────┼─► ┌──────────────────┐  Provider SQS Queues                   │
                          │   │  retry-lambda     │──────────────────────────────────────► │
                          │   │(PERetryLambda)    │                                         │
                          │   └──────────────────┘                                         │
                          │                                                                 │
  PEBatchRequestRetry     │   ┌──────────────────┐                                         │
  .fifo ──────────────────┼─► │ batch-request-    │── Batch JSON → s3://..-pe-as-           │
                          │   │  retry (ECS)      │   persistence/<provider>/requests/      │
                          │   └──────────────────┘                                         │
                          └─────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────────────────────────────────────────────────┐
                          │                         METRICS PIPELINE                        │
                          │                                                                 │
  S3 Object Created       │   ┌──────────────────┐  Redshift → DB                          │
  (*.csv on sitemetrics)──┼─► │  metrics-loader  │                                         │
                          │   │  (Lambda)         │                                         │
                          │   └──────────────────┘                                         │
                          │                                                                 │
  cron(10 * * * ? *) ─────┼─► ┌──────────────────┐  Parquet → s3://..-as-scheduled-       │
                          │   │runtime-metrics-   │  comparison/runtime_metrics/            │
                          │   │data-loader (ECS)  │  (Glue: scheduling_db)                 │
                          │   └──────────────────┘                                         │
                          │                                                                 │
  cron(~xx:50) ───────────┼─► ┌──────────────────┐  Parquet → s3://..-as-scheduled-       │
                          │   │preemptive-metrics-│  comparison/preemptive_metrics/         │
                          │   │data-loader (ECS)  │  (Glue: scheduling_db)                 │
                          │   └──────────────────┘                                         │
                          │                                                                 │
  cron(0 5 * * ? *) ──────┼─► ┌──────────────────┐  Parquet → s3://..-as-scheduled-       │
                          │   │site-metric-valid- │  comparison/site-metric-validation/     │
                          │   │ation-data-loader  │  (Glue: site_metric_validation)         │
                          │   └──────────────────┘                                         │
                          │                                                                 │
  site-metrics-stepfunc.  │   ┌──────────────────┐                                         │
  SUCCEEDED event─────────┼─► │site-metrics-      │── SES Email → Recipients               │
                          │   │reporting (ECS)    │                                         │
                          │   └──────────────────┘                                         │
                          │                                                                 │
  (ad hoc / triggered) ───┼─► ┌──────────────────┐  Parquet → s3://..-as-scheduled-       │
                          │   │scheduling-        │  comparison/scheduling-comparison/      │
                          │   │comparison (ECS)   │  (Glue: schedule_comparison)            │
                          │   └──────────────────┘                                         │
                          └─────────────────────────────────────────────────────────────────┘
```

---

## Orchestration

There is no single master Step Function orchestrating all components. Instead, each component is triggered independently:

- **CRON-triggered ECS Fargate tasks** (via EventBridge scheduled rules) drive the runtime scheduler, preemptive polling, and schedule cutover.
- **SQS-driven ECS tasks** drive the auto-scheduler (polling `PEAutoScheduleRequest.fifo`) and batch-request-retry (polling `PEBatchRequestRetry.fifo`).
- **SQS Lambda trigger** drives the retry-lambda (consuming `PERetry.fifo`).
- **S3 EventBridge notification** (new `.csv` file in the site-metrics bucket) drives the metrics-loader Lambda.
- **Step Function execution event** (`site-metrics-stepfunction` SUCCEEDED) drives the site-metrics-reporting task.

The Step Function named `site-metrics-stepfunction` is defined and managed outside this repository; `priceeye-scheduling` only subscribes to its completion event.

---

## S3 Buckets

| Logical Name | Bucket Pattern | Purpose |
|---|---|---|
| AutoSchedulePersistence | `s3-atp-3victors-{env}-use1-pe-as-persistence` | Intermediate auto-schedule persistence for batch provider requests |
| SiteMetrics | `s3-atp-3victors-{env}-use1-sitemetrics` | Receives site-metrics CSV files that trigger the metrics-loader Lambda |
| ASConvertedPersistence | `s3-atp-3victors-{env}-use1-as-converted-persistence` | Auto-schedule converter output (unused in current Glue mapping) |
| ASScheduledComparison | `s3-atp-3victors-{env}-use1-as-scheduled-comparison` | Primary analytics bucket: contains all Glue-backed Parquet tables |

---

## SQS Queues

All queues are FIFO with content-based deduplication and a 15-minute visibility timeout.

| Queue | DLQ | Purpose |
|---|---|---|
| `PEAutoScheduleRequest.fifo` | `FAILED-PEAutoScheduleRequest.fifo` | Carries `PEAutoSchedulingRequest` messages consumed by `auto-scheduler` |
| `PERetry.fifo` | `FAILED-PERetry.fifo` | Carries `PERetryMessage` objects consumed by `retry-lambda` |
| `PEBatchRequestRetry.fifo` | `FAILED-PEBatchRequestRetry.fifo` | Carries `PERequestBundle` objects for batch providers; consumed by `batch-request-retry` |

Per-provider queues (named in the PriceEye DB provider config) are also written to by the runtime scheduler and preemptive-polling tasks but are not managed in this repository.

---

## Components

_(Ordered by logical pipeline role — schedule generation first, then runtime dispatch, then retry infrastructure, then metrics.)_

---

### auto-scheduler

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: SQS — polls `PEAutoScheduleRequest.fifo` for a `PEAutoSchedulingRequest` message; falls back to command-line args for testing
**Compute**: Default 1024 MB / 1024 CPU (1 vCPU); configured at deploy time
**Main Class**: `com.threevictors.aws.priceeye.scheduling.autoscheduler.PEAutoScheduler`
**Source**: `source/auto-scheduling/auto-scheduler/`

**What it does**: Generates a multi-day flight-price collection plan for every active provider/site-code pair by reading customer collections, input requests, site hierarchy, and site capacity metrics from the PriceEye scheduling database and Redshift. The scheduler runs `PEAutoSchedulerTask` in a thread pool (configurable, default 8 threads) per provider/site, fits requests into time-boxed capacity slots, resolves scheduling hierarchy substitutions, validates capacity constraints, then persists the resulting generation record to the scheduling database. If a generation is already active it may be used as the base for incremental re-scheduling. On fatal configuration errors it posts a Slack alert.

**Input**:
- SQS: `PEAutoScheduleRequest.fifo` (one `PEAutoSchedulingRequest` per invocation)
- Database (MySQL/Aurora — PriceEye scheduling DB): customer collections, input requests, site metrics, site hierarchies, provider transaction rates, provider schedule adjustments
- Database (PriceEye core DB): provider map, site map, site-carrier map, customer map

**Output**:
- Database: persists `AutoScheduleGeneration` records with hourly collection plans
- Slack webhook notification on errors

---

### auto-schedule-converter

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: Typically invoked after a successful `auto-scheduler` run (ad hoc or via wrapper process)
**Compute**: Default 1024 MB / 1024 CPU; configured at deploy time
**Main Class**: `com.threevictors.aws.priceeye.scheduling.autoschedule.converter.generator.AutoScheduleConverter`
**Source**: `source/auto-scheduling/auto-schedule-converter/`

**What it does**: Reads a completed auto-schedule generation from the scheduling database and flattens its hierarchical collection plan into individual `FlattenedAutoScheduler` records (one per scheduled request slot). It writes these records as Parquet files to S3 using 4 writer threads and 16 flattening threads. The output S3 location and version string are configurable via `AutoScheduleConverterJob.properties`. Each generation produces multiple sequentially numbered parquet files (e.g. `auto-schedule-<generationId>-<seq>.parquet`).

**Input**:
- Database (scheduling DB): auto-schedule generation, collection map, input requests, customer map, site hierarchy
- `generationId` passed as command-line argument

**Output**:
- S3: `s3://{output.bucket}/auto-schedule-output/auto-schedule-{generationId}-{seq}.parquet`
- Glue table: `auto_schedule_output` in `glue-atp-3victors-{env}-use1-scheduling_db`

---

### schedule-cutover

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: EventBridge CRON — `cron(0,25 * * * ? *)` (every 25 minutes)
**Compute**: Default 1024 MB / 1024 CPU; configured at `source/deploy/commonfiles/scheduledtaskv2.yaml`
**Main Class**: `com.threevictors.aws.priceeye.scheduling.schedulecutover.PEScheduleCutover`
**Source**: `source/auto-scheduling/schedule-cutover/`

**What it does**: Atomically swaps the active auto-schedule generation. It reads the scheduling database to find the generation in `scheduled` status (next generation) and the generation in `active` status (current generation), then calls `PriceEyeSchedulingWriter.swapGenerationStatuses()` to promote the scheduled generation to active and demote the previously active one. If no scheduled generation exists the task exits without making changes.

**Input**:
- Database (scheduling DB): `GenerationDetails` records filtered by `status = scheduled` and `status = active`

**Output**:
- Database: updated generation status (scheduled → active, active → historical)

---

### scheduler

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: EventBridge CRON — `cron(0,25 * * * ? *)` (every 25 minutes)
**Compute**: Configured at `source/deploy/commonfiles/scheduledtaskv2.yaml`; default 1024 MB / 1024 CPU
**Main Class**: `com.threevictors.aws.priceeye.scheduling.scheduler.PESchedulerApplication`
**Source**: `source/scheduler/`

**What it does**: The core runtime request dispatcher. It reads the active hourly collection plan from the database (either the classic runtime schedule or the active auto-scheduler generation depending on configuration), expands each collection's input requests into `PEExpandedInputRequest` objects for the upcoming scheduling window, applies flight-check filters (direct flight cache, cabin validity), checks Redis for previously fulfilled requests (cache deduplication), then publishes valid request bundles to per-provider SQS queues. Failed or retry-eligible requests are published to `PERetry.fifo`. It also interacts with the audit data stream (Kinesis Firehose) to write `PEProviderRequestAudit` records.

**Input**:
- Database (PriceEye core DB): providers, sites, customers, collections, cache keys, site mappings, cabin config
- Database (scheduling DB): active auto-schedule generation, hourly collection plans, site hierarchies
- Redis: flight-check and cache-check data
- Configuration: `PEAutoScheduler.properties`

**Output**:
- SQS: per-provider queues (named in provider config, e.g. `Travelport-US`)
- SQS: `PERetry.fifo` (failed/retry requests)
- Kinesis Firehose (via audit stream): `PEProviderRequestAudit` and `PEProviderRequestAuditDetail` records

---

### preemptive-polling

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: EventBridge CRON — `cron(0,25 * * * ? *)` (every 25 minutes); processes a 60-minute window, looks 60–90 minutes ahead
**Compute**: Configured at `source/deploy/commonfiles/task.yaml`; default 1024 MB / 1024 CPU
**Main Class**: `com.threevictors.aws.priceeye.scheduling.preemptivepolling.PEPreemptivePollingApplication`
**Source**: `source/preemptive-polling/`

**What it does**: Identifies provider requests that are approaching their drop-dead time (within 60–90 minutes) and have not yet received a response. For each such audit record, it looks up a substitute provider/site-code using either the legacy retry-substitution table (non-auto-scheduler mode) or the site hierarchy (auto-scheduler mode), validates the substitution against cabin and cache checks, creates new `PEProviderRequestAudit` records, publishes them to the audit stream, and dispatches substitute request bundles to provider SQS queues. Batch provider requests are accumulated and flushed to `PEBatchRequestRetry.fifo` instead.

**Input**:
- Database (PriceEye report DB): `PEProviderRequestAudit` records with drop-dead times in the next 60–90 minutes
- Database (PriceEye core DB): provider map, site-carrier map, retry-substitution map, collection map
- Database (scheduling DB): site hierarchies (when auto-scheduler mode enabled)
- Redis: cache check (via `CacheCheckUtil`)
- Config property: `batch.request.bucket` (S3 bucket for batch provider fallback)

**Output**:
- SQS: per-provider queues (for standard providers)
- SQS: `PEBatchRequestRetry.fifo` (for batch providers)
- Kinesis Firehose (audit stream): `PEProviderRequestAudit`, `PEProviderRequestAuditDetail`, `PERetryAudit` records

---

### batch-request-retry

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: SQS — long-polls `PEBatchRequestRetry.fifo`; this is a queue-driven consumer, not scheduled
**Compute**: Configured at `source/deploy/commonfiles/batch-request-retry.yaml`; default 1024 MB / 1024 CPU
**Main Class**: `com.threevictors.aws.priceeye.scheduling.batchrequestretry.PEBatchRequestRetryApplication`
**Source**: `source/batch-request-retry/`

**What it does**: Buffers incoming `PERequestBundle` messages from `PEBatchRequestRetry.fifo` by provider code in an in-memory `ProviderBuffer`. When a provider's buffer reaches 1,000 messages or has been inactive for 10 minutes, it flushes the accumulated requests to S3 as a JSON file under `{provider}/requests/{provider}|{uuid}.json`. This batching pattern allows batch-mode providers (e.g. QL2-style scrapers) to pick up large request payloads from S3 rather than processing individual SQS messages.

**Input**:
- SQS: `PEBatchRequestRetry.fifo`
- Config: `batch.request.bucket` property

**Output**:
- S3: `s3://{batch.request.bucket}/{provider}/requests/{provider}|{uuid}.json`

---

### retry-lambda

**Type**: AWS Lambda Function (arm64, container image)
**Trigger**: SQS event source mapping on `PERetry.fifo`; batch size 10, max concurrency 32
**Compute**: 624 MB memory, 60 second timeout
**Main Class**: `com.threevictors.priceeye.scheduling.lambda.functions.retry.PERetryLambdaHandler`
**Source**: `source/lambda-functions/retry-lambda/`
**CloudFormation template**: `source/deploy/commonfiles/retry-lambda.yaml`

**What it does**: Processes retry messages from `PERetry.fifo`. On startup it reads `PERetryLambda.properties` to determine whether auto-scheduler mode is active, selecting either `PEASRetryProcessor` (hierarchy-based substitution) or `PERetryProcessor` (legacy substitution map). For each SQS batch it validates each `PERetryMessage`, determines substitute provider/site-code pairings, separates cached from non-cached requests, then publishes all substitution requests to provider queues. Uses `ReportBatchItemFailures` to allow partial batch retries. A CloudWatch alarm fires when function duration equals its timeout.

**Input**:
- SQS: `PERetry.fifo` (event source mapping, batch size 10)
- Config: `PERetryLambda.properties`

**Output**:
- SQS: per-provider queues (substitution requests)

---

### metrics-loader

**Type**: AWS Lambda Function (arm64, container image)
**Trigger**: EventBridge rule — S3 Object Created event for any `*.csv` file in bucket `s3-atp-3victors-{env}-use1-sitemetrics`
**Compute**: 624 MB memory, 120 second timeout
**Main Class**: `com.threevictors.priceeye.scheduling.lambda.functions.metrics.loader.MetricsLoader`
**Source**: `source/lambda-functions/metrics-loader/`
**CloudFormation template**: `source/deploy/commonfiles/metrics-loader.yaml`

**What it does**: Listens for new CSV files landing in the site-metrics S3 bucket. Depending on the S3 key prefix (`retry.prefix`, `cache.prefix`, or `capacity.prefix` from `MetricsLoader.properties`), it routes the file to `MetricsParser` for the appropriate parse-and-insert path (`parseAndInsertRetry`, `parseAndInsertCache`, or `parseAndInsertCapacity`), loading the parsed metrics into the PriceEye scheduling database. Files with non-matching prefixes are silently ignored. A CloudWatch alarm fires when function duration equals its timeout.

**Input**:
- EventBridge: S3 Object Created event (bucket: `s3-atp-3victors-{env}-use1-sitemetrics`, key: `*.csv`)
- S3: the CSV file at the notified key
- Config: `MetricsLoader.properties` (bucket name, key prefixes)

**Output**:
- Database (scheduling DB): retry metrics, cache metrics, or capacity metrics records

---

### runtime-metrics-data-loader

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: EventBridge CRON — `cron(10 * * * ? *)` (10 minutes past each hour)
**Compute**: Configured at `source/deploy/commonfiles/runtime-metrics-data-loader.yaml`; default 1024 MB / 1024 CPU; requires `glue:CreatePartition`, `glue:GetPartitions`, `glue:GetTable`
**Main Class**: `com.threevictors.priceeye.scheduling.data.loaders.runtime.metrics.PERuntimeMetricsDataLoader`
**Source**: `source/data-loaders/runtime-metrics-data-loader/`

**What it does**: Runs 10 minutes past each hour to capture the previous hour's runtime metrics. Queries Redshift (via `CoreRedshiftReader`) for any missing request hours relative to the run time, always including the previous full hour. For each hour to backfill, it fetches `PERuntimeMetric` records from Redshift in parallel (5 threads), writes them as Parquet to S3 under `runtime_metrics/YYYY/MM/DD/HH/output.parquet`, then registers a new Glue partition for that path. Only hours with data receive partitions.

**Input**:
- Redshift (core analytics DB): `PERuntimeMetric` records for missing/current hours
- Config: `PERuntimeMetricsDataLoader.properties` (`output.bucket.name`, `glue.database`, `glue.table`, `profile.role.arn`)

**Output**:
- S3: `s3://{output.bucket}/runtime_metrics/YYYY/MM/DD/HH/output.parquet`
- Glue: new partition on table `runtime_metrics` (or configured table name) in the scheduling Glue database

---

### preemptive-metrics-data-loader

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: EventBridge CRON — runs near the top of each hour (xx:50 based on source comment)
**Compute**: Configured at `source/deploy/commonfiles/preemptive-metrics-data-loader.yaml`; default 1024 MB / 1024 CPU; requires `glue:CreatePartition`, `glue:GetPartitions`, `glue:GetTable`
**Main Class**: `com.threevictors.priceeye.scheduling.data.loaders.preemptive.metrics.PEPreemptiveMetricsDataLoader`
**Source**: `source/data-loaders/preemptive-metrics-data-loader/`

**What it does**: Fetches in-progress request data from Redshift for the current hour using `CoreRedshiftReader.getCurrentRequestProgress()`, then passes each `PEPreemptiveMetric` record through `PreemptiveChecker.decideToPreempt()` in parallel (10 threads) to determine whether each request should be flagged for preemption. Results are written as Parquet to S3 under `preemptive_metrics/YYYY/MM/DD/HH/output.parquet`, and a Glue partition is created for each run. Errors are logged with the `NOTIFY` prefix for alerting integration.

**Input**:
- Redshift (core analytics DB): current-hour request progress records (`PEPreemptiveMetric`)
- Config: `PEPreemptiveMetricsDataLoader.properties` (`output.bucket.name`, `glue.database`, `glue.table`, `profile.role.arn`)

**Output**:
- S3: `s3://{output.bucket}/preemptive_metrics/YYYY/MM/DD/HH/output.parquet`
- Glue: new partition on the configured preemptive-metrics table in the scheduling Glue database

---

### site-metric-validation-data-loader

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: EventBridge CRON — `cron(0 5 * * ? *)` (daily at 05:00 UTC)
**Compute**: Configured at `source/deploy/commonfiles/site-metric-validation-data-loader.yaml`; default 1024 MB / 1024 CPU; requires `glue:CreatePartition`, `glue:GetPartitions`, `glue:GetTable`
**Main Class**: `com.threevictors.priceeye.scheduling.data.loaders.site.metric.validation.PESiteMetricValidationDataLoader`
**Source**: `source/data-loaders/site-metric-validation-data-loader/`

**What it does**: Runs once per day at 05:00 UTC to validate the previous day's site metrics. Queries Redshift for `PESiteMetricValidation` records for yesterday's sales date, compares observed collection throughput and delay against expected site-metric thresholds, then writes the results as Parquet to S3 under `site-metric-validation/YYYY/MM/DD/output.parquet`. A Glue partition with `(year, month, day)` partition keys is registered for each day that produces data.

**Input**:
- Redshift: `PESiteMetricValidation` records for yesterday
- Config: `PESiteMetricValidationDataLoader.properties`

**Output**:
- S3: `s3://{output.bucket}/site-metric-validation/YYYY/MM/DD/output.parquet`
- Glue table: `site_metric_validation` (partitions: `year`, `month`, `day`)

---

### site-metrics-reporting

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: EventBridge rule — fires when `site-metrics-stepfunction` (external Step Function) completes with status `SUCCEEDED`
**Compute**: Configured at `source/deploy/commonfiles/site-metrics-reporting.yaml`; 2048 MB / 1024 CPU; requires `ses:SendEmail`
**Main Class**: `com.threevictors.aws.priceeye.reporting.util.site.metrics.reporting.SiteMetricsReporter`
**Source**: `source/util/site-metrics-reporting/`

**What it does**: Runs a suite of configuration sanity checks using `ConfigurationChecker` (which validates active customer site codes against site hierarchies, provider codes, and site metric thresholds) then builds a summary report of all errors and warnings and emails it to a configured list of recipients via Amazon SES. The task is triggered by the successful completion of an external `site-metrics-stepfunction` Step Function, ensuring checks run after fresh site-metrics data is available. Email body is capped at 9,000 characters; retries up to 3 times on send failure.

**Input**:
- EventBridge: `Step Functions Execution Status Change` from `site-metrics-stepfunction` (SUCCEEDED)
- Database (scheduling DB): active customer site codes, site hierarchies, site metrics, customer collections, provider codes
- Config: `SiteMetricsReporter.properties` (`sender.identity`, `recipient.emails`)

**Output**:
- SES: email summary to configured recipient list

---

### scheduling-comparison (data loader)

**Type**: ECS Fargate Task (ARM64, Linux)
**Trigger**: Triggered ad hoc or after a schedule generation run (no built-in CRON in this repo); requires `generationId` as a command-line argument
**Compute**: Configured at `source/deploy/commonfiles/scheduling-comparison.yaml`; default 1024 MB / 1024 CPU; requires `glue:CreatePartition`, `glue:GetPartitions`, `glue:GetTable`
**Main Class**: `com.threevictors.aws.priceeye.scheduling.data.loaders.scheduling.comparison.PESchedulingComparisonDataLoader`
**Source**: `source/data-loaders/scheduling-comparison/`

**What it does**: Merges runtime and auto-scheduler scheduling comparison data for a given generation ID. Reads `PESchedulingComparison` records from Redshift (tagged `generationType = runtime`) and from the scheduling database for the given `generationId` (tagged `generationType = autoschedule`), combines them, then writes a unified Parquet file to S3. A Glue partition is created for the run date. The resulting dataset enables direct comparison of runtime vs. auto-scheduler request allocation decisions.

**Input**:
- Redshift: runtime scheduling comparison records
- Database (scheduling DB): auto-scheduler comparison for `generationId`
- Config: `PESchedulingComparison.properties`
- CLI args: `<generationId>`

**Output**:
- S3: `s3://{output.bucket}/scheduling-comparison/YYYY/MM/DD/output.parquet`
- Glue table: `schedule_comparison` (partitions: `generation_id`, `date`)

---

## Glue Databases

### `glue-atp-3victors-{env}-use1-scheduling_db`

All tables are external Parquet tables backed by S3 in `s3-atp-3victors-{env}-use1-as-scheduled-comparison/`.

---

#### Table: `schedule_comparison`

**Location**: `s3://s3-atp-3victors-{env}-use1-as-scheduled-comparison/scheduling-comparison/`
**Format**: Parquet (MapredParquet)
**Produced by**: `scheduling-comparison` data loader

| Column | Type | Notes |
|---|---|---|
| customer | varchar(32) | |
| customercollectionid | bigint | |
| customercollectionname | varchar(128) | |
| observationtimestamp | timestamp | |
| customersitecode | varchar(64) | |
| providercode | varchar(32) | |
| sitecode | varchar(32) | |
| totalrequests | bigint | |
| validrequests | bigint | |
| generationtype | varchar(32) | `autoschedule` or `runtime` |

_Partition keys: `generation_id` (int), `date` (int)_

---

#### Table: `auto_schedule_output`

**Location**: `s3://s3-atp-3victors-{env}-use1-as-scheduled-comparison/auto-schedule-output/`
**Format**: Parquet (MapredParquet)
**Produced by**: `auto-schedule-converter`

| Column | Type | Notes |
|---|---|---|
| requestid | bigint | |
| plan_date | int | |
| plan_hour | int | |
| collectiontype | varchar(128) | |
| hourlycollectionplanid | bigint | |
| timeboxstartdate | int | |
| timeboxstarttime | int | |
| timeboxenddate | int | |
| timeboxendtime | int | |
| ownersequence | int | |
| collectionid | bigint | |
| collectioncustomer | varchar(128) | |
| collectionname | varchar(128) | |
| collectionfrequencyrequestowner | varchar(128) | |
| collectionearlieststarttime | int | |
| collectionexpecteddeliverytime | int | |
| collectionstatus | varchar(128) | |
| collectioncustomerpackagingid | bigint | |
| collectionhint | varchar(128) | |
| inputrequestid | bigint | |
| inputfilename | varchar(128) | |
| inputreference | varchar(128) | |
| inputcabin | varchar(128) | |
| inputmaxstops | int | |
| inputlengthofstay | varchar(128) | |
| inputcustomersitecode | varchar(128) | |
| inputpos | varchar(128) | |
| inputdepartdow | varchar(128) | |
| inputreturndow | varchar(128) | |
| inputfrequency | varchar(128) | |
| ownertimeboxstartdate | int | |
| ownertimeboxstarttime | int | |
| ownertimeboxenddate | int | |
| ownertimeboxendtime | int | |
| provider | varchar(128) | |
| site | varchar(128) | |
| hierarchycustomer | varchar(128) | |
| hierarchycustomersitecode | varchar(128) | |
| hierarchypriority | int | |
| ownersitecategory | varchar(128) | |
| qualityscore | double | |
| relevancyscore | double | |
| requestcabin | varchar(128) | |
| requestmaxstops | int | |
| requesttriptype | varchar(128) | |
| requestpassengercount | int | |
| requestrefundable | boolean | |
| requestpriority | int | |
| requestdropdeadtimestamp | bigint | |
| requestlos | int | |
| requestdepartdate | int | |
| requestreturndate | int | |
| requestorigin | varchar(128) | |
| requestdestination | varchar(128) | |
| crawl_date | int | |
| crawl_hour | int | |
| requestpos | varchar(128) | |
| requestcarriercodes | varchar(128) | |
| requestconnectionairports | varchar(128) | |
| requestap | int | |

_Partition keys: `generation_id` (int)_

---

#### Table: `site_metric_validation`

**Location**: `s3://s3-atp-3victors-{env}-use1-as-scheduled-comparison/site-metric-validation/`
**Format**: Parquet (MapredParquet)
**Produced by**: `site-metric-validation-data-loader` (daily)

| Column | Type | Notes |
|---|---|---|
| crawled | bigint | Timestamp in milliseconds |
| crawl_hour | int | Hour of day (0–23) |
| providercode | varchar(32) | |
| sitecode | varchar(32) | |
| total_requests | bigint | |
| firstresponse | bigint | Timestamp in milliseconds |
| lastresponse | bigint | Timestamp in milliseconds |
| delay_in_minutes | int | Time between crawl and first response |
| collection_time_in_minutes | int | Time between first and last response |
| rate_per_hour | int | Collection rate per hour |
| metrics_delay | int | Expected delay from site_metrics table |
| metrics_tph | int | Expected throughput per hour from site_metrics table |

_Partition keys: `year` (int), `month` (int), `day` (int)_

---

## Glue Databases Summary

| Database | Tables |
|---|---|
| `glue-atp-3victors-{env}-use1-scheduling_db` | `schedule_comparison`, `auto_schedule_output`, `site_metric_validation` |

Additional tables for runtime and preemptive metrics (e.g. `runtime_metrics`, `preemptive_metrics`) are registered dynamically by the data-loader tasks using `GluePartitionCreator`; their database and table names are read from the respective `.properties` config files at runtime and are not defined in the CloudFormation YAML in this repo.

---

## Infrastructure Summary

| Resource | Count | Notes |
|---|---|---|
| ECS Fargate Task Definitions | 9 | `auto-scheduler`, `auto-schedule-converter`, `schedule-cutover`, `scheduler`, `preemptive-polling`, `batch-request-retry`, `runtime-metrics-data-loader`, `preemptive-metrics-data-loader`, `site-metric-validation-data-loader`, `site-metrics-reporting`, `scheduling-comparison` (some defined outside this repo; 9 have CloudFormation templates here) |
| Lambda Functions | 2 | `retry-lambda`, `metrics-loader` |
| Step Functions | 0 defined here | `site-metrics-stepfunction` is external; this repo subscribes to its completion event |
| S3 Buckets | 4 | `pe-as-persistence`, `sitemetrics`, `as-converted-persistence`, `as-scheduled-comparison` |
| SQS Queues (main) | 3 | `PEAutoScheduleRequest.fifo`, `PERetry.fifo`, `PEBatchRequestRetry.fifo` |
| SQS Queues (DLQ) | 3 | `FAILED-PEAutoScheduleRequest.fifo`, `FAILED-PERetry.fifo`, `FAILED-PEBatchRequestRetry.fifo` |
| Glue Databases | 1 | `glue-atp-3victors-{env}-use1-scheduling_db` |
| Glue Tables (CloudFormation) | 3 | `schedule_comparison`, `auto_schedule_output`, `site_metric_validation` |
| Glue Tables (dynamic, runtime) | 2+ | runtime_metrics, preemptive_metrics (partitions created at runtime) |
| EventBridge Rules | 7+ | 5 CRON rules (schedule-cutover, scheduler, preemptive-polling, runtime-metrics-data-loader, site-metric-validation-data-loader) + 1 S3 event rule (metrics-loader) + 1 Step Function success rule (site-metrics-reporting) |
| CloudWatch Alarms | 2 | Timeout alarms on both Lambda functions |
| ECS Cluster | 1 | `ECS-priceeye` (shared, not defined in this repo) |

---

## Key Source Modules

| Module path | Artifact | Role |
|---|---|---|
| `source/auto-scheduling/auto-scheduler/` | `PEAutoScheduler` | Multi-threaded schedule generation engine |
| `source/auto-scheduling/auto-schedule-converter/` | `AutoScheduleConverter` | Flattens generated schedule to Parquet |
| `source/auto-scheduling/schedule-cutover/` | `PEScheduleCutover` | Atomic generation activation |
| `source/auto-scheduling/configuration-check/` | `ConfigurationChecker` | Sanity checks on site metrics & hierarchies |
| `source/auto-scheduling/auto-scheduler-lite/` | `AutoSchedulerLite` | Prototype capacity-check library (not deployed independently) |
| `source/scheduler/` | `PESchedulerApplication` | Runtime request dispatcher |
| `source/preemptive-polling/` | `PEPreemptivePollingApplication` | Preemptive substitute request dispatch |
| `source/batch-request-retry/` | `PEBatchRequestRetryApplication` | Batch-provider request buffering to S3 |
| `source/lambda-functions/retry-lambda/` | `PERetryLambdaHandler` | SQS Lambda retry processor |
| `source/lambda-functions/metrics-loader/` | `MetricsLoader` | S3-triggered CSV metrics ingestion Lambda |
| `source/data-loaders/runtime-metrics-data-loader/` | `PERuntimeMetricsDataLoader` | Hourly runtime metrics → Parquet/Glue |
| `source/data-loaders/preemptive-metrics-data-loader/` | `PEPreemptiveMetricsDataLoader` | Hourly preemptive metrics → Parquet/Glue |
| `source/data-loaders/site-metric-validation-data-loader/` | `PESiteMetricValidationDataLoader` | Daily site-metric validation → Parquet/Glue |
| `source/data-loaders/scheduling-comparison/` | `PESchedulingComparisonDataLoader` | Per-generation scheduling comparison → Parquet/Glue |
| `source/util/site-metrics-reporting/` | `SiteMetricsReporter` | Configuration sanity report via SES |
| `source/common/` | `SiteHierarchyFactory` | Shared site hierarchy loading utility |
| `source/util/publishers/` | `ProviderPublisher`, `RetryPublisher` | Shared SQS publish helpers |

---

## Environment Naming Convention

All AWS resource names use an `{env}` token, which maps to an environment suffix string passed at deploy time:

| Token | Environment |
|---|---|
| `3vprod` | Production |
| `3vgold` | Gold / staging |
| `3vdev` | Development |

Example: `s3-atp-3victors-3vprod-use1-sitemetrics` is the production site-metrics bucket.

All resources are deployed to `us-east-1` (region suffix `-use1` in bucket names).

All ECS tasks run on ARM64 / Linux (Amazon Corretto 17 JVM) in the `ECS-priceeye` cluster, inside private VPC subnets with no public IP, secured by the `FMSSecuritygroupApp` security group.
