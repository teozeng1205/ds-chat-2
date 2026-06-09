# priceeye-v2

> Multi-provider airline pricing aggregation system that executes flight shopping requests across 20+ airline and GDS APIs, caches and audits the results, packages itineraries for delivery, and feeds analytics pipelines.

> **Current branch**: `develop` — this document reflects the `develop` branch. The **`master` branch represents what is currently running in production**.

---

## Architecture Overview

```
[EventBridge: rate(1 min)]                         [EventBridge: rate(5 min)]
        │                                                    │
        ▼                                                    ▼
[ecs-launcher (Lambda)]                       [dropdead-detector (Lambda)]
 monitors SQS depths,                          detects past-deadline requests
 launches ECS tasks                                         │
        │                                                    ▼
        ▼                                    [SQS: PEPackagerScheduler.fifo]
[Provider Lambdas (20+)]                                    │
 AA, UA, DL, AS, WN, QL2,                                   ▼  (ARCHIVED)
 TP, AI, TS, WN ingest, etc.           [packager-scheduler-queue → packager-launcher]
        │                                        [packager-components]
        ▼                                                    │
[SQS: PEPollCache.fifo]                           S3: packaged-output
        │                                                    │
        ▼                                                    ▼  (ARCHIVED)
[persist-response-data (Lambda)]              [delivery: S3 / Azure / GCloud]
        │
        ├──► DynamoDB: itinerary cache
        │
        ├──► [SQS: PETPFCCache.fifo]
        │          └──► [persist-tpfc-cache (Lambda)] ──► Redis (20-day TTL)
        │
        ├──► [SQS: PEPublishRawSearch.fifo]
        │          └──► [publish-raw-search (Lambda)]
        │                    └──► Kinesis: ingest-priceeye-raw-search
        │                              └──► S3: dataset-ingest-{env}/
        │
        └──► Kinesis: audit-persist  (via common-audit-delivery-stream-publisher)
                   │
                   ▼
          [persist-audit-data (Lambda)]  batch=100, parallelism=4
                   ├──► [SQS: PEPersistAuditDataMySQL.fifo]
                   │          └──► [persist-audit-data-mysql (Lambda)] ──► Aurora MySQL
                   └──► [SQS: PEPersistAuditDataRedshift.fifo]
                              └──► [persist-audit-data-redshift (Lambda)] ──► Redshift

─── Input Side ──────────────────────────────────────────────────────────────
[EventBridge cron / manual]
        │
        ▼
[input-importer (ECS)]
 reads CSV from SFTP/S3/Rsync
        │
        └──► S3: input-archive → [data-parsing lib] → SQS provider queues

─── Billing ─────────────────────────────────────────────────────────────────
[Glue Job: billing-customer-daily-request-unload SUCCEEDED]
        │
        ▼
[daily-billing (ECS)] ──► S3: s3-atp-3victors{env}-use1-billing/
                      ──► Glue: customer_daily_requests_v1/v2

─── Monitoring & Analytics ──────────────────────────────────────────────────
[status-monitoring (Lambda, manual)]
        └──► Redshift queries ──► Slack alerts

[response-rate (ECS, daily)]
        └──► Aurora reads → Aurora: response_rate table

[response-converter (ECS batch)]
        └──► S3: PECacheLoader → S3: dataset-fbc-cache-csv/
```

---

## Orchestration

There are no AWS Step Functions in this repo. The pipeline is coordinated entirely through **EventBridge rules**, **SQS fan-out**, and **Kinesis streams**.

### EventBridge Rules

| Rule | Schedule | Target |
|------|----------|--------|
| ecs-launcher rule | `rate(1 minute)` | Lambda: ecs-launcher |
| dropdead-detector rule | `rate(5 minutes)` | Lambda: dropdead-detector |
| input-importer rule | `cron(0,25 * * * ? *)` (every hour at :00 and :25) | ECS: input-importer |
| daily-billing rule | Glue job `billing-customer-daily-request-unload` SUCCEEDED event | ECS: daily-billing |
| Scheduled task rules | Configurable cron per task | Various ECS tasks |

---

## Components

_(Ordered by pipeline stage — earliest / trigger-driven first, then downstream processors.)_

---

### input-importer

**Type**: ECS Fargate Task
**Trigger**: EventBridge cron — every hour at :00 and :25 (`cron(0,25 * * * ? *)`)
**Compute**: 1024 MB RAM, 1 vCPU (ARM64)

**What it does**: Imports customer-submitted flight search request files from SFTP, Rsync, or S3 sources. Reads per-customer configuration files (`input-[sftp|rsync|s3]-[customer]-[collectionId].properties`) to locate and retrieve CSV files, then archives them and hands them to the data-parsing library which produces `PEInputRequest` objects enqueued for the provider lambdas.

**Input**:
- SFTP servers / Rsync endpoints / S3 buckets (per-customer config)
- CSV files with search criteria (origin, destination, dates, carriers, cabin, stops, passenger count)

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-pe-input-archive/{hour}/` (raw archive)
- SQS: provider request queues (parsed `PEInputRequest` objects)
- Aurora MySQL: import records

---

### ecs-launcher

**Type**: Lambda Function
**Trigger**: EventBridge — `rate(1 minute)`
**Compute**: 512 MB RAM, 60 s timeout

**What it does**: Acts as the dynamic autoscaler for the pipeline. Every minute it reads each monitored SQS queue's depth via CloudWatch metrics and computes how many ECS tasks should be running (`tasks = messages / divisor + 1`, max 10). It launches missing tasks and also monitors running tasks for stalls — if a task has been running for more than 2 hours it fires a Slack alert. Queue-to-task mappings and divisors are loaded from Aurora MySQL via `ECSLauncherConfigs`. Uses 16 concurrent threads for parallel queue evaluation.

**Input**:
- CloudWatch: `ApproximateNumberOfMessagesVisible` on configured SQS queues
- Aurora MySQL: `ECSLauncherConfigs` (queue name, task definition, divisor)

**Output**:
- ECS `RunTask` calls (launches provider or processing tasks)
- Slack alerts for stalled tasks

---

### Provider Lambdas (20+)

**Type**: Lambda Functions (one per provider/airline)
**Trigger**: SQS — per-provider request queues (fed by input-importer / ecs-launcher)
**Compute**: Varies per provider (512–2048 MB, 60–300 s timeout)

**What they do**: Execute HTTP shopping requests against airline and GDS APIs, parse the XML/JSON responses using provider-specific deserialization models (`external-data` library), and produce `PERequestResponse` objects. Each provider lambda handles authentication, rate limiting, and error classification for its respective API. Providers include:

| Category | Providers |
|----------|-----------|
| US Carriers | AA (American), UA (United), DL (Delta), AS (Alaska), WN (Southwest), HA (Hawaiian), SP (Spirit), B6 (JetBlue) |
| International | LA (LATAM), TS (Thai), UX (Air Europa), VY (Vueling), CM (Copa), AR (Aerolíneas) |
| GDS / Aggregators | TP (Travelport), QL2, AI (Amadeus), MTC, PIT |
| Direct Ingest | DL ingest (e-Stream), WN ingest (Southwest direct feed) |

**Input**:
- SQS: provider-specific request queues
- HTTP: airline/GDS API endpoints
- Configuration: `ConfigurationReader` (per-provider credentials, endpoints)

**Output**:
- SQS: `PEPollCache.fifo` (serialized `PERequestResponse` objects)
- Audit records via `common-audit-data-stream-publisher` → Kinesis: `audit-persist`

---

### persist-response-data

**Type**: Lambda Function
**Trigger**: SQS — `PEPollCache.fifo` (batch size: 10, max concurrency: 128)
**Compute**: 1024 MB RAM, 60 s timeout

**What it does**: The central fan-out hub for raw provider responses. It deserializes each `PERequestResponse`, inserts itineraries into the DynamoDB itinerary cache (max 2000 per key), and then fans the message out to three downstream queues based on business rules:
1. **PEPackagerScheduler.fifo** — for all eligible responses (skips preemptive and enrichment categories)
2. **PETPFCCache.fifo** — only for Travelport (TP) + site 1G + no connections
3. **PEPublishRawSearch.fifo** — for all providers except TP, TS, QL2Vacation

It also writes `CacheLoaderAudit` records and handles S3 payload offloading for oversized messages.

**Input**:
- SQS: `PEPollCache.fifo`

**Output**:
- DynamoDB: itinerary cache
- SQS: `PEPackagerScheduler.fifo`
- SQS: `PETPFCCache.fifo`
- SQS: `PEPublishRawSearch.fifo`
- Kinesis: `audit-persist` (`CacheLoaderAudit` records)

---

### persist-tpfc-cache

**Type**: Lambda Function
**Trigger**: SQS — `PETPFCCache.fifo` (batch size: 10, max concurrency: 64)
**Compute**: 1024 MB RAM, 60 s timeout; DLQ: `FAILED-PETPFCCache.fifo` (4 retries)

**What it does**: Persists Travelport Full Content (TPFC) pricing data into Redis for fast downstream lookups. Only processes Travelport (TP) responses with site code `1G` and no connection airports. Builds a structured Redis hash: the cache key encodes the route/dates/cabin/carriers/stops/refundability, and each hash field encodes the itinerary cabin/carrier/stops/refundable combination. Values are Kryo-serialized `TPFCCacheValue` objects containing itinerary counts and brand-level pricing statistics (min, mean, max, percentiles). TTL is set to 20 days. Batches Redis writes every 60 seconds and emits `TPFCCacheAudit` records.

**Input**:
- SQS: `PETPFCCache.fifo`

**Output**:
- Redis: TPFC hash cache (20-day TTL, Kryo-serialized values)
- Kinesis: `audit-persist` (`TPFCCacheAudit` records)

---

### publish-raw-search

**Type**: Lambda Function
**Trigger**: SQS — `PEPublishRawSearch.fifo` (batch size: 10, max concurrency: 128)
**Compute**: 1024 MB RAM, 60 s timeout

**What it does**: Converts `PERequestResponse` objects into the standardized `RawSearch` format (using the `data-converter` library) and publishes them to a Kinesis stream that feeds legacy analytics and data-lake pipelines. Skips providers TP, TS, and QL2Vacation. The converter maps PE parameters to `RawSearch` fields including airport-to-city code mapping, dominant carrier identification (by flight duration), fare basis codes, brand info, and penalty data. The source field is set to `PRICE_EYE|{provider}|{siteCode}`.

**Input**:
- SQS: `PEPublishRawSearch.fifo`

**Output**:
- Kinesis: `ingest-priceeye-raw-search` (Snappy-compressed Avro `RawSearch` records)
- → S3: `s3://dataset-ingest-{env}/` (via Kinesis Firehose downstream)

---

### persist-audit-data

**Type**: Lambda Function
**Trigger**: Kinesis — `kinesis-atp-3victors{env}-use1-audit-persist` (batch size: 100, parallelism factor: 4, max retries: 2, batching window: 20 s)
**Compute**: 1024 MB RAM, 240 s timeout; DLQ: `FAILED-persist-audit-data`

**What it does**: Consumes compressed audit record batches from the Kinesis audit stream, deserializes Snappy-compressed `CommonAuditRecordBatch` objects, validates varchar field lengths against the database schema, and fans each batch to two parallel persistence queues — one for MySQL and one for Redshift.

**Input**:
- Kinesis: `kinesis-atp-3victors{env}-use1-audit-persist`

**Output**:
- SQS: `PEPersistAuditDataMySQL.fifo` (batch size: 10, max concurrency: 64)
- SQS: `PEPersistAuditDataRedshift.fifo` (batch size: 10, max concurrency: 16)

---

### persist-audit-data-mysql

**Type**: Lambda Function
**Trigger**: SQS — `PEPersistAuditDataMySQL.fifo` (batch size: 10, max concurrency: 64)
**Compute**: 1024 MB RAM, 60 s timeout

**What it does**: Persists common audit records (all audit types: Cache, CacheLoader, CollectionRun, Delivery, Enrichment, Packager, ProviderRequest, ProviderResponse, Retry, Scheduler, etc.) into Aurora MySQL for operational querying and near-real-time dashboards.

**Input**:
- SQS: `PEPersistAuditDataMySQL.fifo`

**Output**:
- Aurora MySQL: audit tables (per-type)

---

### persist-audit-data-redshift

**Type**: Lambda Function
**Trigger**: SQS — `PEPersistAuditDataRedshift.fifo` (batch size: 10, max concurrency: 16)
**Compute**: 1024 MB RAM, 60 s timeout

**What it does**: Persists common audit records into Redshift for analytical querying. Lower concurrency (16) than MySQL due to Redshift connection limits.

**Input**:
- SQS: `PEPersistAuditDataRedshift.fifo`

**Output**:
- Redshift: audit tables

---

### dropdead-detector

**Type**: Lambda Function
**Trigger**: EventBridge — `rate(5 minutes)`
**Compute**: 512 MB RAM, 60 s timeout

**What it does**: Runs every 5 minutes and queries the database for requests that have passed their "dropdead" deadline — the latest time at which they must be packaged. For each such request it publishes a `PEReadyForPackaging` message to the `PEPackagerScheduler.fifo` queue to force the packaging workflow to proceed even if normal triggers have not fired. Tracks a `lastDropdead` watermark timestamp to avoid reprocessing.

**Input**:
- Aurora MySQL: `PriceEyeReportReader.getDropDeadRequests()` (watermark-based)

**Output**:
- SQS: `PEPackagerScheduler.fifo` (`PEReadyForPackaging` messages)

---

### packager-scheduler / packager-scheduler-queue / packager-launcher / packager-components / packager-work-expire _(ARCHIVED)_

**Type**: ECS / Java queue consumers
**Status**: Source archived; moved out of active build

**What they did**: Consumed `PEReadyForPackaging` messages from `PEPackagerScheduler.fifo`, batched them into work groups (`PEPackagerWorkGroup`), and launched ECS packager tasks via `packager-launcher`. The `packager-components` library formatted itineraries into various output formats (itinerary, leg, WN-specific). `packager-work-expire` periodically deleted old work records from the database using chunked deletes. Still referenced in CloudFormation artifacts for replay scenarios.

---

### delivery _(ARCHIVED)_

**Type**: ECS / Java queue consumers
**Status**: Source archived

**What it did**: Delivered packaged itinerary files to customer destinations. Supported S3, Azure Blob Storage, and Google Cloud Storage targets with optional PGP encryption. Sub-components: s3, azure, gcloud, manifest (delivery manifests), work-expire (cleanup). Configuration was per-customer property files.

---

### daily-billing

**Type**: ECS Fargate Task
**Trigger**: EventBridge — Glue job `billing-customer-daily-request-unload` SUCCEEDED event
**Compute**: 1024 MB RAM, 1 vCPU (ARM64)

**What it does**: Generates daily customer billing records by querying the database for request volumes per customer per day, formats them as JSON billing records (one per customer per day), and uploads them to S3 partitioned by date. The trigger fires after the upstream Glue billing unload job completes, ensuring the source data is available.

**Input**:
- Aurora MySQL: `billingReader.getDailyRequestsByCustomerForDate()` + customer metadata map

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-billing/{yyyy-MM-dd}/` (JSON billing records)
- Glue: `customer_daily_requests_v1` / `customer_daily_requests_v2` (cataloged by Glue job)

**Table Schema** (customer_daily_requests_v2):

| Column | Type |
|--------|------|
| customer | string |
| cust_run_dt | string |
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
| providercode | string |

_Partition keys: `sales_date` (bigint)_

---

### response-rate

**Type**: ECS Fargate Task (batch, runs daily)
**Trigger**: EventBridge cron (daily)
**Compute**: 1024 MB RAM, 1 vCPU

**What it does**: Calculates hourly response-rate metrics for each provider/site combination. Deletes records older than 7 days, re-populates the `provider_response_time` table from raw response data, then aggregates using median statistics (via Apache Commons Math `DescriptiveStatistics`) to compute `response_rate = count / (median_duration_minutes / 60)` and stores results in the `response_rate` table.

**Input**:
- Aurora MySQL: raw response records (past 7 days)

**Output**:
- Aurora MySQL: `provider_response_time` table (per-response timing data)
- Aurora MySQL: `response_rate` table (aggregated rates by provider/site/date)

---

### status-monitoring

**Type**: Lambda Function
**Trigger**: Manual / per-collection schedule (invoked externally per customer cycle)
**Compute**: 928 MB RAM, 120 s timeout

**What it does**: Monitors PriceEye collection health for each customer by querying Redshift for the current hour's data. Checks four conditions per customer-provider-site-category combination and sends Slack alerts when thresholds are breached: (1) zero valid requests, (2) response rate < 90%, (3) success rate < 90%, (4) high substitution rate in packaged results.

**Input**:
- Redshift: `PriceEyeRedshiftDemoReader.getMonitoringRows()` (current-hour metrics)

**Output**:
- Slack: alert messages via `SlackHelper`

---

### response-converter

**Type**: ECS Fargate Task (batch job)
**Trigger**: Manual or scheduled cron
**Compute**: 1024 MB RAM, 1 vCPU; 5-thread pool

**What it does**: Reads `PECacheLoader` serialized itinerary files from S3, deserializes them, and converts them to a flat CSV format for fare-basis-code (FBC) and carrier analysis. Filters for a specific set of carriers (QR, AZ, LA, DL, UA, IB, AC, LH, B6, UX, TP, DY, D8, G3, CM, AS, AR). Exports in batches of 1 million rows.

**Input**:
- S3: `s3://3v-upload-bucket/PECacheLoader/{date}/*` (Kryo-serialized itinerary lists)

**Output**:
- S3: `s3://dataset-fbc-cache-csv/datasets/{date}/{timestamp}-output.csv`
- Columns: provider_code, site_code, pos, origin, destination, dates, price, carrier, booking_code, fare_basis_code, cabin, operating_carrier, dominant_carriers, brand, etc.

---

### orphaned-input-requests-cleanup

**Type**: ECS Fargate Task
**Trigger**: Scheduled cron
**Compute**: 1024 MB RAM, 1 vCPU

**What it does**: Cleans up orphaned input requests that were never processed, freeing database space and preventing stale data accumulation.

**Input**: Aurora MySQL: orphaned input request records

**Output**: Aurora MySQL: deleted records

---

### Shared Libraries

| Library | Purpose |
|---------|---------|
| `data-parsing` | Parses customer CSV input files into `PEInputRequest` objects; supports multiple column formats and validates fields |
| `data-converter` | Converts `PERequestResponse` → `RawSearch` for Kinesis ingest; maps airports to cities, identifies dominant carriers |
| `data-serde` | Serialization/deserialization for PriceEye data objects (Avro, Kryo) |
| `common-audit-data-stream-publisher` | Publishes `CommonAuditRecord` batches to Kinesis data streams (Snappy-compressed Avro) |
| `common-audit-delivery-stream-publisher` | Publishes audit records to Kinesis Firehose → S3 Parquet |
| `common-audit-generator` | Test/load generator for all 13 audit record types; used for load and sanity testing |
| `external-data` | JAXB/GSON data models for airline API responses (Sabre NDC, Thai AirShopping, etc.) |
| `enrichment` | Enriches responses with OAG flight data, tax regression, fare basis codes, brand hacks |
| `dao` | Database access objects for Aurora MySQL and Redshift |
| `input-common` | Shared input processing utilities |
| `diagnostics` | Troubleshooting utility: reads ECS task metadata, tests DB connectivity on port 3306 |

---

## Glue Databases

| Database | Tables |
|----------|--------|
| `glue-atp-3victors-{env}-use1-billing_db` | `customer_daily_requests_v1`, `customer_daily_requests_v2` |
| `glue-atp-3victors-{env}-use1-priceeye_output_db` | `provider_error_ai`, `provider_success_ai`, `provider_success_ql2` |
| `glue-atp-3victors-{env}-use1-tax_reg_db` | `tax_reg_aa_output_v1`, `tax_reg_market_list_v1`, `tax_reg_output_com_v1`, `tax_reg_output_v1`, `tax_reg_raw_com_v1`, `tax_reg_raw_v1` |
| `glue-atp-3victors-{env}-use1-midt_external_db` | `midt_daily` (60+ columns, partition: `feed_date`) |
| `glue-atp-3victors-{env}-use1-common_output_db` | `common_output_format` (100+ columns, partitions: `sales_date`, `customer`) |
| `glue-atp-3victors-{env}-use1-data_lakes_db` | `city_summary`, `daily_representative_itinerary_v4` |
| `glue-atp-3victors-{env}-use1-priceeye_audits_db` | `cache_audit`, `cache_loader_audit`, `collection_run_audit` |
| `glue-atp-3victors-{env}-athena-tables` | `tp_poller_output_table` (partition: `sales_date`, `sales_hour`) |

### Notable Table Schemas

**provider_success_ai / provider_success_ql2** — S3: `s3://s3-atp-3victors{env}-use1-pe-ai-provider-archive/v1/` (CSV, partition: `sales_date`)

| Column | Type |
|--------|------|
| market | string |
| site | string |
| pos | string |
| cxr | string |
| ddate | int |
| rdate | int |
| dstp | string |
| droute | string |
| rstp | string |
| rroute | string |
| base | decimal |
| taxes | decimal |
| fare | decimal |
| currency | string |
| reference | string |
| fbc1–fbc6 | string |
| fuel_surcharge | decimal |
| operating_carriers | string |
| booking_classes | string |

_Partition keys: `sales_date` (int)_

**tp_poller_output_table** — S3: `s3://s3-atp-3victors{env}-use1-tp-poller-output/v1/` (Parquet)

| Column | Type |
|--------|------|
| timestamp | string |
| pos | string |
| originairportcode | string |
| destinationairportcode | string |
| departdate | string |
| returndate | string |
| pcc | string |
| providerrequestid | string |
| requestcarriers | string |
| requestconnections | string |
| cabin | string |
| xmlcontent | string |

_Partition keys: `sales_date` (int), `sales_hour` (int)_

---

## Infrastructure Summary

| Resource | Count | Details |
|----------|-------|---------|
| ECS Fargate Tasks | ~10 | input-importer, daily-billing, response-rate, response-converter, orphaned-cleanup, replay-*, sanity-test-auto, perftest-input-generator |
| Lambda Functions | ~30 | ecs-launcher, dropdead-detector, persist-audit-data, persist-response-data, persist-tpfc-cache, publish-raw-search, status-monitoring, persist-audit-data-mysql, persist-audit-data-redshift + 20+ provider lambdas |
| Kinesis Data Streams | 2 | `audit-persist` (4 shards), `ingest-priceeye-raw-search` |
| Kinesis Firehose | ~10 | audit delivery streams, common-output delivery streams (per env) |
| SQS FIFO Queues | 10+ | PEPollCache, PEPublishRawSearch, PETPFCCache, PEPersistAuditDataMySQL, PEPersistAuditDataRedshift, PEPackagerScheduler + FAILED-* DLQs |
| Glue Databases | 8 | See table above |
| Glue Tables | 20+ | See table above |
| EventBridge Rules | 5+ | rate(1 min), rate(5 min), Glue billing event, input-importer cron, scheduled task crons |
| Aurora MySQL | 1 | Operational data: audit, metadata, response rate, scheduler state |
| Redshift | 1 | Analytics: monitoring rows, historical audit, provider success/error archives |
| Redis | 1 | TPFC cache (Travelport Full Content, 20-day TTL) |
| DynamoDB | 1 | Itinerary cache (per-key, max 2000 entries) |
| ECR Repository | 1 | `732267085676.dkr.ecr.us-east-1.amazonaws.com/3victors/priceeyev2/` |

### Environment Naming Convention

All resources follow the pattern:

```
s3://s3-atp-3victors{env}-use1-{purpose}/
kinesis-atp-3victors{env}-use1-{purpose}
glue-atp-3victors-{env}-use1-{database}_db
```

| AWS Account | Environment Tag |
|-------------|----------------|
| 891377228241 | `-3vdevds` |
| 590183652635 | `-3vdev` |
| 590183916591 | `-3vgold` |
| 539247469204 | `-3vprod` |

### Compute Platform

- All ECS tasks: **Fargate ARM64**, `awsvpc` networking, 3-subnet VPC (SubnetApp0/1/2), security group `FMSSecuritygroupApp`
- All Lambdas: **ARM64 Docker images** from ECR, VPC-attached, CloudWatch Logs 7-day retention
- Timeout alarms for all Lambdas → SNS: `HighPriorityAlarm`
- Secrets: all credentials via `secretsmanager:GetSecretValue`
