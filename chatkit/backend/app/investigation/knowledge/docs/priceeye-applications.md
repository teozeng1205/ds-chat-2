# priceeye-applications

> Real-time flight fare collection and delivery pipeline: filters raw pricing responses from data providers, optionally enriches them, packages them into customer-specific formats, and delivers the output to customer destinations via multiple transport methods (S3, SFTP, email, API, GCS, Azure, Google Drive).

> **Current branch**: `develop` — _this document reflects the `develop` branch. The `master` branch represents what is currently running in production; there may be differences._

---

## Architecture Overview

```
[Upstream Ingest System]  (out of scope for this repo)
        │
        └──► SQS: PEGlobalFilter.fifo
                        │
                        ▼
        ┌─────────────────────────────┐
        │   global-filter-lambda      │  (Lambda, ARM64, 624 MB)
        │   - toss MultiCarrier       │
        │   - toss NoOutboundLegs     │
        └──────────┬──────────────────┘
                   │
         ┌─────────┴──────────────────────┐
         │  enrichment needed?            │
         ▼ yes                            ▼ no
SQS: PEEnrichment.fifo          SQS: PEPollCache.fifo ──► [Other system]
         │
         ▼
  ┌─────────────────┐
  │   enrichment    │  (ECS Fargate, 1 vCPU / 1 GB)
  │  OAG, tax,      │
  │  brand, cache   │
  └────────┬────────┘
           │
           └──► SQS: PEPollCache.fifo ──► [Other system]
                                               │
                                     SQS: PEPackagerScheduler.fifo
                                               │
                                               ▼
                                ┌──────────────────────────────┐
                                │  packager-scheduler-queue     │  (ECS Fargate)
                                │  Groups PEReadyForPackaging   │
                                │  into PEPackagerWorkGroup     │
                                └─────────────┬────────────────┘
                                              │
                                    SQS: PEPackagerLauncher.fifo
                                              │
                                              ▼
                                  ┌───────────────────────┐
                                  │  packager-launcher    │  (ECS Fargate)
                                  │  Launches ECS tasks   │
                                  └──────────┬────────────┘
                                             │  ecs:RunTask packager-application
                                             ▼
                                  ┌───────────────────────┐
                                  │  packager-application │  (ECS Fargate, dynamic)
                                  │  Customer filters,    │
                                  │  formatting, dedup    │
                                  └──────────┬────────────┘
                               ┌─────────────┤
                               │             │
                               ▼             ▼
              S3: price-eye-customer-delivery/  SQS: PECommonOutput.fifo
                               │                       │
       ┌───────────────────────┘                       ▼
       │  (records in Aurora: delivery_type_queue)  ┌─────────────────┐
       │                                            │  common-output  │  (ECS Fargate)
       │                                            │  Parquet writer │
       │                                            └────────┬────────┘
       │                                                     │
       │                                       S3: price-eye-common-output/
       │
       │  [EventBridge cron: every hour at :00 and :25]
       │                     │
       │                     ▼
       │          ┌──────────────────────┐
       │          │  delivery-scheduler  │  (ECS Fargate, scheduled)
       │          │  Reads delivery      │
       │          │  configs from Aurora │
       │          └──────────┬───────────┘
       │                     │
       │           SQS: PEDeliveryCombinerWork.fifo
       │                     │
       └────────────────────►│
                             ▼
                  ┌──────────────────────┐
                  │  delivery-combiner   │  (ECS Fargate, 200 GiB ephemeral)
                  │  Downloads, combines,│
                  │  archives files      │
                  └──────────┬───────────┘
                             │
                  S3: price-eye-delivery-archive/
                             │
           ┌─────────────────┼──────────────────────────────┐
           ▼                 ▼          ▼          ▼         ▼
   PEDeliveryS3       PEDeliverySFTP  PEDeliveryEmail  PEDeliveryAPI  ...etc
           │                 │          │          │         │
     ┌─────┴──┐        ┌────┴──┐  ┌───┴──┐  ┌───┴──┐  ┌───┴──┐
     │delivery│        │delivery│  │deliver│  │deliver│  │deliver│
     │  -s3   │        │ -sftp  │  │ -email│  │  -api │  │ -azure│
     └────────┘        └────────┘  └───────┘  └───────┘  └───────┘
                                                (+ gcloud, gdrive)

── Separate scheduled pipelines ──────────────────────────────────────────

[EventBridge cron: every hour at :00 and :25]
   ├──► ECS: blacklist-ai       ──► Aurora: market_date_blacklist
   ├──► ECS: blacklist-ql2      ──► Aurora: blacklist_market_summary
   ├──► ECS: blacklist-spark-ai ──► Redis:  blacklistMarketSummary_*
   └──► ECS: blacklist-spark-ql2

[EventBridge cron: 2:15 PM, 5:15 PM, 9:15 PM UTC]
   └──► Lambda: delivery-monitor ──► Slack (delivery volume alerts)
```

---

## Orchestration

There are no AWS Step Functions in this repository. Orchestration is entirely **queue-driven** — components hand off work to each other via SQS FIFO queues. The main sequencing logic lives in:

- **global-filter-lambda**: routes filtered responses to enrichment or directly to the poll-cache path
- **packager-scheduler-queue**: batches `PEReadyForPackaging` messages into work groups, then enqueues them to the launcher
- **delivery-scheduler**: reads Aurora's `delivery_type_queue` on a cron and triggers the combiner/delivery chain
- **delivery-combiner**: collects and combines files, then fans out to delivery-type-specific queues

---

## Components

_(Ordered by pipeline position — earliest first.)_

---

### global-filter-lambda

**Type**: AWS Lambda Function (ARM64, Docker image)
**Trigger**: SQS `PEGlobalFilter.fifo` — batch size 10, max concurrency 32
**Compute**: 624 MB memory, 60 s timeout

**What it does**: Processes batches of `PERequestResponse` messages from the upstream ingest system. For each response it applies toss rules — itineraries with multiple marketing carriers across legs (`MultiCarrier`) or missing outbound legs (`NoOutboundLegs`) are dropped. Valid itineraries are routed based on provider/site configuration: if the provider+site combination requires enrichment (queried from Aurora), the response is published to `PEEnrichment.fifo`; otherwise it goes directly to `PEPollCache.fifo` for downstream packager processing. Publishes `GlobalFilterAudit` and `GlobalFilterAuditSummary` records to Kinesis.

**Input**:
- SQS: `PEGlobalFilter.fifo` — `PERequestResponse` messages (may include S3 payload offloading for large messages)
- Aurora DB: enrichment-required table (provider+site config), cabin hierarchy, site map, expanded input requests (fallback if request missing from message)

**Output**:
- SQS: `PEEnrichment.fifo` — responses needing enrichment
- SQS: `PEPollCache.fifo` — responses skipping enrichment
- Kinesis: `GlobalFilterAudit`, `GlobalFilterAuditSummary` audit records

**Monitoring**: CloudWatch Alarm fires SNS `HighPriorityAlarm` if duration ≥ 60,000 ms (timeout hit)

---

### enrichment

**Type**: ECS Fargate Task (ARM64, queue-driven, persistent consumer)
**Trigger**: SQS `PEEnrichment.fifo`
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap

**What it does**: Consumes `PERequestResponse` messages from the enrichment queue and applies a configurable chain of enrichers to each itinerary. The enricher chain is determined per provider+site and includes: OAG flight data (operating carrier, flight times, equipment, duration, intermediate stops), booking code and fare basis code cache lookups, tax calculations (combined tax, tax cache, tax engine, tax regression), brand assignments (brand hacks, brand hacks v2, brand hacks DB), FareCloud pricing, and directional price normalization. After enrichment the response is published to `PEPollCache.fifo` for packager scheduling. Per-itinerary enrichment success/failure is captured in `PEEnrichmentAudit` records.

**Input**:
- SQS: `PEEnrichment.fifo` — `PERequestResponse` messages
- Aurora DB: enrichment configuration, OAG cache, tax tables, brand tables, booking code/fare basis cache
- SageMaker: `InvokeEndpoint` (for ML-based enrichments where configured)

**Output**:
- SQS: `PEPollCache.fifo` — enriched `PERequestResponse`
- Kinesis: `PEEnrichmentAudit` records

---

### packager-scheduler-queue

**Type**: ECS Fargate Task (ARM64, queue-driven, persistent consumer)
**Trigger**: SQS `PEPackagerScheduler.fifo` — `PEReadyForPackaging` messages (populated by the poll-cache/prep layer, outside this repo)
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap

**What it does**: Aggregates `PEReadyForPackaging` messages (individual fare responses ready to be packaged) into batch work groups bounded by configurable max itineraries (default 1,000,000) and max requests (default 10,000) per group. Once a work group is complete, it publishes a `PEPackagerWorkGroup` message to `PEPackagerLauncher.fifo`. A 1-minute periodic scheduler flushes any incomplete work groups. This component effectively controls parallelism and batch sizing for the packager tier.

**Input**:
- SQS: `PEPackagerScheduler.fifo` — `PEReadyForPackaging` messages
- Aurora DB: customer collection map

**Output**:
- SQS: `PEPackagerLauncher.fifo` — `PEPackagerWorkGroup` messages
- Aurora DB: packager work records (via `PriceEyeWriter`)

---

### packager-launcher

**Type**: ECS Fargate Task (ARM64, queue-driven, persistent consumer)
**Trigger**: SQS `PEPackagerLauncher.fifo` — `PEPackagerWorkGroup` messages
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap

**What it does**: Acts as a foreman — picks up `PEPackagerWorkGroup` messages from the launcher queue and immediately calls `ecs:RunTask` to spin up a new `packager-application` ECS task, passing the customer name, collection ID, group ID, and SQS receipt handle as task arguments. The launched task will then independently consume and process that work group directly.

**Input**:
- SQS: `PEPackagerLauncher.fifo` — `PEPackagerWorkGroup` messages

**Output**:
- ECS: `ecs:RunTask packager-application` — one task per work group
- (The launched task inherits the SQS receipt handle to extend message visibility and delete on completion)

---

### packager-application

**Type**: ECS Fargate Task (ARM64, dynamically launched per work group)
**Trigger**: Launched by `packager-launcher` via `ecs:RunTask`; consumes its assigned `PEPackagerWorkGroup` from `PEPackagerLauncher.fifo`
**Compute**: Configured per deployment; uses up to 4 processing threads

**What it does**: The core customer-facing packaging engine. Reads raw fare itineraries for a specific customer collection and work group from Aurora and S3. Applies the full packaging pipeline: customer-specific filtering (economy, cabin, carrier, connection, duration, stop count, price, codeshare, joint business, market share filters), currency conversion, GDS surcharges, deduplication, brand formatting, pre/post hooks (demo data, tax settings, price overrides), and output formatting (leg-level, itinerary-level, WN format, etc.). Writes packaged output files to `price-eye-customer-delivery` S3 and records file URIs in Aurora's `delivery_type_queue` table with status `ready`. Also publishes `PECommonOutputMessage` records to `PECommonOutput.fifo` for Parquet archival.

**Input**:
- SQS: `PEPackagerLauncher.fifo` — `PEPackagerWorkGroup` (customer, collection ID, group ID)
- Aurora DB: customer packaging config, site map, GDS surcharges, airport metadata, itinerary data
- S3: OAG cache data

**Output**:
- S3: `price-eye-customer-delivery/` — customer-formatted output files
- Aurora DB: `delivery_type_queue` rows with status `ready` (triggers delivery-scheduler)
- SQS: `PECommonOutput.fifo` — `PECommonOutputMessage` records for Parquet archival
- Kinesis: `PEPackagerAudit` records

---

### common-output

**Type**: ECS Fargate Task (ARM64, queue-driven, persistent consumer)
**Trigger**: SQS `PECommonOutput.fifo`
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap

**What it does**: Consumes `PECommonOutputMessage` records published by the packager and writes them as Parquet files to the `price-eye-common-output` S3 bucket, partitioned by customer and hour. Handles currency normalization, great-circle distance calculations, airport lat/lon enrichment, and cabin-type normalization. Files are rolled and uploaded when they reach `MAX_FILE_RECORDS` rows or when the processing hour changes.

**Input**:
- SQS: `PECommonOutput.fifo` — `PECommonOutputMessage` records
- Aurora DB: airport metadata (lat/lon, country, city codes), currency conversion tables

**Output**:
- S3: `price-eye-common-output/{customer}/{YYYY}/{MM}/{DD}/{HH}/` — Parquet files, partitioned by hour

---

### delivery-scheduler

**Type**: ECS Fargate Task (ARM64, scheduled)
**Trigger**: EventBridge cron — every hour at :00 and :25 (`cron(0,25 * * * ? *)`)
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap

**What it does**: On each run, reads all active customer delivery configurations from Aurora and scans the `delivery_type_queue` table for files with status `ready`. For each configuration whose cron frequency matches the current run window (accounting for customer timezone), it groups the ready file URIs and publishes a `PECombinerWork` message to `PEDeliveryCombinerWork.fifo`, transitioning file status to `queued`. This is the clock that drives the delivery tier. Publishes `DeliverySchedulerAudit` records for observability.

**Input**:
- Aurora DB: `customer_delivery_config` table (delivery configs per customer), `delivery_type_queue` table (ready file URIs), customer active status and timezone

**Output**:
- SQS: `PEDeliveryCombinerWork.fifo` — `PECombinerWork` messages (customer, deliveryId, groupId, fileUris)
- Aurora DB: updates `delivery_type_queue` status to `queued`
- Kinesis: `DeliverySchedulerAudit` records

---

### delivery-combiner

**Type**: ECS Fargate Task (ARM64, queue-driven, persistent consumer)
**Trigger**: SQS `PEDeliveryCombinerWork.fifo`
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap; **200 GiB ephemeral storage**

**What it does**: The aggregation and fan-out hub for the delivery tier. Consumes `PECombinerWork` messages from the scheduler queue. For each work item: downloads the referenced output files from `price-eye-customer-delivery` S3 to local ephemeral storage (`/data/`), optionally combines multiple files into a single output file (respecting per-customer `combine` and `file_name_divider` / `last_file_suffix` config), archives the result to `price-eye-delivery-archive`, and publishes a `PEDeliveryMessage` to the appropriate delivery-type SQS queue (`PEDeliveryAPI.fifo`, `PEDeliveryS3.fifo`, `PEDeliverySFTP.fifo`, `PEDeliveryEmail.fifo`, `PEDeliveryGCloud.fifo`, `PEDeliveryGDrive.fifo`, or `PEDeliveryAzure.fifo`). A heartbeat thread extends message visibility every 75 seconds to prevent re-processing during large file operations.

**Input**:
- SQS: `PEDeliveryCombinerWork.fifo` — `PECombinerWork` messages
- S3: `price-eye-customer-delivery/` — packaged output files (source)
- Aurora DB: customer delivery config, customer collection map

**Output**:
- S3: `price-eye-delivery-archive/{customer}/{customerCollectionId}/{YYYY}/{MM}/{DD}/{HH}/` — archived delivery files
- SQS: `PEDeliveryAPI.fifo` / `PEDeliveryAzure.fifo` / `PEDeliveryEmail.fifo` / `PEDeliveryGCloud.fifo` / `PEDeliveryGDrive.fifo` / `PEDeliveryS3.fifo` / `PEDeliverySFTP.fifo` — `PEDeliveryMessage` for the relevant delivery method
- Kinesis: `DeliveryCombinerAudit` records

---

### delivery-s3 / delivery-api / delivery-azure / delivery-gcloud / delivery-gdrive / delivery-sftp

**Type**: ECS Fargate Task (ARM64, queue-driven, persistent consumer)
**Trigger**: SQS `PEDeliveryS3.fifo` / `PEDeliveryAPI.fifo` / `PEDeliveryAzure.fifo` / `PEDeliveryGCloud.fifo` / `PEDeliveryGDrive.fifo` / `PEDeliverySFTP.fifo`
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap

**What it does**: Each delivery-type application is a specialization of `PEAbstractDeliveryApplication`. It consumes `PEDeliveryMessage` messages, downloads the referenced archived file(s) from `price-eye-delivery-archive`, optionally applies PGP encryption, and transmits to the customer's configured destination:
- **delivery-s3**: copies file to a customer-configured S3 bucket (supports STS role assumption, access key auth, or default credentials)
- **delivery-api**: POSTs file contents to a customer HTTP/HTTPS API endpoint
- **delivery-azure**: uploads to Azure Blob Storage
- **delivery-gcloud**: uploads to Google Cloud Storage
- **delivery-gdrive**: uploads to Google Drive
- **delivery-sftp**: transmits via SFTP using customer credentials from Secrets Manager
- **delivery-email**: sends file as email attachment via SES (`ses:SendRawEmail`)

Customer connection credentials and delivery parameters are loaded from AWS Secrets Manager via the Config Server.

**Input**:
- SQS: type-specific delivery queue (e.g., `PEDeliveryS3.fifo`)
- S3: `price-eye-delivery-archive/` — file to deliver
- Aurora DB: customer delivery config (destination URL, credentials config name)
- Secrets Manager: per-customer delivery credentials

**Output**:
- Customer destination (S3 bucket, API endpoint, Azure container, GCS bucket, Google Drive folder, SFTP server, or email recipient)
- Aurora DB: updates `delivery_type_queue` status to `delivered`
- Kinesis: `PEDeliveryAudit` records

---

### delivery-email

**Type**: ECS Fargate Task (ARM64, queue-driven)
**Trigger**: SQS `PEDeliveryEmail.fifo`
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap

**What it does**: Variant of the delivery tasks specialized for email delivery. Uses AWS SES (`ses:SendRawEmail`) to send output files as email attachments to configured customer recipients. IAM policy includes `AmazonSESFullAccess` (distinguished from other delivery types).

_(See delivery-s3/api/... above for shared behavior.)_

---

### delivery-monitor

**Type**: AWS Lambda Function (ARM64, Docker image)
**Trigger**: EventBridge scheduled rule — `cron(15 14,17,21 * * ? *)` (2:15 PM, 5:15 PM, 9:15 PM UTC daily)
**Compute**: 512 MB memory, 60 s timeout; runs inside VPC

**What it does**: Monitors delivery health for all active non-internal customers. On each invocation, lists files in `price-eye-delivery-archive` S3 for today, last week, and two weeks ago (for each customer's `customerCollectionId`). Compares today's delivery file count and total size against the two-week average; if deviations exceed ±10%, an `ALERT` flag is appended. Sends the formatted report to a Slack channel via webhook.

**Input**:
- EventBridge scheduled trigger (fires 3× daily)
- S3: `price-eye-delivery-archive/{customer}/{customerCollectionId}/` — file listings
- Aurora DB: active customer list

**Output**:
- Slack webhook: delivery volume summary with ALERT flags where applicable

**Monitoring**: CloudWatch Alarm fires SNS `HighPriorityAlarm` if duration ≥ 60,000 ms

---

### blacklist-ai / blacklist-ql2

**Type**: ECS Fargate Task (ARM64, scheduled)
**Trigger**: EventBridge cron — every hour at :00 and :25 (`cron(0,25 * * * ? *)`)
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap; SageMaker `InvokeEndpoint` permission

**What it does**: Maintains the market date blacklist used by the collection system to exclude low-quality fare data. Reads from Aurora's `JobMonitoring` table to determine the next unprocessed UTC date. Rebuilds `market_date_blacklist` by calculating per-provider data quality exclusions using a configurable look-back window (at minimum 7 days, at most `blacklist_days × 2`). Updates the `blacklist_market_summary` table. Writes the current blacklist summary into a Redis cluster (key prefix `blacklistMarketSummary_`) with a TTL, and removes stale Redis summary keys. The `-AI` variant processes only the AI provider; the `-QL2` variant processes only QL2; the base (no suffix) variant processes all non-explicit providers.

**Input**:
- Aurora DB: `JobMonitoring` table (last processed date tracking), `provider` config table, raw quality data

**Output**:
- Aurora DB: `market_date_blacklist` table (rebuilt and updated), `blacklist_market_summary` table
- Redis: `blacklistMarketSummary_{provider}` sorted set, `blacklistMarketSummaryLatest_{provider}` key
- Aurora DB (`JobMonitoring`): updates last run time and last sales date processed

---

### blacklist-spark-ai / blacklist-spark-ql2 / blacklist-spark

**Type**: ECS Fargate Task (ARM64, scheduled, Apache Spark)
**Trigger**: EventBridge cron — every hour at :00 and :25 (`cron(0,25 * * * ? *)`)
**Compute**: 1 vCPU, 1024 MB, `-Xmx2G` heap

**What it does**: A Spark-based alternative implementation of the blacklist pipeline. Uses Apache Spark DataFrames to compute blacklist summaries — reads fare observation data (including from S3), computes per-market-date quality metrics using distributed aggregations (count distinct observations, sum, etc.), and writes results back to Aurora and Redis. Produces the same outputs as the non-Spark variant (`market_date_blacklist`, `blacklist_market_summary`, Redis cache) but leverages Spark for higher-volume processing. Provider scoping follows the same `-AI` / `-QL2` / base convention.

**Input**:
- Aurora DB: `JobMonitoring`, provider config
- S3: raw fare observation data (via Spark reads)

**Output**:
- Aurora DB: `market_date_blacklist`, `blacklist_market_summary`
- Redis: `blacklistMarketSummary_*` sorted sets with 72-hour TTL

---

### packager-work-expire / delivery-work-expire / delivery-type-queue-expiry

**Type**: ECS Fargate Task (queue-driven or scheduled housekeeping utilities)

**What they do**: Expire stale work items and reset stuck queue entries back to `ready` so they can be retried. These prevent the delivery and packaging pipelines from stalling when a consumer dies mid-processing without completing a work item.

---

## SQS Queues

All queues are **FIFO** with `ContentBasedDeduplication` enabled and a max message size of 262 KB. Dead letter queues (prefixed `FAILED-`) receive messages after 4 failed receive attempts.

### Application Processing Queues (`priceeye-applications-queues.yaml`)

| Queue | Visibility Timeout | Consumer | Purpose |
|---|---|---|---|
| `PEGlobalFilter.fifo` | 900 s | global-filter-lambda | Raw pricing responses from ingest system |
| `PEEnrichment.fifo` | 900 s | enrichment ECS task | Responses needing OAG/tax/brand enrichment |
| `PEPackagerScheduler.fifo` | 900 s | packager-scheduler-queue | Ready-to-package fare responses |
| `PEPackagerLauncher.fifo` | 1800 s | packager-launcher + packager-application | Work groups awaiting ECS task launch |
| `PECommonOutput.fifo` | 900 s | common-output ECS task | Packaged records for Parquet archival |
| `PEDelivery.fifo` | 900 s | _(general delivery)_ | General-purpose delivery queue |

### Delivery Queues (`delivery-queues.yaml`)

| Queue | Visibility Timeout | Consumer | Destination |
|---|---|---|---|
| `PEDeliveryCombinerWork.fifo` | 300 s | delivery-combiner | Combiner work items from delivery-scheduler |
| `PEDeliveryAPI.fifo` | 900 s | delivery-api ECS task | HTTP/HTTPS API endpoints |
| `PEDeliveryAzure.fifo` | 900 s | delivery-azure ECS task | Azure Blob Storage |
| `PEDeliveryEmail.fifo` | 900 s | delivery-email ECS task | Email via SES |
| `PEDeliveryGCloud.fifo` | 900 s | delivery-gcloud ECS task | Google Cloud Storage |
| `PEDeliveryGDrive.fifo` | 900 s | delivery-gdrive ECS task | Google Drive |
| `PEDeliveryS3.fifo` | 900 s | delivery-s3 ECS task | Customer S3 buckets |
| `PEDeliverySFTP.fifo` | 900 s | delivery-sftp ECS task | SFTP servers |

---

## S3 Buckets

| Bucket | Written by | Purpose |
|---|---|---|
| `price-eye-customer-delivery` | packager-application | Per-customer packaged output files (pre-delivery staging) |
| `price-eye-common-output` | common-output | Parquet archive of all packaged fare records, partitioned by customer+hour |
| `price-eye-delivery-archive` | delivery-combiner | Final combined/archived delivery files before customer transmission |

---

## Key Aurora DB Tables

| Table | Written by | Read by | Purpose |
|---|---|---|---|
| `market_date_blacklist` | blacklist, blacklist-spark | Upstream collection system | Market+date combinations excluded from fare collection |
| `blacklist_market_summary` | blacklist, blacklist-spark | Upstream collection system | Summary stats for blacklisted markets |
| `delivery_type_queue` | packager-application | delivery-scheduler, delivery-combiner | File URIs ready for delivery, tracks status (ready→queued→delivered) |
| `customer_delivery_config` | _(ops)_ | delivery-scheduler, delivery-combiner | Per-customer delivery schedule, type, destination config |
| `JobMonitoring` | blacklist tasks | blacklist tasks | Tracks last processed date for each recurring job |

---

## Infrastructure Summary

| Resource | Count |
|---|---|
| ECS Fargate Task Definitions (scheduled) | 6 (blacklist-ai, blacklist-ql2, blacklist-spark-ai, blacklist-spark-ql2, blacklist-spark, delivery-scheduler) |
| ECS Fargate Task Definitions (queue-driven) | 9+ (enrichment, packager-scheduler-queue, packager-launcher, packager-application, common-output, delivery-combiner, delivery-email, + delivery types S3/API/Azure/GCloud/GDrive/SFTP via generic template) |
| Lambda Functions | 2 (global-filter-lambda, delivery-monitor) |
| SQS FIFO Queues | 14 primary + 14 dead-letter = 28 total |
| EventBridge Scheduled Rules | 4 (blacklist-ai, blacklist-ql2, blacklist-spark-ai+ql2+base, delivery-scheduler at :00/:25; delivery-monitor 3×/day) |
| S3 Buckets (referenced) | 3 |
| Glue Tables | 0 (no Glue catalog; Parquet written directly to S3) |
| Step Functions | 0 |
| ECS Clusters | 1 per environment (`ecs-{env}-use1-price-eye`) |

---

## Notes for Newcomers

- **No Step Functions**: The pipeline is entirely queue-driven. Follow the SQS queue names to trace execution flow.
- **`PEQueueInfo` enum**: The canonical list of queue names is defined in `PEQueueInfo` (a shared library, not in this repo). Queue names match the FIFO queue names defined in the CloudFormation YAML files.
- **Config Server**: Runtime application properties (S3 bucket names, job names, batch sizes, customer-specific delivery credentials) are loaded at startup from a central Config Server (AWS Secrets Manager + properties files), not hardcoded.
- **ARM64 everywhere**: All Fargate tasks and Lambdas use `arm64` (Graviton) for cost efficiency.
- **`/data/` directory**: Queue-driven tasks that process files (combiner, common-output, delivery tasks) write intermediate files to `/data/` on the Fargate ephemeral filesystem; this directory is cleaned up on shutdown.
- **Audit trail**: Almost every component publishes structured audit records (`PEAuditType.*Audit`) to Kinesis Data Streams via `CommonAuditDataStreamPublisher` for downstream observability.
