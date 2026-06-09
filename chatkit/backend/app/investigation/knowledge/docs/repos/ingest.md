# ingest

> A multi-provider airline/travel data ingestion platform that polls external sources, publishes raw search data to per-provider Kinesis streams, and fans out through reader workers that write processed Avro datasets to S3.

> **Current branch**: `develop` — this document reflects the `develop` branch. The **`master` branch represents what is currently running in production**; check out `master` for the exact production configuration.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           FETCHER-BASED PROVIDERS                                       │
│                                                                                         │
│  [Atlas API (atriptech.com)]         ─► [atlas-fetcher  ECS Service]                   │
│  [Delta S3 (cross-acct 494528744408)] ─► [delta-fetcher  ECS Service]                  │
│  [Southwest SFTP (AWS Transfer)]      ─► [southwest-fetcher  ECS Service]              │
│                                              │                                          │
│                              Kinesis: ingest-{provider}-raw-search                      │
│                                              │                                          │
│                                   [Reader  ECS Service]                                 │
│                                   (KCL consumer)                                        │
│                                        │        │          │                            │
│                               [DemandSummary] [SearchWI] [IngestCache]                 │
│                               Plugin          Plugin       Plugin                       │
│                                    │              │            │                        │
│                               S3 Avro         S3 Avro    Kinesis: ingest-cache         │
└──────────────────────────────────────────────────────────────────────────────────────  ┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        DIRECT-FEED PROVIDERS (no fetcher)                               │
│                                                                                         │
│  [Estream upstream]  ─► Kinesis: ingest-estream-raw-search                             │
│  [PriceEye upstream] ─► Kinesis: ingest-priceeye-raw-search                            │
│  [Sstream upstream]  ─► Kinesis: ingest-sstream-raw-search                             │
│                                   │                                                     │
│                          [Reader  ECS Service]                                          │
│                          (KCL consumer)                                                 │
│                              │          │          │                                    │
│                    [DemandSummary] [SearchWI] [IngestCache]                             │
│                         (varies per provider)                                           │
│                              └──────────┴──► S3: dataset-ingest-{env}/                │
└──────────────────────────────────────────────────────────────────────────────────────  ┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        AVAILABILITY (standalone pipeline)                               │
│                                                                                         │
│  [Travelport SFTP (xfer.prod.travelport.com)] ──►                                      │
│       [availability-fetcher  ECS Service]                                              │
│              │                                                                          │
│       S3: 3v-ingest-inbound-travelport-availability-{env}  (staging bucket)           │
│              │                                                                          │
│       [availability-reader  ECS Service]                                               │
│              │                                                                          │
│       Redis cluster + S3 persistence                                                    │
└──────────────────────────────────────────────────────────────────────────────────────  ┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        MONITORS  (all providers)                                        │
│                                                                                         │
│  [EventBridge cron: every 5 min]                                                       │
│       │                                                                                 │
│  [{provider}-monitor  ECS Scheduled Task]                                              │
│       ├── reads CloudWatch metrics (MAX_AGE, IncomingRecords, CURRENT_READ_RATE)       │
│       ├── makes ECS scale-in / scale-out decisions                                     │
│       └── detects & repairs KCL lease table "time warp" corruption                    │
└──────────────────────────────────────────────────────────────────────────────────────  ┘
```

---

## S3 Output Path Schema

All processed data lands in a single consolidated bucket per environment. The S3 key is structured as:

```
s3://dataset-ingest-{env}/{source}/{streamName}/v1/{year}/{month}/{day}/{hour}/
    {source}-{streamName}-avro-stream-{year}-{month}-{day}-{hour}-{min}-{sec}-{uuid}.avro
```

| Segment | Example values |
|---------|----------------|
| `{env}` | _(empty = prod)_, `-dev`, `-3vdev` |
| `{source}` | `delta`, `atlas`, `estream`, `priceeye`, `southwest`, `sstream` |
| `{streamName}` | `demand-summary`, `search-with-itineraries` |

Files are **Avro format with Snappy compression**, uploaded via S3 multipart.

---

## Deployment Patterns

Two CloudFormation templates in `deploy/commonfiles/` cover every component:

| Template | When used | Key behaviour |
|----------|-----------|---------------|
| `service.yaml` | Long-running ingestion (fetchers & readers) | ECS Fargate service, `DesiredCount=0` by default, scale manually |
| `scheduled-task.yaml` | Periodic jobs (monitors, fetchers) | EventBridge cron rule → ECS task run; default schedule `cron(*/10 * * * ? *)` |

**Common settings (both templates):**
- Launch type: FARGATE, ARM64 Linux
- Network: private subnets, `FMSSecuritygroupApp` security group
- `StopTimeout`: 120 s (SIGTERM graceful shutdown)
- CloudWatch log group: `ingest/{StackName}`, 7-day retention
- IAM: broad S3, Kinesis, DynamoDB, CloudWatch, Secrets Manager access

---

## Components

Components are listed provider-by-provider, with each provider's fetcher → reader → monitor in order.

---

### atlas-fetcher

**Type**: ECS Fargate Service (continuous)
**Trigger**: Continuous polling; sleeps 60 s between requests
**Compute**: 1024 CPU units, 1280 MB RAM, Java heap 1 GB

**What it does**: Calls the Atlas HTTP API (`https://api-sg.atriptech.com/gather/file.do`) using credentials from AWS Secrets Manager, parses returned search data, and publishes records onto the Atlas Kinesis stream. Threads: 1 poll/process, 16 parse, 16 publish. Handles SIGTERM with a 120 s flush.

**Input**:
- HTTP: `https://api-sg.atriptech.com/gather/file.do` (Atlas external API)
- Secrets Manager: `x-atlas-client-id`, `x-atlas-client-secret`

**Output**:
- Kinesis stream: `ingest-atlas-raw-search` (32 shards in prod)

---

### atlas-reader

**Type**: ECS Fargate Service (continuous)
**Trigger**: KCL consumer on `ingest-atlas-raw-search`
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Consumes records from `ingest-atlas-raw-search` via the Kinesis Client Library (KCL) and fans each record out to the configured plugins. The dynamic read rate adjusts between 100–1 000 records/sec based on queue depth. Plugin thread pool: 24 threads for fanout.

**Input**:
- Kinesis stream: `ingest-atlas-raw-search`
- DynamoDB KCL lease table: `FargateIngestAtlas`

**Output**:
- S3 (DemandSummary plugin): `s3://dataset-ingest-{env}/atlas/demand-summary/v1/…`
- S3 (SearchWithItineraries plugin): `s3://dataset-ingest-{env}/atlas/search-with-itineraries/v1/…`
- Kinesis stream (IngestCache plugin): `ingest-cache`

---

### atlas-monitor

**Type**: ECS Fargate Scheduled Task
**Trigger**: EventBridge cron — every 5 minutes
**Compute**: 1024 CPU units, 1024 MB RAM

**What it does**: Reads CloudWatch metrics (`MAX_AGE`, `IncomingRecords`, `CURRENT_READ_RATE`, `THROTTLED_READS`) for the Atlas pipeline and makes ECS scale-in/scale-out decisions (min 0, max 4 reader tasks). Detects KCL lease-table "time warps" (max search age jumps > 4× expected interval), which indicate DynamoDB lease corruption; when detected it stops the service, purges the lease table, and restarts. Scale-in cooldown: 1 800 s; scale-out cooldown: 2 700 s.

**Input**: CloudWatch metrics for Atlas pipeline; ECS service state

**Output**: ECS `UpdateService` API calls (desired count changes); DynamoDB `DeleteItem` calls (lease table repair)

---

### delta-fetcher

**Type**: ECS Fargate Service (continuous)
**Trigger**: Continuous polling; sleeps 60 s between S3 LIST calls
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Assumes a cross-account IAM role (`arn:aws:iam::494528744408:role/delegate-admin-edw-atpco-lambda-role`) to list and download files from Delta's partner S3 bucket. Each file is parsed and published to the Delta Kinesis stream, then archived and deleted from the source bucket. Thread pools: 1 process, 16 parse, 16 publish, 16 archive, 16 delete.

**Input**:
- S3 (cross-account): `s3://dl-use1-edw-494528744408-atpco/` (Delta partner account 494528744408)

**Output**:
- S3 archive: `s3://3v-ingest-inbound-delta-archive-{env}/`
- Kinesis stream: `ingest-delta-raw-search`

---

### delta-reader

**Type**: ECS Fargate Service (continuous)
**Trigger**: KCL consumer on `ingest-delta-raw-search`
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Consumes Delta records from `ingest-delta-raw-search` via KCL and fans out to the DemandSummary, SearchWithItineraries, and IngestCache plugins. Outbound data (replies) are held in a Redis cluster until paired with their inbound.

**Input**:
- Kinesis stream: `ingest-delta-raw-search`
- Redis cluster: `ingest-cluster-delta-outbounds.…clustercfg.use1.cache.amazonaws.com` (outbound pairing)
- DynamoDB KCL lease table: `FargateIngestDelta`

**Output**:
- S3 (DemandSummary plugin): `s3://dataset-ingest-{env}/delta/demand-summary/v1/…`
- S3 (SearchWithItineraries plugin): `s3://dataset-ingest-{env}/delta/search-with-itineraries/v1/…`
- Kinesis stream (IngestCache plugin): `ingest-cache`

---

### delta-monitor

**Type**: ECS Fargate Scheduled Task
**Trigger**: EventBridge cron — every 5 minutes
**Compute**: 1024 CPU units, 1024 MB RAM

**What it does**: Same monitoring and auto-scaling logic as `atlas-monitor`, applied to the Delta reader service and its KCL lease table (`FargateIngestDelta`).

---

### southwest-fetcher

**Type**: ECS Fargate Service (continuous)
**Trigger**: Continuous polling; reads from SFTP-backed S3 bucket
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Reads Southwest fare files that arrive in the AWS Transfer Family SFTP bucket (`s3://3v-sftp-inbound-{env}/southwest/`), parses them, and publishes records to the Southwest Kinesis stream. Thread pools: 1 process, 8 combined parse+publish. Kinesis publisher tuned for 16 shards, 2 048-byte mean record size.

**Input**:
- S3 (SFTP inbound): `s3://3v-sftp-inbound-{env}/southwest/` (AWS Transfer Family)

**Output**:
- S3 archive: `s3://3v-ingest-inbound-southwest-archive-{env}/`
- Kinesis stream: `ingest-southwest-raw-search`

---

### southwest-reader

**Type**: ECS Fargate Service (continuous)
**Trigger**: KCL consumer on `ingest-southwest-raw-search`
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Consumes Southwest records from `ingest-southwest-raw-search` via KCL and fans out to SearchWithItineraries and IngestCache plugins. (DemandSummary plugin is not configured for Southwest.)

**Input**:
- Kinesis stream: `ingest-southwest-raw-search` (32 shards in prod)
- DynamoDB KCL lease table: `FargateIngestSouthwest`

**Output**:
- S3 (SearchWithItineraries plugin): `s3://dataset-ingest-{env}/southwest/search-with-itineraries/v1/…`
- Kinesis stream (IngestCache plugin): `ingest-cache`

---

### southwest-monitor

**Type**: ECS Fargate Scheduled Task
**Trigger**: EventBridge cron — every 5 minutes
**Compute**: 1024 CPU units, 1024 MB RAM

**What it does**: Same monitoring and auto-scaling pattern as `atlas-monitor`, applied to the Southwest reader and KCL lease table `FargateIngestSouthwest`.

---

### availability-fetcher

**Type**: ECS Fargate Service (continuous)
**Trigger**: Continuous SFTP polling
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Connects to Travelport's external SFTP server (`xfer.prod.travelport.com:22`) using the ATPCO credential from Secrets Manager (`travelport/sftp/availability/ATPCO`). Downloads multiple file variants — AVS (general availability), CN-HK (China/HK seamless), OTH (other seamless), and US seamless — each optionally with AA and/or DL airline attribute variants (16 total type flags). Downloaded files are staged to an S3 inbound bucket and archived after processing. Thread pools: 24 fetcher, 12 reader.

**Input**:
- SFTP: `xfer.prod.travelport.com:22` (Travelport external server)
- Secrets Manager: `travelport/sftp/availability/ATPCO`

**Output**:
- S3 staging: `s3://3v-ingest-inbound-travelport-availability-{env}/`
- S3 archive: `s3://3v-ingest-inbound-travelport-availability-archive-{env}/`

---

### availability-reader

**Type**: ECS Fargate Service (continuous)
**Trigger**: Polls the availability S3 inbound bucket
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Reads staged availability files from S3 and processes them into the availability cache. Persists results to a Redis cluster (batch size 10 000) and optionally to S3. Cache content filtering is driven by a CSV file fetched from S3 (`config-server-{env}/default/cache-content.csv`).

**Input**:
- S3: `s3://3v-ingest-inbound-travelport-availability-{env}/`
- S3 config: `s3://config-server-{env}/default/cache-content.csv`

**Output**:
- Redis cluster: `travelport-availability.….clustercfg.use1.cache.amazonaws.com:6379`
- S3 (optional persistence): `s3://3v-ingest-inbound-travelport-availability-{env}/` (processed records)

---

### estream-reader

**Type**: ECS Fargate Service (continuous)
**Trigger**: KCL consumer on `ingest-estream-raw-search`
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Consumes eStream records from `ingest-estream-raw-search` (data published upstream by the eStream provider, no in-repo fetcher). Fans out to DemandSummary, SearchWithItineraries, and IngestCache plugins. Max concurrent reader tasks: 24.

**Input**:
- Kinesis stream: `ingest-estream-raw-search` (8 shards)
- DynamoDB KCL lease table: `FargateIngestEstream`

**Output**:
- S3 (DemandSummary plugin): `s3://dataset-ingest-{env}/estream/demand-summary/v1/…`
- S3 (SearchWithItineraries plugin): `s3://dataset-ingest-{env}/estream/search-with-itineraries/v1/…`
- Kinesis stream (IngestCache plugin): `ingest-cache`

---

### estream-monitor

**Type**: ECS Fargate Scheduled Task
**Trigger**: EventBridge cron — every 5 minutes

**What it does**: Same monitoring and auto-scaling logic as `atlas-monitor`, applied to the eStream reader service and KCL lease table `FargateIngestEstream`.

---

### priceeye-reader

**Type**: ECS Fargate Service (continuous)
**Trigger**: KCL consumer on `ingest-priceeye-raw-search`
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Consumes PriceEye records from `ingest-priceeye-raw-search` (data produced upstream by PriceEye, no in-repo fetcher). Only the SearchWithItineraries plugin is configured (no DemandSummary for PriceEye). Auto-stops after 120 minutes of idle (no incoming records).

**Input**:
- Kinesis stream: `ingest-priceeye-raw-search` (8 shards)
- DynamoDB KCL lease table: `FargateIngestPriceeye`

**Output**:
- S3 (SearchWithItineraries plugin): `s3://dataset-ingest-{env}/priceeye/search-with-itineraries/v1/…`
- Kinesis stream (IngestCache plugin): `ingest-cache`

---

### priceeye-monitor

**Type**: ECS Fargate Scheduled Task
**Trigger**: EventBridge cron — every 5 minutes

**What it does**: Same monitoring and auto-scaling logic as `atlas-monitor`, applied to the PriceEye reader service and KCL lease table `FargateIngestPriceeye`.

---

### sstream-reader

**Type**: ECS Fargate Service (continuous)
**Trigger**: KCL consumer on `ingest-sstream-raw-search`
**Compute**: 1024 CPU units, 1280 MB RAM

**What it does**: Consumes SStream records from `ingest-sstream-raw-search` (no in-repo fetcher). Fans out to DemandSummary and SearchWithItineraries plugins; IngestCache plugin is conditionally disabled in production but enabled in dev. Max concurrent reader tasks: 12.

**Input**:
- Kinesis stream: `ingest-sstream-raw-search` (8 shards)
- DynamoDB KCL lease table: `FargateIngestSstream`

**Output**:
- S3 (DemandSummary plugin): `s3://dataset-ingest-{env}/sstream/demand-summary/v1/…`
- S3 (SearchWithItineraries plugin): `s3://dataset-ingest-{env}/sstream/search-with-itineraries/v1/…`
- Kinesis stream (IngestCache plugin, dev only): `ingest-cache`

---

### sstream-monitor

**Type**: ECS Fargate Scheduled Task
**Trigger**: EventBridge cron — every 5 minutes

**What it does**: Same monitoring and auto-scaling logic as `atlas-monitor`, applied to the SStream reader service and KCL lease table `FargateIngestSstream`.

---

## Plugin Reference

Three reader plugins are shared across all stream-based providers:

| Plugin | Output | Description |
|--------|--------|-------------|
| **DemandSummary** | S3 Avro (`demand-summary/`) | Computes aggregated demand metrics and marketing-carrier itinerary statistics from raw search records |
| **SearchWithItineraries** | S3 Avro (`search-with-itineraries/`) | Writes raw search records with full itinerary detail; applies legacy 2-character source-code normalization. 16 processing threads. |
| **IngestCache** | Kinesis `ingest-cache` | Filters records by PriceEye active input-request markets; only forwards ≤30-day LOS, 0- or 1-stop itineraries. Refreshes market metadata every 10 min. |

Plugin threads run inside the reader ECS task. Each plugin has its own async queue and executor pool.

---

## Shared AWS Resources

### Kinesis Streams

| Stream | Provider | Shards (prod) |
|--------|----------|---------------|
| `ingest-atlas-raw-search` | Atlas | 32 |
| `ingest-delta-raw-search` | Delta | 8 |
| `ingest-estream-raw-search` | Estream | 8 |
| `ingest-priceeye-raw-search` | PriceEye | 8 |
| `ingest-southwest-raw-search` | Southwest | 32 |
| `ingest-sstream-raw-search` | Sstream | 8 |
| `ingest-cache` | All (IngestCache plugin) | — |

### DynamoDB Tables (KCL lease tables)

| Table | Provider |
|-------|----------|
| `FargateIngestAtlas` | Atlas KCL coordination |
| `FargateIngestDelta` | Delta KCL coordination |
| `FargateIngestEstream` | Estream KCL coordination |
| `FargateIngestPriceeye` | PriceEye KCL coordination |
| `FargateIngestSouthwest` | Southwest KCL coordination |
| `FargateIngestSstream` | Sstream KCL coordination |

### S3 Buckets

| Bucket | Purpose |
|--------|---------|
| `dataset-ingest-{env}` | All processed Avro output (demand-summary, search-with-itineraries) |
| `3v-ingest-inbound-delta-{env}` | Delta local staging |
| `3v-ingest-inbound-delta-archive-{env}` | Delta processed archive |
| `3v-sftp-inbound-{env}` | Southwest SFTP inbound (AWS Transfer Family) |
| `3v-ingest-inbound-southwest-archive-{env}` | Southwest processed archive |
| `3v-ingest-inbound-travelport-availability-{env}` | Availability SFTP staging |
| `3v-ingest-inbound-travelport-availability-archive-{env}` | Availability archive |
| `config-server-{env}` | Configuration: `default/cache-content.csv` |

### Redis Clusters

| Cluster | Purpose |
|---------|---------|
| `ingest-cluster-delta-outbounds.….clustercfg.use1.cache.amazonaws.com` | Delta outbound/inbound pairing |
| `travelport-availability.….clustercfg.use1.cache.amazonaws.com` | Availability data cache |

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| ECS Fargate Services (continuous) | 9 (3× fetchers + 6× readers) |
| ECS Fargate Scheduled Tasks (monitors) | 7 |
| Kinesis Streams | 7 (6 provider + 1 cache) |
| DynamoDB Tables (KCL) | 6 |
| S3 Buckets | 9+ |
| Redis Clusters | 2 |
| Step Functions | 0 |
| EventBridge Rules | 7 (one per monitor) |

---

## Key Operational Notes

- **Graceful shutdown**: All ECS tasks handle SIGTERM with a 120 s flush window to avoid data loss mid-multipart-upload.
- **Auto-scaling**: Each monitor scales readers between 0 and a provider-specific max (e.g., 4 for Atlas, 24 for Estream). Services start at `DesiredCount=0` and are scaled up by the monitor or manually.
- **KCL time-warp recovery**: If the monitor detects a KCL lease-table anomaly (max age jumps > 4×), it automatically stops the service, purges DynamoDB lease entries, and restarts — no manual intervention needed.
- **Dynamic read rate**: Readers auto-tune KCL throughput between 100–1 000 records/sec based on queue depth.
- **Build**: Maven multi-module (`pom.xml` at repo root), Java 17, deploys Docker images to AWS ECR, artifacts to AWS CodeArtifact.
- **Environments**: `3v-dev` / `3VDEV` (dev), `3VGOLD` (gold), `3v-prd` / `3VPROD` (production). Environment is injected as a property macro at deploy time.
- **Cross-account access**: Delta fetcher assumes `arn:aws:iam::494528744408:role/…` to read from Delta's S3 bucket in their AWS account.
