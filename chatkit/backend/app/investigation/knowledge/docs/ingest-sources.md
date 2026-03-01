# ingest-sources

> Ingests real-time flight search data from two GDS providers — Sabre (SStream) and Travelport (eStream) — and publishes normalized search records to AWS Kinesis for downstream analytics processing.

> **Current branch**: `develop` (production runs off `master`; this document reflects `develop`, which may include in-flight changes not yet in production)

---

## Architecture Overview

```
┌─────────────────────────┐        ┌──────────────────────────┐
│  Sabre GDS (external)   │        │  Travelport GDS (external)│
│  HTTP POST gzip JSON    │        │  HTTP POST gzip CSV       │
└───────────┬─────────────┘        └───────────┬──────────────┘
            │                                  │
            ▼                                  ▼
┌────────────────────────┐        ┌──────────────────────────────┐
│  source-sabre-sstream  │        │  source-travelport-estream   │
│  (Elastic Beanstalk)   │        │  (Elastic Beanstalk)         │
│  /sink.html            │        │  /sink.html                  │
│  32 publish threads    │        │  32 publish threads          │
└──────────┬─────────────┘        └─────────────┬────────────────┘
           │                                    │
           ▼                                    ▼
  Kinesis: sstream-raw-search        Kinesis: estream-raw-search
  (48 shards, prod)                  (64 shards, prod)
           │                                    │
           └──────────────┬─────────────────────┘
                          ▼
               [Downstream Analytics]
                (ingest-cache, etc.)

 ─ ─ ─ ─ ─ ─ TEST / REPLAY PATH ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

┌─────────────────────────────────────────────────────────┐
│  source-test (Elastic Beanstalk)                        │
│  /sabre/sink.html + /estreaming/sink.html               │
│  ← receives feeds from BOTH Sabre & Travelport         │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
          S3: s3-atp-3victors-3vprod-use1-website-ingestv2-test
          sabre/YYYY/MM/DD/HH/   travelport/YYYY/MM/DD/HH/
                             │
                             │ (read back for replay)
                             ▼
┌────────────────────────────────────────────────┐
│  test-data-replay (ECS Fargate, scheduled)     │
│  cron: every hour at :10                       │
│  ─ reads from S3 test bucket                   │
│  ─ HTTP POSTs to source-sabre-sstream          │
│  ─ HTTP POSTs to source-travelport-estream     │
└────────────────────────────────────────────────┘
```

---

## Deployment Model

This repo does **not** use the standard CloudFormation nested-stack pattern. Instead:

| Component | Compute type | Deploy mechanism |
|-----------|-------------|-----------------|
| source-sabre-sstream | **Elastic Beanstalk** (WAR) | Manual EB deploy |
| source-travelport-estream | **Elastic Beanstalk** (WAR) | Manual EB deploy |
| source-test | **Elastic Beanstalk** (WAR) | Manual EB deploy |
| test-data-replay | **ECS Fargate scheduled task** | `deploy/commonfiles/scheduled-task.yaml` |

All four are built with Maven and deployed from the build box (see `deploy/README.txt`).

---

## Components

_(Shared libraries first, then active services in logical order.)_

---

### common-sabre _(shared library)_

**Type**: Maven JAR (internal dependency, not deployed standalone)

**What it does**: Format-translation library for Sabre SStream data. Provides GZIP decompression, JSON deserialization, and normalization of Sabre search records into the canonical internal `SearchWithItineraries` Avro object. Supports two protocol versions:

- **Version 1.0** — legacy format; direct JSON deserialization via `SearchWithItinerariesGsonSerde`
- **Version 2.0** — enhanced format (added 2019-08); explicit query/passenger separation, multi-passenger support (added 2020-11), currency conversion to USD, header extraction (GDS code, PCC, restriction flag), and private-fare filtering

**Key classes**:
| Class | Purpose |
|-------|---------|
| `InboundSearchInterpreter` | Factory — instantiates correct parser by `SabreFormat` enum |
| `SabreSearchParserVersion1_0` | Legacy format parser |
| `SabreSearchParserVersion2_0` | Current format parser with full enrichment pipeline |
| `SabreParseResults` | Result object: parsed `SearchWithItineraries`, headers, multiPassenger flag |

**Output headers produced** (injected into downstream Kinesis records):
```
X-Sabre-GDS, X-Sabre-PCC, X-Sabre-Restricted
3v-Passenger-Count, 3v-Passenger-Types
```

---

### common-travelport _(shared library)_

**Type**: Maven JAR (internal dependency, not deployed standalone)

**What it does**: Format-translation library for Travelport eStream data. Parses GZIP-compressed delimited-text files from Travelport into canonical `SearchWithItineraries` objects. Supports three protocol versions:

- **Version 1.1** — pipe/comma-delimited (43 tokens); single ADT passenger assumed; up to 4 legs per direction
- **Version 1.4** — CSV; fixed field counts (40 fields OW / 73 fields RT); enhanced metadata (brand IDs, availability indicators); still single-passenger
- **Version URB-1** — Universal Record Builder (added 2020-12); CSV (45 OW / 77 RT); full multi-passenger-type support; codeshare info; fare/tax breakdown by PTC; controlled by `x-urb-features` header flags

**Key classes**:
| Class | Purpose |
|-------|---------|
| `InboundSearchInterpreter` | Factory — instantiates parser pair (search + itinerary) by `EstreamFormat` enum |
| `EstreamSearchParserVersion_URB_1` | Current multi-passenger parser |
| `EstreamParseResults` | Result: `SearchWithItineraries`, `EstreamSearchParameters`, reason-count error map |
| `TravelportItineraryURB_1` | 45/77-field data model for URB-1 format |

---

### source-sabre-sstream

**Type**: Elastic Beanstalk WAR (Java 17 / Struts2 / Jakarta EE)
**Trigger**: HTTP POST push from Sabre GDS (external, real-time, continuous)
**Compute**: 32 Kinesis publish threads; Tomcat tuned for high concurrency (8192 max threads, 1M accept queue)

**What it does**: Acts as an HTTP sink endpoint that receives GZIP-encoded JSON flight search data pushed by Sabre's SStream feed. Detects the protocol version (1.0 vs 2.0) from the JSON structure, decompresses and parses records via `common-sabre`, applies currency normalization and geo enrichment from Aurora metadata, and publishes each `SearchWithItineraries` record to Kinesis. Filters out searches with only private fares.

**Input**:
- HTTP POST to `/sink.html` — gzip JSON body from Sabre
- Aurora Metadata DB — city-to-country mapping, currency conversion tables

**Output**:
- Kinesis stream: `kinesis-atp-3victors-3vprod-use1-sstream-raw-search` (prod, 48 shards)
- Kinesis stream: `ingest-sstream-raw-search` (dev, 1 shard)

**Health check**: `GET /health.html` → always HTTP 200

**Key configuration** (resolved per environment from `configuration/<ENV>/macros.properties`):

| Property | Dev | Prod |
|----------|-----|------|
| `sstream.kinesis.streamName` | `ingest-sstream-raw-search` | `kinesis-atp-3victors-3vprod-use1-sstream-raw-search` |
| `sstream.kinesis.shardCount` | 1 | 48 |
| `sstream.kinesis.publishThreads` | 4 | 32 |
| `sstream.kinesis.recordSize` | 10240 | 10240 |

**Elastic Beanstalk endpoints**:
- Dev: `ingest-source-sabre-sstream-env.eba-vyxi7nxh.us-east-1.elasticbeanstalk.com`
- Prod: `ingest-source-sabre-sstream-env.eba-fqz3jcah.us-east-1.elasticbeanstalk.com`

---

### source-travelport-estream

**Type**: Elastic Beanstalk WAR (Java 17 / Struts2 / Jakarta EE)
**Trigger**: HTTP POST push from Travelport eStream API (external, real-time, continuous)
**Compute**: 32 Kinesis publish threads; Tomcat high-concurrency config

**What it does**: Acts as an HTTP sink endpoint for Travelport's eStream GDS feed. Receives GZIP-compressed CSV data, extracts search parameters from custom HTTP headers (`X-Original-Request`, `X-Point-Of-Sale`, `X-Transaction-ID`, `x-urb-features`, etc.), parses via `common-travelport` (supporting v1.1, v1.4, and URB-1 formats), applies currency normalization and geo enrichment, and publishes each record to Kinesis. Limits itineraries to 12,000 per search record to stay within Kinesis's 1 MB record limit.

**Input**:
- HTTP POST to `/sink.html` — gzip CSV body from Travelport
- Aurora Metadata DB — city-to-country mapping, currency conversion, timezone data

**Output**:
- Kinesis stream: `kinesis-atp-3victors-3vprod-use1-ingest-estream-raw-search` (prod, 64 shards)
- Kinesis stream: `ingest-estream-raw-search` (dev, 1 shard)

**Health check**: `GET /health.html` → always HTTP 200

**Key configuration** (resolved per environment):

| Property | Dev | Prod |
|----------|-----|------|
| `estream.kinesis.streamName` | `ingest-estream-raw-search` | `kinesis-atp-3victors-3vprod-use1-ingest-estream-raw-search` |
| `estream.kinesis.shardCount` | 1 | 64 |
| `estream.kinesis.publishThreads` | 4 | 32 |
| `estream.toss.maxItinerariesPerSearch` | 12000 | 12000 |
| `estream.toss.otherPointsOfSale` | false | false |

**Elastic Beanstalk endpoints**:
- Dev: `ingest-source-travelport-estream-env.eba-sxuims33.us-east-1.elasticbeanstalk.com`
- Prod: `ingest-source-travelport-estream-env.eba-9pqhxepp.us-east-1.elasticbeanstalk.com`

---

### source-test

**Type**: Elastic Beanstalk WAR (Java 17 / Struts2 / Jakarta EE)
**Trigger**: HTTP POST push from both Sabre and Travelport (same feeds as production, routed separately)
**Compute**: 24 Sabre threads + 8 Travelport threads

**What it does**: A test data capture application that receives live search feeds from both Sabre and Travelport and archives them to S3. It does **not** publish to Kinesis in normal operation (Kinesis publishing is disabled via code flags). The archived data is used for integration testing and can be replayed via `test-data-replay`. Includes 60-second sliding window deduplication to prevent duplicate S3 writes. Missing-header and duplicate requests are written to separate S3 subfolders.

**HTTP endpoints**:
- `/sabre/sink.html` — receives Sabre SStream data (24-thread pool)
- `/estreaming/sink.html` — receives Travelport eStream data (8-thread pool)
- `/sabre/health.html` and `/estreaming/health.html` — health checks

**Input**:
- HTTP POST from Sabre (same gzip JSON format as source-sabre-sstream)
- HTTP POST from Travelport (same gzip CSV format as source-travelport-estream)

**Output**:
- S3 bucket: `s3-atp-3victors-3vprod-use1-website-ingestv2-test`
  - `sabre/YYYY/MM/DD/HH/<filename>` — Sabre raw data (line 1: headers, line 2+: JSON)
  - `travelport/YYYY/MM/DD/HH/<transaction-id>.raw` — Travelport raw data
  - `travelport/YYYY/MM/DD/HH/missing-header/<uuid>` — requests missing required headers
  - `travelport/YYYY/MM/DD/HH/duplicate/<transaction-id>.raw` — deduplicated requests

**Runtime flags** (controlled via system properties / code constants):

| Flag | State | Description |
|------|-------|-------------|
| `WRITE_DATA_TO_S3` | **true** | S3 archival enabled |
| `PARSE_DATA` | false | Parsing disabled |
| `publishEstreamToDevEnabled` | false | Kinesis publish disabled |
| `publishSstreamToDevEnabled` | false | Kinesis publish disabled |

---

### test-data-replay

**Type**: ECS Fargate Scheduled Task (Java 11 executable JAR)
**Trigger**: EventBridge cron — `cron(10 * * * ? *)` → every hour at :10 (UTC)
**Compute**: 2048 MB memory, 1024 CPU (1 vCPU), ARM64
**CloudFormation**: `deploy/commonfiles/scheduled-task.yaml`
**ECS Cluster**: `ecs-atp-3victors-3vprod-use1-ingest` (prod)

**What it does**: Reads the most recent hour's test data from the S3 test bucket (both Sabre and Travelport partitions) and replays it as HTTP POSTs to the live Elastic Beanstalk ingest endpoints. Runs both EStream and SStream replays concurrently with 9 threads each, reconstructing the original gzip payload and HTTP headers from the stored `.raw` files. This generates realistic data volume in dev/staging without depending on live GDS connectivity.

**Input**:
- S3: `s3-atp-3victors-3vprod-use1-website-ingestv2-test/travelport/YYYY/MM/DD/HH/*.raw`
- S3: `s3-atp-3victors-3vprod-use1-website-ingestv2-test/sabre/YYYY/MM/DD/HH/*`
- File format: line 1 = comma-separated `key:value` HTTP headers, line 2+ = raw gzip search data

**Output**:
- HTTP POST to Sabre sink: `<env>.elasticbeanstalk.com/sink.html` (reconstructed gzip JSON)
- HTTP POST to Travelport sink: `<env>.elasticbeanstalk.com/sink.html` (reconstructed gzip CSV)

**Main entry point**: `com.threevictors.ingest.sources.common.TestDataReplay`

---

## Infrastructure Summary

| Resource | Count | Details |
|----------|-------|---------|
| Elastic Beanstalk Apps | 3 | source-sabre-sstream, source-travelport-estream, source-test |
| ECS Fargate Scheduled Tasks | 1 | test-data-replay (hourly at :10) |
| Kinesis Streams (prod) | 2 | sstream-raw-search (48 shards), estream-raw-search (64 shards) |
| S3 Buckets | 1 | `s3-atp-3victors-3vprod-use1-website-ingestv2-test` (test data) |
| Aurora Databases | 1 | Metadata DB (city/country mapping, currency conversion, timezones) |
| ECS Clusters | 1 | `ecs-atp-3victors-3vprod-use1-ingest` |

---

## Environment Configuration

Configuration is environment-specific. Each environment has a `configuration/<ENV>/macros.properties` file that resolves placeholders in the template `configuration/*.properties` files.

| Environment | Config dir | ECS cluster |
|-------------|-----------|-------------|
| 3VDEV | `configuration/3v-dev/` | `ecs-atp-3victors-3vdev-use1-ingest` |
| 3VGOLD | `configuration/3VGOLD/` | `ecs-atp-3victors-3vgold-use1-ingest` |
| 3VPROD | `configuration/3VPROD/` | `ecs-atp-3victors-3vprod-use1-ingest` |

---

## Build & Deploy

The repo builds with Maven from the build box (over VPN via SSM session). Elastic Beanstalk components are deployed **manually** (not via the deploy script). Only `test-data-replay` is deployed as a CloudFormation stack.

```
# Build all modules
mvn -U -T1C clean install

# Deploy (ECS scheduled task only)
cd deploy
build.sh all <version> true    # 'true' = snapshot build
release.sh <version>
deploy.sh 3vdev
```

See `deploy/README.txt` for full instructions including CodeArtifact token setup.

---

## Key Data Flows

### Normal production path
```
Sabre GDS  →  HTTP POST (gzip JSON)  →  source-sabre-sstream (EB)
                                              ↓
                                    Kinesis: sstream-raw-search
                                              ↓
                                      [ingest-cache / analytics]

Travelport  →  HTTP POST (gzip CSV)  →  source-travelport-estream (EB)
                                               ↓
                                    Kinesis: estream-raw-search
                                               ↓
                                      [ingest-cache / analytics]
```

### Test data capture & replay
```
Sabre / Travelport  →  source-test (EB)  →  S3 test bucket
                                                  ↑ (read)
                                           test-data-replay (ECS, hourly)
                                                  ↓ (HTTP POST)
                                    source-sabre-sstream / source-travelport-estream
```
