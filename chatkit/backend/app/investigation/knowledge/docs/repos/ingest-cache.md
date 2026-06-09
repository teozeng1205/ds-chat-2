# ingest-cache

> A real-time pipeline that consumes airline search events from Kinesis and writes them into partitioned Redis clusters, one per data source, so downstream services can read recent fare searches by market.

> **Note**: This document was generated from the `develop` branch. The `master` branch reflects what is currently running in production — verify any configuration values against master before treating them as production truth.

---

## Architecture Overview

```
Kinesis Stream: ingest-cache
        │
        ▼ (KCL — 32 shards, 16 CPUs)
[writer — ECS Fargate, 32 GB / 16 vCPU]
        │
        ├── throttle: 1,000 reads/sec
        ├── inbound queue: 100,000 records
        └── writer thread pool: 64 threads
               │
               ├─► [SearchFilter] (per source)
               ├─► [SearchPartitioner]
               ├─► [KeyValueBuilder]
               │        │
               │    CacheKey: {POS}:{origin}:{dest}:{depart}:{return}
               │    RedisHashKey: {cabin}:{carrier}:{stops}:{refundable}
               │    CacheValue: Kryo → GZIP → Base64
               │
               ├─► Redis cluster (DELTA)
               ├─► Redis cluster (SOUTHWEST)
               ├─► Redis cluster (TRAVELPORT_ES)
               └─► Redis cluster (SABRE_STREAM)

[reader — library used by downstream services]
        ├── ReactiveSingleNodeCacheReader   (async / Lettuce)
        ├── ReactiveClusterCacheReader      (async / Lettuce)
        ├── SynchronousSingleNodeCacheReader (blocking / Lettuce)
        └── SynchronousClusterCacheReader   (blocking / Lettuce)
```

---

## Components

_(Ordered by pipeline stage.)_

---

### writer

**Type**: ECS Fargate Service (continuous, DesiredCount=1)
**Trigger**: Always-on; consumes from Kinesis via KCL
**Compute**: 32,768 MB memory, 16,384 CPU units (16 vCPU), 24 GB Java heap

**What it does**: The writer is the sole data-ingestion component. It runs a Kinesis Consumer Library (KCL) consumer against the `ingest-cache` Kinesis stream and funnels records through a 100,000-record blocking queue into a 64-thread writer pool. Each thread deserializes an Avro `RawSearch`, applies a per-source `SearchFilter`, partitions the search into sub-variants via `SearchPartitioner`, builds Redis key/value pairs via `KeyValueBuilder`, and writes them as Redis hashes to the appropriate source-specific cluster using Redisson. On startup it reads valid Travelport PCCs from the PriceEye Redshift database to pre-filter invalid Travelport requests. A SIGTERM handler (sent by ECS during deploys) drains the in-flight queue before shutdown.

**Input**:
- Kinesis stream: `ingest-cache` (dev) / `kinesis-atp-3victors-3vprod-use1-ingest-cache` (prod)
- Redshift (PriceEye): valid Travelport PCC list (read at startup and refreshed periodically)

**Output**:
- Redis cluster (DELTA): `redis://...ingest-cluster-delta.../6379`
- Redis cluster (SOUTHWEST): `redis://...ingest-cluster-southwest.../6379`
- Redis cluster (TRAVELPORT_ES): `redis://...ingest-cluster-travelport-es.../6379`
- Redis cluster (SABRE_STREAM): `redis://...ingest-cluster-tpes.../6379`

**Key operational parameters**:

| Parameter | Value |
|-----------|-------|
| KCL shard count | 32 |
| KCL CPU count | 16 |
| Max reads/sec (throttle) | 1,000 |
| Inbound queue capacity | 100,000 |
| Outbound queue limit | 100,000 |
| Writer thread count | 64 |
| Redis flush interval | 10 s |
| Cache expire check interval | 60 s |
| Periodic report interval | 60 s |
| Notify interval (lag alert) | 1,800 s (30 min) |
| Notify seconds-behind threshold | 1,800 s (30 min) |
| Memory emergency threshold | 22 GB |
| Memory emergency cooldown | 600 s |

**Thread model**:

| Thread | Count | Purpose |
|--------|-------|---------|
| KCL shard readers | per shard | Feed records into inbound queue |
| Queue reader | 1 | Drain inbound queue → writer pool |
| Cache writer pool | 64 | Deserialize, filter, key-build, write to Redis |
| Per-second ticker | 1 | Reset throttle counter each second |
| Periodic reporter | 1 | Log throughput & age stats every 60 s |
| Cache expire checker | 1 | Evict stale entries every 60 s |

---

### reader (library)

**Type**: Java library (not a deployed service; consumed by downstream services)
**Trigger**: Called synchronously or reactively by the consumer

**What it does**: The reader module provides four concrete implementations of a `CacheReader` interface that upstream services link against to query the Redis clusters. Callers can look up cache entries by `CacheKey` (market + dates), optionally scoped to one or more `RedisHashKey` patterns (cabin/carrier/stops/refundable). Wildcard glob patterns (`*`, `?`, `[]`) are supported on hash keys.

**Input**:
- Redis cluster(s): same clusters written by the writer

**Output**:
- `Map<RedisHashKey, CacheValue>` — deserialized search records keyed by attribute tuple

**Implementations**:

| Class | Mode | Redis topology |
|-------|------|----------------|
| `ReactiveSingleNodeCacheReader` | Async / Project Reactor | Single-node |
| `ReactiveClusterCacheReader` | Async / Project Reactor | Cluster mode |
| `SynchronousSingleNodeCacheReader` | Blocking | Single-node |
| `SynchronousClusterCacheReader` | Blocking | Cluster mode |

---

## Data Models

### Redis Key Format

```
CacheKey (Redis top-level key):
  {pointOfSaleCountryCode}:{originAirport}:{destinationAirport}:{departDate}:{returnDate}

Example:
  US:LAX:JFK:20240301:20240308
```

```
RedisHashKey (field within the Redis hash):
  {cabin}:{carrier}:{stopCount}:{refundable}

Example:
  E:UA:1:false
  *:*:0:*       ← wildcard scan
```

### Redis Hash Value

Each field value is a `CacheValue` serialized as:

```
CacheValue → Kryo → Output stream → GZIP → Base64 → String
```

The `CacheValue` contains:
- `itineraryCount` (int)
- `searchMap` — nested: `cabin → carrier → stopCount → refundable → CacheSearch`

`CacheSearch` holds:
- Timestamp, source, origin/destination airports, depart/return dates
- Carrier code, connection airports, cabin, maxStops, passengerCount, refundable
- Number of itineraries, raw itinerary list (prices/legs), additionalInfo, PCC

> **Kryo registration IDs must never be changed.** New classes may only be appended at the end of the registration list to maintain backward compatibility with data already in Redis.

---

## Ingest Sources

| Source | Redis Cluster (dev suffix) | Notes |
|--------|---------------------------|-------|
| `DELTA` | `ingest-cluster-delta` | Delta Air Lines |
| `SOUTHWEST` | `ingest-cluster-southwest` | Southwest Airlines |
| `TRAVELPORT_ES` | `ingest-cluster-travelport-es` | Travelport E-Stream GDS; PCC-filtered |
| `SABRE_STREAM` | `ingest-cluster-tpes` (prod shares travelport cluster) | Sabre GDS |

Each source has a dedicated `SearchFilter`, `RedisWriter`, and Redis connection. Removed sources: `TRAVELFUSION`.

---

## Maven Modules

| Module | Purpose |
|--------|---------|
| `source/common` | Singleton `IngestCacheConfiguration` — loads properties from S3, exposes all tunable params |
| `source/data` | `CacheKey`, `CacheValue`, `RedisHashKey`, `CacheSearch`, `CacheKeyValuePair` — the shared data model |
| `source/kryo` | `KryoSerde<T>` — Object ↔ Kryo/GZIP/Base64 string; registers all domain classes |
| `source/database` | Redshift data-access layer; reads valid Travelport PCCs from PriceEye |
| `source/reader` | `CacheReader` interface + reactive and synchronous implementations (Lettuce + Redisson) |
| `source/writer` | `CacheWriter` main class — KCL consumer, queue, thread pools, Redis writers |
| `source/dns-test` | Utility for testing DNS resolution of Redis cluster endpoints |

---

## Infrastructure

### CloudFormation Templates

| Template | Type | Notes |
|----------|------|-------|
| `deploy/commonfiles/service.yaml` | ECS Fargate **Service** | Continuous; DesiredCount=1, ARM64 |
| `deploy/commonfiles/scheduled-task.yaml` | ECS Fargate **Scheduled Task** | EventBridge cron; default every 10 min |

Both templates create:
- `AWS::ECS::TaskDefinition` (ARM64, Fargate)
- `AWS::IAM::Role` (ECR pull, S3 full, ECS RunTask, Kinesis full, SecretsManager read)
- `AWS::Logs::LogGroup` (`ingest/[StackName]`, 7-day retention)

The scheduled-task template additionally creates:
- `AWS::Events::Rule` targeting the ECS cluster with the given `ScheduleExpression`

### ECS Clusters

| Environment | Cluster |
|-------------|---------|
| 3vdev | `ecs-atp-3victors-3vdev-use1-ingest` |
| 3vgold | `ecs-atp-3victors-3vgold-use1-ingest` |
| 3vprod | `ecs-atp-3victors-3vprod-use1-ingest` |

Stack prefix: `{cluster}-cache`

### Infrastructure Summary

| Resource | Count |
|----------|-------|
| ECS Fargate Services | 1 (writer) |
| ECS Fargate Scheduled Tasks | 1 (template available) |
| Redis Clusters (ElastiCache) | 4 (one per source) |
| Kinesis Streams | 1 (ingest-cache) |
| CloudFormation Templates | 2 |
| Maven Modules | 7 |

---

## Configuration

Configuration is stored on S3 as `ingest-cache.properties` and loaded at startup by `IngestCacheConfiguration` (singleton). Environment-specific values are injected by macro substitution from `configuration/{ENV}/macros.properties`.

**Key properties**:

| Property | Purpose |
|----------|---------|
| `ingest.cache.streamName` | Kinesis stream to consume |
| `ingest.cache.kclApplicationName` | KCL checkpoint table name (DynamoDB) |
| `ingest.cache.kclCpuCount` / `kclShardCount` | KCL resource allocation |
| `ingest.cache.currentReadsAllowedPerSecond` | Throttle cap |
| `ingest.cache.inboundQueueSizeLimit` / `outboundQueueSizeLimit` | Queue back-pressure |
| `ingest.cache.memoryEmergencyThresholdInGB` | Triggers emergency pause |
| `ingest.cache.redisClusterUri{Source}` | Redis URI per source |
| `ingest.cache.writerThreadCount` | Parallel write threads |
| `ingest.cache.redisFlushIntervalInSeconds` | Async flush cadence |

---

## Observability

**CloudWatch Logs**: `ingest/[StackName]` — 7-day retention, `LOG_LEVEL=INFO`

**Periodic report (every 60 s)** includes:
- Records processed in period
- Result age: min / max / avg / stddev (seconds)
- Throttle event count
- Active shard IDs

**Alerts**:
- Lag alert: notifies when the consumer falls > 30 minutes behind the stream tip
- Memory alert: notifies and pauses when JVM heap exceeds 22 GB (cooldown 600 s)
- Queue-full warning: logged when the inbound queue cannot accept a record within 20 s

---

## Build & Deploy

```bash
# 1. Connect via GlobalProtect VPN
# 2. SSM into the build box
aws ssm start-session --target i-0a405720a7aafbf82

# 3. On the build box
cd ~/git/ingest-cache
export CODEARTIFACT_AUTH_TOKEN=$(aws codeartifact get-authorization-token \
  --domain atpco-3v --domain-owner 590183652635 \
  --region us-east-1 --query authorizationToken --output text)

git fetch && git pull
mvn -U -T1C clean install

# 4. Build and deploy
cd deploy
./build.sh all 0.60-SNAPSHOT true   # 'true' = use SNAPSHOT; omit for release
./release.sh 0.60-SNAPSHOT
./deploy.sh 3vdev                   # or 3vgold / 3vprod
```

**CodeArtifact domain**: `atpco-3v` (account `590183652635`)
**ECR repo prefix**: `3victors/ingest-cache`
