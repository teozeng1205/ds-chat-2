# partition-creator

> Automatically creates and manages AWS Glue table partitions by reacting to S3 Object Created events, driven by pattern rules stored in an Aurora MySQL metadata database.

> **Production branch**: `master` (this document reflects the `develop` branch — the documented state may differ from production)

---

## Architecture Overview

```
[EventBridge cron: every 1 hour]
      │
      ▼
[partition-rule-updater Lambda]
      │  reads distinct S3 buckets from Aurora
      │
      └──► updates EventBridge S3 rule with current bucket list
                         │
                         ▼
          [EventBridge rule: S3 Object Created events
           on configured buckets]
                         │
                         ▼
          [partition-creator Lambda]
                         │  matches S3 key against patterns in Aurora
                         │  extracts partition values from key path
                         │
                         ├──► Glue API: creates partition on Glue table
                         │
                         └──► (optional) EventBridge "Partition Created" event
                                  on "data-pipeline" bus
                                  (downstream pipelines consume)

[Manual / ad-hoc backfill]
      │
      ▼
[partition-filler (standalone JAR)]
      │  lists S3 directories recursively (20 threads)
      └──► same pattern-matching → Glue partition creation
```

---

## Orchestration

There is no Step Function in this repo. The two components are independently triggered:

### Scheduled Rule: partition-rule-updater
- **Trigger**: EventBridge scheduled rule — every 1 hour (`rate(1 hour)`)
- **Purpose**: Keeps the S3 event rule's bucket filter up to date dynamically — no manual re-deployment needed when buckets are added or removed.
- **CloudFormation**: `source/deploy/commonfiles/partition-rule-updater.yaml`

### Event-Driven Rule: partition-creator
- **Trigger**: EventBridge rule filtering on `aws.s3` / `Object Created` events for a dynamic list of S3 buckets (managed by partition-rule-updater above)
- **CloudFormation**: `source/deploy/commonfiles/partitioncreator.yaml`

---

## Components

_(Ordered by pipeline sequence: rule updater runs first to keep the trigger current; creator reacts to S3 events.)_

---

### partition-rule-updater

**Type**: Lambda Function (container image, arm64)
**Trigger**: EventBridge scheduled rule — every 1 hour
**Compute**: 624 MB memory, 60 s timeout

**What it does**: Runs hourly to keep the EventBridge S3 event rule synchronized with the current set of monitored buckets. It reads the distinct list of S3 bucket names from the Aurora metadata database (`partition_creator.partition_details`) and calls the EventBridge `PutRule` API to overwrite the event pattern of the partition-creator rule with the latest bucket list. This means that adding a new dataset to Aurora automatically causes the rule to begin monitoring that bucket within an hour, without any infrastructure re-deployment.

**Input**:
- Aurora MySQL: `partition_creator.partition_details` — reads distinct `bucket` values
- Properties file: `PEPartitionRuleUpdaterLambda.properties` — rule name and event bus name

**Output**:
- EventBridge rule (updated in-place): event pattern listing all configured S3 buckets

**Event pattern written**:
```json
{
  "source": ["aws.s3"],
  "detail-type": ["Object Created"],
  "detail": {
    "bucket": {
      "name": ["bucket-a", "bucket-b", "..."]
    }
  }
}
```

**Source**: `source/partition-rule-updater/src/main/java/.../PEPartitionRuleUpdaterLambda.java`

---

### partition-creator

**Type**: Lambda Function (container image, arm64)
**Trigger**: EventBridge rule — S3 `Object Created` events on configured buckets
**Compute**: 624 MB memory, 60 s timeout, max retries: 0, max event age: 60 s

**What it does**: Reacts to every S3 object upload event. It loads all partition configuration rules from Aurora MySQL on cold start, then matches the uploaded object's S3 key against those rules using `{macro}` placeholder patterns. When a match is found, it extracts the partition column values from the key path, constructs the Glue partition location, and calls the Glue API to add (or skip if already present) the partition to the target Glue table. If the matched rule has `emit_event = 1`, it also publishes a "Partition Created" event to the `data-pipeline` EventBridge bus, allowing downstream pipelines to react.

**Input**:
- EventBridge event payload: S3 bucket name + object key (from `Object Created` notification)
- Aurora MySQL: `partition_creator.partition_details` — full partition rule configuration (loaded at cold start)

**Output**:
- Glue catalog: new partition added to `{destination_database}.{destination_table}`
- EventBridge event _(optional, when `emit_event = 1`)_: `"Partition Created"` on bus `"data-pipeline"`, source `"threevictors.partitioncreator"`

**Pattern matching logic**:

Patterns use `{macro}` placeholders in S3 key templates. Special handling:
- `{sales_year}`, `{sales_month}`, `{sales_day}` are concatenated into a single `sales_date` partition value (format `YYYYMMDD`).
- Other date-derivative columns (`feed_date`, `observation_date`) are auto-populated from `sales_date`.
- `{hour}` is validated as an integer.

| Pattern | S3 key | Extracted partitions |
|---------|--------|----------------------|
| `v1/{sales_year}/{sales_month}/{sales_day}` | `v1/2024/12/31/file.parquet` | `sales_date=20241231` |
| `v1/{customer}/{sales_year}/{sales_month}/{sales_day}/{hour}` | `v1/AA/2024/12/31/22/f.parquet` | `customer=AA, sales_date=20241231, hour=22` |

**Source**: `source/partition-creator/src/main/java/.../PEPartitionCreatorLambda.java`

---

### partition-filler _(batch / manual)_

**Type**: Standalone executable JAR (not deployed as Lambda or ECS; run manually for backfills)
**Trigger**: Manual — invoked directly via `java -jar` for ad-hoc backfills
**Compute**: 20 fixed threads, up to 3-hour run time

**What it does**: Backfills Glue partitions for existing S3 data that was written before the partition-creator Lambda existed or was configured. Given a partition rule (bucket + preamble + pattern), it recursively lists all S3 subdirectories under the configured prefix up to the depth implied by the pattern, applies the same `buildPartitionInfo()` matching logic used by the event-driven Lambda, and creates any missing Glue partitions in parallel across 20 threads.

**Input**:
- Aurora MySQL: `partition_creator.partition_details` — reads the applicable rule(s)
- S3: scans all subdirectories under `s3://{bucket}/{preamble}/` up to pattern depth

**Output**:
- Glue catalog: missing historical partitions created on the target table

**Source**: `source/partition-filler/src/main/java/.../PEPartitionFiller.java`

---

## Metadata Database

Partition rules are stored in Aurora MySQL and drive all three components.

**Database**: `partition_creator`
**Table**: `partition_details`
**Schema file**: `docs/aurora-master.sql`

| Column | Type | Description |
|--------|------|-------------|
| `bucket` | VARCHAR(128) PK | S3 bucket name |
| `preamble` | VARCHAR(256) | Directory prefix before the pattern begins |
| `pattern` | VARCHAR(256) PK | Key template with `{macro}` placeholders |
| `partition_pattern` | VARCHAR(256) | S3 location template for the partition |
| `partition_order` | VARCHAR(256) | Comma-separated partition column names in order |
| `destination_database` | VARCHAR(256) | Target Glue catalog database |
| `destination_table` | VARCHAR(256) | Target Glue catalog table |
| `emit_event` | TINYINT | `1` = emit "Partition Created" EventBridge event |
| `last_updated` | TIMESTAMP | Auto-updated on row change |

The sample data in the SQL file covers 18 dataset families including flight summaries, Price Eye audit logs (cache, delivery, enrichment), AI provider archives, QL2 archives, and channel comparison datasets.

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| Lambda Functions | 2 (`partition-creator`, `partition-rule-updater`) |
| EventBridge Rules | 2 (S3 Object Created, hourly schedule) |
| CloudFormation Stacks | 2 |
| Aurora MySQL Databases | 1 (`partition_creator`) |
| Aurora MySQL Tables | 1 (`partition_details`) |
| Glue Databases / Tables | N (managed externally; this repo only adds partitions) |

---

## Build & Deployment

**Language**: Java 17 (compiled to Java 11 target)
**Build**: Maven multi-module (`pom.xml` at repo root)

**Modules**:
| Module | Artifact |
|--------|----------|
| `source/data` | Shared data models (`PartitionDetail`, `PartitionEventDetails`) |
| `source/dao` | `MetadataReader` — reads Aurora partition rules |
| `source/partition-creator` | Lambda handler JAR → container image |
| `source/partition-rule-updater` | Lambda handler JAR → container image |
| `source/partition-filler` | Standalone backfill JAR |

**Container base**: `amazoncorretto:17` (linux/arm64)
**Lambda entry points**:
- `PEPartitionCreatorLambda::handleRequest`
- `PEPartitionRuleUpdaterLambda::handleRequest`

**Key external dependencies**:
- AWS Lambda Java Runtime Interface
- AWS SDK v2 (EventBridge, Glue, S3)
- 3Victors common libraries (Glue, S3, Aurora, Config Reader, Notification Publisher)
- MySQL Connector 8.2

**Artifact registry**: AWS CodeArtifact (internal 3Victors repository)
