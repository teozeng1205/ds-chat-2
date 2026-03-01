# event-launcher

> An event-driven AWS Lambda service that listens for S3 object-creation events via EventBridge and dynamically launches downstream ECS Fargate tasks based on routing rules stored in an Aurora MySQL database.

> **Production branch**: `master` (this document reflects the master branch, which is what runs in production)

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────┐
                        │           Amazon EventBridge             │
                        │   (S3 Object Created notification)       │
                        │   source: aws.s3                         │
                        │   detail-type: Object Created            │
                        └──────────────────┬──────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │         event-launcher Lambda            │
                        │  (AWS Lambda — arm64, 624 MB, 60 s)     │
                        │                                          │
                        │  1. Receive EventBridgeNotification      │
                        │     { bucket, key }                      │
                        │                                          │
                        │  2. Query Aurora MySQL                   │
                        │     event_launcher.event_launcher_config │
                        │     for matching routing rules           │
                        │                                          │
                        │  3. Apply optional custom conditional    │
                        │     (e.g. key must end with _SUCCESS)    │
                        │                                          │
                        │  4. Parse S3 key against pattern,        │
                        │     extract named macro arguments        │
                        │     (sales_year, sales_month, etc.)      │
                        │                                          │
                        │  5. Call ECSTaskLauncher.runTask()       │
                        │     with extracted argument list         │
                        └──────────────────┬──────────────────────┘
                                           │
                       ECS RunTask API call│
                                           │
                  ┌────────────────────────▼────────────────────────┐
                  │               ECS Fargate Cluster                │
                  │   (ecs-3vprod-use1-price-eye  /                  │
                  │    ecs-3vgold-use1-price-eye  /                  │
                  │    ecs-3vdev-use1-price-eye)                     │
                  │                                                  │
                  │   Spawns one or more downstream Fargate tasks    │
                  │   identified by task_definition column,          │
                  │   passing extracted key parts as CLI arguments   │
                  └──────────────────────────────────────────────────┘
```

**Key data-flow summary:**

1. An upstream process writes a file to S3.
2. EventBridge delivers an `Object Created` notification to the event-launcher Lambda.
3. The Lambda looks up matching routing rules in Aurora MySQL (`event_launcher.event_launcher_config`).
4. For each matching rule it optionally applies a conditional (e.g. `SparkSuccessConditional` — only fire when the key ends with `_SUCCESS`).
5. If the conditional passes, it parses the S3 key against a pattern template, extracts named values (date parts, provider codes, etc.), and calls `ECSTaskLauncher.runTask()` to start the configured Fargate task with those values as arguments.

---

## Components

_(This repository contains a single deployable component — the event-launcher Lambda — supported by two library modules.)_

---

### event-launcher (Lambda)

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: EventBridge rule — `aws.s3` source, `Object Created` detail-type (S3 object-creation notification)
**Compute**: 624 MB memory, 60-second timeout
**Runtime**: Amazon Corretto 17 (Java 17), packaged as a Docker container image
**ECR repository**: `3victors/eventlauncher`
**CloudWatch Logs**: `eventlauncher/<stack-name>` (7-day retention)
**Monitoring**: CloudWatch alarm fires to `HighPriorityAlarm` SNS topic when Lambda duration equals or exceeds the 60-second timeout

**What it does**: The event-launcher Lambda is the central router of the pipeline. On each invocation it receives an `EventBridgeNotification` carrying the bucket name and object key of a newly created S3 object. It queries the Aurora MySQL configuration table (`event_launcher.event_launcher_config`) to find all routing rules whose `bucket` and `preamble` match the inbound S3 path. For each matching rule it optionally applies a pluggable conditional check (e.g. only trigger when the key ends with `_SUCCESS` to confirm a Spark job completed successfully). If all checks pass, the Lambda parses the S3 key against the rule's pattern template, extracts named macro values (such as `sales_year`, `sales_month`, `sales_day`, `sales_hour`, and arbitrary provider/feed identifiers), and starts the designated ECS Fargate task via `ECSTaskLauncher`, passing the extracted values as positional command-line arguments.

**Input**:
- EventBridge notification: `aws.s3` / `Object Created` event carrying `{ detail: { bucket: { name }, object: { key } } }`
- Aurora MySQL table: `event_launcher_config` from `event_launcher` database (read on every cold-start invocation)

**Output**:
- ECS `RunTask` API call — launches the Fargate task identified by `task_definition` with the extracted key arguments

**IAM permissions granted to the Lambda role**:
- `AmazonKinesisFullAccess`
- `AmazonKinesisFirehoseFullAccess`
- `AmazonS3FullAccess`
- `AmazonSQSFullAccess`
- `AWSLambdaVPCAccessExecutionRole`
- `AWSGlueServiceRole`
- `ecs:RunTask`, `ecs:StartTask`, `ecs:UpdateTaskProtection`, `ecs:ListTaskDefinitions`
- `iam:PassRole`
- `secretsmanager:GetSecretValue`
- `sts:AssumeRole`

**Network**: Deployed inside the PriceEye VPC (`FMSSecuritygroupApp` security group, subnets `SubnetApp0/1/2`)

**Source**: `source/event-launcher/src/main/java/com/threevictors/aws/eventlauncher/eventlauncher/EventLauncher.java`
**CloudFormation template**: `source/deploy/commonfiles/eventlauncher.yaml`

---

### Conditionals (plug-in routing filters)

Two concrete conditional implementations are shipped with the Lambda image. A rule row in `event_launcher_config` may reference one by its simple class name in the `custom_conditional` column.

| Class name | What it checks |
|---|---|
| `AnalyticsConditional` | S3 key must end with `_SUCCESS` |
| `SparkSuccessConditional` | S3 key must end with `_SUCCESS` |

Both implementations extend `AbstractConditional` and are dynamically instantiated via reflection at runtime:

```java
// AbstractConditional.java
public abstract boolean isApplicable(String bucket, String key);

// SparkSuccessConditional.java
public boolean isApplicable(String bucket, String key) {
    return key.endsWith("_SUCCESS");
}
```

**Source**: `source/event-launcher/src/main/java/com/threevictors/aws/eventlauncher/eventlauncher/conditionals/`

---

### Pattern matching and argument extraction

The `getArguments()` method in `EventLauncher` implements a lightweight template engine that maps S3 key segments to named macro variables. Patterns look like:

```
v1/{provider}/{sales_year}/{sales_month}/{sales_day}/_SUCCESS
```

The method splits both the pattern and the actual S3 key on `/`, matches literal segments exactly, and uses regex to extract values from templated segments (`{macro_name}`). Special handling is applied to date macros:

- `sales_year`, `sales_month`, `sales_day` are individually extracted and also concatenated into a single `sales_date` value (e.g. `20250312`).
- Any other `*_date` columns in the argument order string are also set to the same concatenated date.
- `sales_hour` must parse as an integer.
- Keys with path segments ending in `_$folder$` (Hadoop directory markers) are explicitly ignored.

The ordered list of extracted values is passed as positional arguments to `ECSTaskLauncher.runTask()`.

---

## Database Schema

### Database: `event_launcher`

Located in the Aurora MySQL cluster. Connection details are loaded from `EventLauncher.properties` via the internal `ConfigurationReader` library.

#### Table: `event_launcher_config`

This is the sole configuration store for the entire routing system. Each row defines one trigger rule.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `bucket` | `VARCHAR(128)` | NOT NULL | S3 bucket name to match (part of composite PK) |
| `preamble` | `VARCHAR(1024)` | NOT NULL | Path prefix used for fast pre-filter before full pattern match (part of composite PK) |
| `pattern` | `VARCHAR(1024)` | NOT NULL | Full S3 key pattern with `{macro_name}` placeholders |
| `task_definition` | `VARCHAR(512)` | NOT NULL | ECS task definition family name to launch (part of composite PK) |
| `arguments` | `VARCHAR(512)` | NOT NULL | Comma-separated ordered list of macro names to pass as CLI arguments to the task |
| `custom_conditional` | `VARCHAR(512)` | YES | Simple class name of an `AbstractConditional` to apply; `NULL` means no additional filter |
| `last_updated` | `TIMESTAMP` | YES | Auto-set on insert/update |

**Primary key**: `(bucket, preamble, task_definition)`

**DDL source**: `docs/mysql/aurora_master.sql`

**Example routing rule** (from unit test evidence):
```
bucket:          s3-atp-3victors-3vdev-use1-competitive-position
preamble:        v1/B6/
pattern:         v1/{provider}/{sales_year}/{sales_month}/{sales_day}/_SUCCESS
task_definition: <downstream-ecs-task-family>
arguments:       provider, sales_year, sales_month, sales_day, sales_date
custom_conditional: SparkSuccessConditional
```

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| Lambda Functions | 1 (`event-launcher`) |
| ECS Fargate Task Definitions | 0 (defined in downstream repos; this repo only launches them) |
| Step Functions | 0 |
| EventBridge Rules | 1 (S3 Object Created → Lambda; defined outside this repo) |
| Aurora MySQL Databases | 1 (`event_launcher`) |
| Aurora MySQL Tables | 1 (`event_launcher_config`) |
| Glue Databases | 0 |
| Glue Tables | 0 |
| CloudWatch Alarms | 1 (timeout alarm per deployment) |
| ECR Repositories | 1 (`3victors/eventlauncher`) |

---

## Build and Deployment

**Language / runtime**: Java 17 (Amazon Corretto), Maven multi-module project
**Current version**: `0.04-SNAPSHOT` (develop branch); `0.03` is the latest tagged release on `master`
**Maven modules**:

| Module | Artifact | Description |
|--------|----------|-------------|
| `source/event-launcher` | `threevictors-eventlauncher-eventlauncher` | Lambda handler and conditional logic |
| `source/dao` | `threevictors-eventlauncher-dao` | `MetadataReader` — queries Aurora for launcher configs |
| `source/data` | `threevictors-eventlauncher-data` | `LauncherConfig` POJO (Lombok `@Data`) |

**Docker images**:

| Dockerfile | Purpose |
|------------|---------|
| `source/deploy/dockerfiles/Dockerfile.lambda` | Production Lambda container image (arm64, Corretto 17, AWS Lambda RIC entrypoint) |
| `source/deploy/dockerfiles/Dockerfile` | Generic ECS Fargate task image template (not used by this Lambda itself) |

**Artifact registry**: `732267085676.dkr.ecr.us-east-1.amazonaws.com` (us-east-1)
**Package registry**: AWS CodeArtifact — `atpco-3v-590183652635.d.codeartifact.us-east-1.amazonaws.com/maven/3V-ATP/`

**Environment-specific ECS clusters**:

| Environment | Cluster name |
|-------------|-------------|
| `3vprod` | `ecs-3vprod-use1-price-eye` |
| `3vgold` | `ecs-3vgold-use1-price-eye` |
| `3vdev` | `ecs-3vdev-use1-price-eye` |

**CloudFormation templates**:
- `source/deploy/commonfiles/eventlauncher.yaml` — Lambda function, IAM role, CloudWatch log group, timeout alarm
- `source/deploy/commonfiles/task.yaml` — Generic ECS Fargate task definition template used by downstream tasks (referenced here for completeness)

---

## Key External Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `threevictors-priceeye-ecs-task-launcher` | 0.146 | Wraps ECS `RunTask` API; used by `EventLauncher` to start Fargate tasks |
| `threevictors-configuration-reader-heavy` | 0.59 | Reads `EventLauncher.properties` (cluster name, security group, DB connection) from Secrets Manager / S3 |
| `threevictors-aws-data` | 0.84 | Provides `EventBridgeNotification` POJO and other shared data objects |
| `threevictors-common-database-data-access-aurora-metadata` | 0.198 | `AuroraMetadataReader` base class used by `MetadataReader` |
| `aws-lambda-java-core` | 1.2.3 | Lambda `RequestHandler` interface |
| `aws-lambda-java-runtime-interface-client` | 2.6.0 | Lambda Runtime Interface Client (container image mode) |
| `aws-java-sdk` / `software.amazon.awssdk` | 2.29.22 | AWS SDK v2 (ECS, S3, SQS, etc.) |
