# priceeye-monitoring

> Monitors the PriceEye flight-shopping pipeline by extracting and deduplicating audit data from Redshift into S3/Glue, then running daily health checks, error analysis, and reporting tasks.

> **Current branch**: `develop` — this document reflects the `develop` branch. The `master` branch represents what is currently running in production; some details may differ.

---

## Architecture Overview

```
Redshift: priceeye{env} schema  (source audit tables written by the pipeline)
  │
  ├──► [Glue ETL: Unload-Deduped-Provider-Request-Audit]        ──► s3://deduped-datasets{env}/v1/provider_request_audit/
  ├──► [Glue ETL: Unload-Deduped-Provider-Request-Audit-Detail]  ──► s3://deduped-datasets{env}/v1/provider_request_audit_detail/
  ├──► [Glue ETL: Unload-Deduped-Provider-Response-Audit]        ──► s3://deduped-datasets{env}/v1/provider_response_audit/
  ├──► [Glue ETL: Unload-Deduped-Cache-Loader-Audit]             ──► s3://deduped-datasets{env}/v1/cache_loader_audit/
  ├──► [Glue ETL: Unload-Deduped-Enrichment-Audit]               ──► s3://deduped-datasets{env}/v1/enrichment_audit/
  ├──► [Glue ETL: Unload-Deduped-Global-Filter-Audit-Summary]    ──► s3://deduped-datasets{env}/v1/global_filter_audit_summary/
  ├──► [Glue ETL: Unload-Deduped-Packager-Audit]                 ──► s3://deduped-datasets{env}/v1/packager_audit/
  ├──► [Glue ETL: Unload-Deduped-Retry-Audit]                    ──► s3://deduped-datasets{env}/v1/retry_audit/
  └──► [Glue ETL: Unload-Deduped-Delivery-Audit]                 ──► s3://deduped-datasets{env}/v1/delivery_audit/
                                                                           │
                                                              Glue Data Catalog: monitoring{env} DB
                                                              (9 deduped tables, partitioned by sales_date)
                                                                           │
                                                                           ▼
  Redshift: monitoring{env} schema  ◄──── [Glue ETL: Unload-Combined-Audit] ──► s3://deduped-datasets{env}/v1/combined_audit/
  (external schema backed by Glue)                                         (joins all 9 deduped tables)
            │
            ▼
  Redshift: monitoring_metadata schema  (native — materialized views, error mapping)
            │
            ├──► [ECS: verify-dedupe]       — refreshes materialized views, checks for duplicate records, Slack alert
            ├──► [ECS: error-mapper]        — discovers new error patterns, updates error_mapping table, email alert
            ├──► [ECS: swav-report]         — queries wn_request_health view, sends WN Vacations health email
            └──► [ECS: delete-old-audits]   — purges combined_audit records older than 8 days
```

---

## Components

_Ordered by data flow: ETL first (data preparation), then operational tasks (health checks and reporting)._

---

### Glue ETL: Unload-Deduped-Provider-Request-Audit

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)
**Run modes**: `daily_normal_load` (single date, hourly window) · `manual_historical_load_one_time` (date range backfill)

**What it does**: Reads `priceeye{env}.provider_request_audit` from Redshift via the `monitoring-user_code` Glue connection, deduplicates rows by grouping on all non-aggregated fields (emitting an `occurrences` count), and writes the result to S3 as Snappy-compressed Parquet. It then registers or updates the Glue table partition for the processed date, making the data available to downstream consumers.

**Input**:
- Redshift table: `priceeye{env}.provider_request_audit` — raw provider request records including schedule timestamps, provider/site codes, route details, passenger info, and filter reason

**Output**:
- S3: `s3://deduped-datasets{env}/v1/provider_request_audit/{YYYY}/{MM}/{DD}/{HH}/` (Parquet, snappy)
- Glue table: `deduped_provider_request_audit` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Deduped-Provider-Request-Audit-Detail

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)

**What it does**: Reads `priceeye{env}.provider_request_audit_detail` from Redshift, deduplicates rows, and writes to S3 as Parquet. Registers the Glue partition for the date. This table carries customer-level context for each request (customer, site code, POS, site category, input request ID).

**Input**:
- Redshift table: `priceeye{env}.provider_request_audit_detail`

**Output**:
- S3: `s3://deduped-datasets{env}/v1/provider_request_audit_detail/{YYYY}/{MM}/{DD}/{HH}/`
- Glue table: `deduped_provider_request_audit_detail` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Deduped-Provider-Response-Audit

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)

**What it does**: Reads `priceeye{env}.provider_response_audit` from Redshift and deduplicates response records (status, error message, itinerary count, POS site, timestamps). Writes Parquet to S3 and updates the Glue partition.

**Input**:
- Redshift table: `priceeye{env}.provider_response_audit`

**Output**:
- S3: `s3://deduped-datasets{env}/v1/provider_response_audit/{YYYY}/{MM}/{DD}/{HH}/`
- Glue table: `deduped_provider_response_audit` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Deduped-Cache-Loader-Audit

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)

**What it does**: Reads and deduplicates cache loader audit records from Redshift, capturing itinerary counts processed through the cache layer. Writes to S3 as Parquet and registers the Glue partition.

**Input**:
- Redshift table: `priceeye{env}.cache_loader_audit`

**Output**:
- S3: `s3://deduped-datasets{env}/v1/cache_loader_audit/{YYYY}/{MM}/{DD}/{HH}/`
- Glue table: `deduped_cache_loader_audit` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Deduped-Enrichment-Audit

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)

**What it does**: Reads and deduplicates enrichment audit records tracking per-enrichment-type success/fail counts (brand, tax, OAG, booking code, directional price, operating carrier, fare basis, cache). Writes to S3 as Parquet and registers the Glue partition.

**Input**:
- Redshift table: `priceeye{env}.enrichment_audit`

**Output**:
- S3: `s3://deduped-datasets{env}/v1/enrichment_audit/{YYYY}/{MM}/{DD}/{HH}/`
- Glue table: `deduped_enrichment_audit` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Deduped-Global-Filter-Audit-Summary

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)

**What it does**: Reads and deduplicates global filter audit summary records, capturing itinerary counts before and after filtering. Writes to S3 as Parquet and registers the Glue partition.

**Input**:
- Redshift table: `priceeye{env}.global_filter_audit_summary`

**Output**:
- S3: `s3://deduped-datasets{env}/v1/global_filter_audit_summary/{YYYY}/{MM}/{DD}/{HH}/`
- Glue table: `deduped_global_filter_audit_summary` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Deduped-Packager-Audit

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)

**What it does**: Reads and deduplicates packager audit records, including reference group, file URI, record counts, and substitute provider/site fields. Writes to S3 as Parquet and registers the Glue partition.

**Input**:
- Redshift table: `priceeye{env}.packager_audit`

**Output**:
- S3: `s3://deduped-datasets{env}/v1/packager_audit/{YYYY}/{MM}/{DD}/{HH}/`
- Glue table: `deduped_packager_audit` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Deduped-Retry-Audit

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)

**What it does**: Reads and deduplicates retry audit records tracking which requests were retried, by which provider/site, and the retry reason. Writes to S3 as Parquet and registers the Glue partition.

**Input**:
- Redshift table: `priceeye{env}.retry_audit`

**Output**:
- S3: `s3://deduped-datasets{env}/v1/retry_audit/{YYYY}/{MM}/{DD}/{HH}/`
- Glue table: `deduped_retry_audit` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Deduped-Delivery-Audit

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled (daily, per-hour incremental)

**What it does**: Reads and deduplicates delivery audit records capturing delivery customer, reference group, delivery ID, type, statuses, and failure reason. Writes to S3 as Parquet and registers the Glue partition.

**Input**:
- Redshift table: `priceeye{env}.delivery_audit`

**Output**:
- S3: `s3://deduped-datasets{env}/v1/delivery_audit/{YYYY}/{MM}/{DD}/{HH}/`
- Glue table: `deduped_delivery_audit` in `monitoring{env}` database, partitioned by `sales_date`

---

### Glue ETL: Unload-Combined-Audit

**Type**: AWS Glue Job (Python / PySpark)
**Trigger**: Scheduled daily (runs after all 9 deduped ETL jobs complete)
**Run modes**: `daily_normal_load` · `manual_historical_load_one_time`

**What it does**: Joins all nine deduped Glue tables into a single wide `combined_audit` view. Uses a multi-CTE SQL query spanning provider request, request detail, response, error mapping, retry, global filter, enrichment, cache loader, packager, and delivery tables. Applies the `monitoring_metadata.error_mapping` regex rules to classify response errors into `issue_source` and `issue_reason` fields. Writes the result to S3 as Snappy Parquet and registers the partition in the Glue catalog, making it immediately available to the Redshift `monitoring` external schema.

**Input**:
- Glue tables (all from `monitoring{env}` database):
  - `deduped_provider_request_audit`
  - `deduped_provider_request_audit_detail`
  - `deduped_provider_response_audit`
  - `deduped_global_filter_audit_summary`
  - `deduped_enrichment_audit`
  - `deduped_cache_loader_audit`
  - `deduped_packager_audit`
  - `deduped_retry_audit`
  - `deduped_delivery_audit`
- Redshift table: `monitoring_metadata.error_mapping` (regex patterns for error classification)

**Output**:
- S3: `s3://deduped-datasets{env}/v1/combined_audit/{YYYY}/{MM}/{DD}/` (Parquet, snappy)
- Glue table: `combined_audit` in `monitoring{env}` database, partitioned by `sales_date`
- Visible in Redshift as `monitoring.combined_audit` (external schema)

**Key combined_audit fields** _(selected)_:

| Column | Type | Description |
|--------|------|-------------|
| `id` | bigint | Provider request audit ID |
| `input_req_id` | bigint | Input request ID |
| `customer` | varchar(16) | Customer name |
| `customer_site_code` | varchar(64) | Customer site code |
| `schedule_date` / `schedule_time` | int | Scheduled date and time |
| `actual_schedule_timestamp` | timestamp | Actual scheduled time |
| `provider_code` | char(16) | Provider identifier |
| `site_code` | char(64) | Site identifier |
| `origin_airport_code` / `destination_airport_code` | char(3) | Route |
| `trip_type` | char(2) | `OW` one-way, `RT` round-trip |
| `filter_reason` | varchar(64) | Why request was filtered (empty if valid) |
| `response_status` | varchar(16) | `success`, `failed`, `timeout`, etc. |
| `response_error_message` | varchar(4096) | Raw error string from provider |
| `issue_source` | varchar(4096) | Error classification: `site`, `request`, or unclassified |
| `issue_reason` | varchar(4096) | Human-readable issue label from error_mapping |
| `itins_before_filtering` / `itins_after_filtering` | bigint | Global filter throughput |
| `enrichment_status` | varchar(2048) | Enrichment pipeline status |
| `packager_file_uri` | varchar(8192) | S3 path of packaged output file |
| `delivery_status` | varchar(8192) | Delivery outcome |
| `sales_date` | int (partition) | Partition key, YYYYMMDD format |

---

### ECS: verify-dedupe

**Type**: ECS Fargate Task (Java 11, ARM64)
**Compute**: 2048 MB memory, 1024 vCPU
**Trigger**: Daily (scheduled via ECS run-task; Slack alert fired at 12:00 PM UTC)
**Docker image**: `732267085676.dkr.ecr.us-east-1.amazonaws.com/3victors/monitoring/verify-dedupe`
**Logs**: CloudWatch `priceeye-monitoring/{StackName}`, 7-day retention

**What it does**: Connects to the Redshift monitoring cluster, refreshes all materialized views in the `monitoring_metadata` schema, then queries each of the nine deduped tables (and `combined_audit`) for duplicate rows. If duplicates are found, it sends a Slack notification with a summary of affected tables, provider codes, and duplicate counts. Credentials are retrieved from AWS Secrets Manager.

**Input**:
- Redshift external tables in `monitoring` schema (via Glue): all 9 deduped tables + `combined_audit`
- AWS Secrets Manager: Redshift credentials

**Output**:
- Refreshed materialized views in `monitoring_metadata` (see list below)
- Slack alert if duplicates are detected

**Materialized views refreshed** (all in `monitoring_metadata`):

| View | Description |
|------|-------------|
| `customer_issue_summary` | Error counts by customer and issue_reason (yesterday) |
| `customer_request_triptype_summary` | OW vs RT success/failure rates by customer (last 5 days) |
| `customer_rollup` | Daily customer-level pipeline health: valid, successful, packaged, delivered (last 7 days) |
| `daily_provider_performance` | Daily success/timeout/failure counts by provider (last 7 days) |
| `hourly_provider_performance` | Hourly success/timeout/failure counts by provider (last 7 days) |
| `request_summary_by_customer_ref_group` | Total itineraries by customer and packager reference group (last 5 days) |
| `request_triptype_summary` | OW vs RT success rates (last 5 days) |
| `site_categories_by_customer_site` | Valid requests by customer, site, and site category (yesterday) |
| `site_categories_by_site` | Valid requests by site and site category (yesterday) |
| `site_issues_report` | Site and request issue rates with breakdown by issue type (yesterday) |
| `summary_by_customer_site_code` | Hourly valid requests, results, issue counts by customer+site (yesterday) |
| `summary_by_site_code` | Hourly valid requests and issue counts by site (yesterday) |
| `wn_request_health` | WN Vacations request health by Air/AirHotel/AirHotelCar product type (last 60 days) |
| `wn_request_summary` | WN Vacations request counts by product type (last 60 days) |

---

### ECS: error-mapper

**Type**: ECS Fargate Task (Java 11, ARM64)
**Compute**: 2048 MB memory, 1024 vCPU
**Trigger**: Daily (scheduled via ECS run-task)
**Docker image**: `732267085676.dkr.ecr.us-east-1.amazonaws.com/3victors/monitoring/error-mapper`
**Logs**: CloudWatch `priceeye-monitoring/{StackName}`, 7-day retention

**What it does**: Queries `monitoring.provider_response_audit` for recent error messages not yet covered by existing regex patterns in `monitoring_metadata.error_mapping`. Trims each new error to its first 16 characters, deduplicates, escapes special regex characters, and batch-inserts new `^{escapedError}.*` patterns into `error_mapping`. Sends an email report listing new error examples per provider with their occurrence counts. Credentials are retrieved from AWS Secrets Manager.

**Input**:
- Redshift: `monitoring_metadata.error_mapping` (existing regex patterns)
- Redshift: `monitoring.provider_response_audit` (recent error messages)
- AWS Secrets Manager: Redshift credentials + email config
- Config file: `ErrorMapper.properties`

**Output**:
- Redshift: `monitoring_metadata.error_mapping` (new error regex rows inserted, batch size 500)
- Email report: new error patterns and examples, sent via AWS SNS

---

### ECS: swav-report

**Type**: ECS Fargate Task (Java 11, ARM64)
**Compute**: 2048 MB memory, 1024 vCPU
**Trigger**: Daily (scheduled via ECS run-task; default date: yesterday)
**Docker image**: `732267085676.dkr.ecr.us-east-1.amazonaws.com/3victors/monitoring/swav-report`
**Logs**: CloudWatch `priceeye-monitoring/{StackName}`, 7-day retention

**What it does**: Queries the `monitoring_metadata.wn_request_health` materialized view for the target sales date, groups results by product type (`Air`, `AirHotel`, `AirHotelCar`) and customer site code, and sends a formatted email report showing request volume, valid-request counts, success percentages, and issue breakdowns (site, request, unclassified) for WN Vacations.

**Input**:
- Redshift: `monitoring_metadata.wn_request_health` (pre-aggregated WN Vacations health data)
- AWS Secrets Manager / Config: Redshift credentials, email recipients
- Config file: `verify-dedupe.properties` (shared config)

**Output**:
- Email report: SWAV health summary per product type, sent via AWS SNS

---

### ECS: delete-old-audits

**Type**: ECS Fargate Task (Java 11, ARM64)
**Compute**: 2048 MB memory, 1024 vCPU
**Trigger**: Daily (scheduled via ECS run-task)
**Docker image**: `732267085676.dkr.ecr.us-east-1.amazonaws.com/3victors/monitoring/delete-old-audits`
**Logs**: CloudWatch `priceeye-monitoring/{StackName}`, 7-day retention

**What it does**: Connects to Redshift and deletes records from `monitoring_metadata.combined_audit` where `sales_date` is older than the configured retention window (default: 8 days). This keeps the native `combined_audit` table from growing unbounded. The retention window is configurable via command-line argument.

**SQL executed**:
```sql
DELETE FROM monitoring_metadata.combined_audit WHERE sales_date <= {date}
```

**Input**:
- Redshift: `monitoring_metadata.combined_audit`
- Credentials: `ConfigurationReader` files `psu-no-macros.txt` / `psp-no-macros.txt`

**Output**:
- Redshift: rows purged from `monitoring_metadata.combined_audit` (sales_date ≤ cutoff)

---

## Glue Databases

| Database | Tables | Backed by S3 |
|----------|--------|--------------|
| `monitoring{env}` (e.g., `monitoring`, `monitoring_dev`) | `combined_audit`, `deduped_provider_request_audit`, `deduped_provider_request_audit_detail`, `deduped_provider_response_audit`, `deduped_cache_loader_audit`, `deduped_enrichment_audit`, `deduped_global_filter_audit_summary`, `deduped_packager_audit`, `deduped_retry_audit`, `deduped_delivery_audit` | `s3://deduped-datasets{env}/v1/` |

All tables are partitioned by `sales_date` (int, YYYYMMDD), stored as Parquet with Snappy compression.

---

## Redshift Schemas

| Schema | Type | Description |
|--------|------|-------------|
| `priceeye{env}` | Native (source) | Raw audit tables written by the PriceEye pipeline; read by Glue ETL jobs |
| `monitoring` | External (Glue-backed) | 10 external tables backed by `s3://deduped-datasets{env}/v1/`; registered from Glue Data Catalog `monitoring{env}` DB |
| `monitoring_metadata` | Native | Error mapping table, combined_audit native copy, and 14 materialized views for dashboards and reports |

**Redshift cluster**: `redshift-monitoring.clients.3victorsaws.com:5439`

---

## Infrastructure Summary

| Resource | Count | Notes |
|----------|-------|-------|
| ECS Fargate Tasks | 4 | delete-old-audits, error-mapper, swav-report, verify-dedupe |
| AWS Glue Jobs | 10 | 9 deduped unload scripts + 1 combined audit |
| Glue Databases | 1 per env | `monitoring{env}` |
| Glue Tables | 10 | 9 deduped + combined_audit |
| Redshift Schemas | 3 per env | priceeye, monitoring (external), monitoring_metadata |
| Materialized Views | 14 | All in monitoring_metadata schema |
| CloudWatch Log Groups | 1 per task | `priceeye-monitoring/{StackName}`, 7-day retention |
| ECR Registry | 1 | `732267085676.dkr.ecr.us-east-1.amazonaws.com` |

---

## Build & Deployment

**Language / Runtime**: Java 11 (Maven multi-module), Python 3 (Glue ETL)
**Docker base image**: `amazoncorretto:11` (ARM64)
**Maven build**: `mvn clean package`
**Container registry**: `732267085676.dkr.ecr.us-east-1.amazonaws.com/3victors/monitoring/<component>`

**Environments**:

| Env name | ECS cluster | Notes |
|----------|-------------|-------|
| `3vdev` | `ecs-3vdev-use1-generaladmintask` | Dev/test |
| `3vgold` | `ecs-3vgold-use1-generaladmintask` | Staging/Gold |
| `3vprod` | `ecs-3vprod-use1-generaladmintask` | Production |

**CloudFormation stack prefix**: `ECS-monitoring`
**ECS task template**: `source/common-scripts/commonfiles/task.yaml`

**Glue ETL environment substitution variables** (replaced at job invocation time):

| Variable | Description |
|----------|-------------|
| `#YEAR#` / `#MONTH#` / `#DAY#` / `#HOUR#` | Date/time components |
| `#SALES_DATE#` | Partition value, YYYYMMDD |
| `#SALES_DATE_LIST#` | Comma-separated dates for SQL `IN` clause |
| `#HOUR_START#` / `#HOUR_END#` | Timestamp range for hourly filtering |
| `#S3_ENV#` | S3 bucket suffix (`""`, `-dev`, `-gold`, `-prod`) |
| `#SCHEMA_ENV#` | Redshift/Glue schema suffix (`""`, `_dev`, etc.) |
| `#GLUE_ENV#` | Glue environment identifier (`n/a` for prod) |

---

## Key Source Files

| File | Description |
|------|-------------|
| `source/common-scripts/commonfiles/task.yaml` | ECS TaskDefinition CloudFormation template shared by all 4 Java components |
| `source/common-scripts/project-config.sh` | Deployment configuration: environment → cluster mappings |
| `source/glue-etl/source/Unload-Combined-Audit.py` | Main combined-audit ETL; contains the master multi-CTE join SQL |
| `source/delete-old-audits/src/main/java/.../DeleteAudits.java` | Redshift purge logic |
| `source/error-mapper/src/main/java/.../ErrorMapper.java` | Error pattern discovery and mapping |
| `source/swav-report/src/main/java/.../SwavReporter.java` | WN Vacations health report |
| `source/verify-dedupe/src/main/java/.../VerifyDedupe.java` | Duplicate detection and materialized view refresh |
| `docs/redshift/monitoring.sql` | External schema + Glue-backed table DDL |
| `docs/redshift/monitoring_metadata.sql` | Native schema DDL (error_mapping, combined_audit_new) |
| `docs/redshift/views.sql` | All 14 materialized view definitions |
| `docs/schemas/combined_audit.json` | JSON schema for combined_audit (136 fields) |
