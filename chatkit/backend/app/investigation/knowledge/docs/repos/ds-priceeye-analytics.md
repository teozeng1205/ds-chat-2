# ds-priceeye-analytics

> Analytics pipeline for PriceEye that computes airline market/segment-level anomaly scores, competitive position, brand equivalence, OAG seat capacity scores, revenue scores, and price anomaly alerts — all deployed as ECS Fargate tasks, Lambda functions, a Glue ETL job, and a Step Function on AWS.

> **Current branch**: `develop` — **this document reflects the `develop` branch**. The `master` branch represents what is currently running in production; content documented here may differ from production.

---

## Architecture Overview

```
─────────────────────────────── SCHEDULED (cron) ────────────────────────────────

[EventBridge cron: daily 02:00 UTC]
      │
      ▼
[oag-score]  ──────────────────────────────────────► S3: {env}-use1-anomaly-datasets/oag_score/

[EventBridge cron: Saturdays 21:30 UTC]
      │
      ▼
[brands-equivalence] ──────────────────────────────► S3: {env}-use1-ds-standard-brands/brand_equivalence/v1/
                                                      Glue: brand_equivalence (brands_enrichment_db)

[EventBridge cron: daily 12:30 UTC]
      │
      ▼
[daily-itins] ─────────────────────────────────────► Redshift analytics (daily representative itineraries)

[EventBridge cron: daily 23:00 UTC]
      │
      ▼
[pax-midt] ────────────────────────────────────────► Redshift analytics (MIDT pax booking data)

[EventBridge cron: daily 23:15 UTC]
      │
      ▼
[revenue-score] ───────────────────────────────────► S3: {env}-use1-anomaly-datasets/revenue_score/v1/

[EventBridge cron: every hour at :30 UTC]
      │
      ▼
[alerts (Lambda)] ──reads Redshift segment data──► EventBridge: data-pipeline bus ("SegmentLevel" Price Anomaly)
      │                                              S3 audit: {env}-use1-pe-analytics-audits/alerts/v1/
      │
      └──────────────────────────────────────────► Glue: alerts_audit_v1 (pe_analytics_audits_db)

─────────────────────────── EVENT-DRIVEN PIPELINE ───────────────────────────────

[Upstream pipeline] ─── "SegmentLevel Task Completed" ──► [EventBridge: data-pipeline bus]
                                                                   │
                                                                   ▼
                                                  [Step Function: DS-Analytics-EventDriven-Jobs]
                                                          │
                                              ┌───────────┴──────────────┐
                                              ▼                          ▼
                                  [market-level-generator]  [segment-level-generator]
                                              │                          │
                                              ▼                          ▼
                               S3: anomaly-datasets/                S3: anomaly-datasets/
                               market_level/                        segment_level/
                               (Glue: market_level_v4+)            (Glue: segment_level_v4+)

─────────────────────────── TRIGGERED / NO SCHEDULE ────────────────────────────

[competitive-position]  ──reads S3 DCO data──► Redshift analytics.competitive_position
[market-level-analysis] ──reads Redshift────► Redshift analytics (market aggregates)
[segment-level-analysis] ─reads Redshift───► Redshift analytics → emits "SegmentLevel Task Completed"

─────────────────────────────── DCO TIMESCALE ──────────────────────────────────

[S3: derived-common-output/v1/B6/*/_SUCCESS]
      │  (EventBridge S3 Object Created)
      ▼
[ins-dco-generator (Lambda)]
      └──transforms B6→INS format──► S3: derived-common-output/v1/INS/

[S3: derived-common-output/v2/B6/*/_SUCCESS]
      │  (EventBridge S3 Object Created)
      ▼
[dcov2-tsdb-upload-b6 (Lambda)]
      └──────────────────────────► TimescaleDB (derived_common_output_v2)

[dco-tsdb-unload (Glue ETL)]
      └──reads Redshift analytics─► S3: (unloaded DCO data for downstream use)
```

---

## Orchestration

### Step Function: DS-Analytics-EventDriven-Jobs

- **Trigger**: EventBridge rule on the `data-pipeline` bus, listening for `Task Completed` events from source `threevictors.ecs.analytics` where `taskName` is `MarketLevel` or `SegmentLevel`. Only `SegmentLevel` events cause tasks to run; `MarketLevel` events are a no-op (`Nothing to run`).
- **Pipeline** (when triggered by SegmentLevel): `market-level-generator` → `segment-level-generator`
- **Arguments passed**: `customer` and `sales_date` extracted from the incoming event payload and injected as `ARGUMENTS` environment variable into each ECS task override.
- **Definition**: `source/deploy/definitions/ds-analytics-eventdriven-jobs-step-function.asl.json`

### Standalone EventBridge Cron Rules (not in Step Function)

| Component | Schedule |
|-----------|----------|
| `oag-score` | Daily 02:00 UTC |
| `daily-itins` | Daily 12:30 UTC |
| `pax-midt` | Daily 23:00 UTC |
| `revenue-score` | Daily 23:15 UTC |
| `brands-equivalence` | Saturdays 21:30 UTC |
| `alerts` | Every hour at :30 past |

---

## Components

_Ordered by when they conceptually run in the pipeline._

---

### oag-score

**Type**: ECS Fargate Task
**Trigger**: EventBridge cron — daily at 02:00 UTC
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Reads airline seat capacity data (OAG schedule) and the customer's own seat inventory from the PriceEye MySQL database. Computes a normalized market score (seats in market), customer score (customer's seats), and a `carrier_scores` (per-carrier market share fraction), then sums them into a composite `OAG_score_sum` for each metro O&D pair per carrier. Applies carrier substitutions from the `analytics.demo_carrier_substitutions` table. Writes the scored output as parquet to S3.

**Input**:
- MySQL (`analytics`): OAG seat data tables, `analytics.demo_carrier_substitutions`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-anomaly-datasets/oag_score/` (parquet)

---

### brands-equivalence

**Type**: ECS Fargate Task
**Trigger**: EventBridge cron — Saturdays at 21:30 UTC
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Reads itinerary-level fare family data from the Redshift `common_output` schema (via `common_output_format` table), aggregates price and booking counts by `(carrier, source, outbound_fare_family, cabin)`, then applies a multi-tier heuristic algorithm to elect which fare brand per airline represents the "discount economy" product (e.g., Basic Economy, Saver, Light). Outputs a parquet file with a `discount_economy` boolean flag and a `confidence_score` per brand, written to S3 and catalogued in the `brands_enrichment_db` Glue database.

**Input**:
- Redshift `common_output.common_output_format` (or configured input table), filtered to `sales_date` = yesterday

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-standard-brands/brand_equivalence/v1/{YYYY}/{MM}/{DD}/data.parquet`
- Glue table: `brand_equivalence` in `glue-atp-3victors-{env}-use1-brands_enrichment_db`

**Table Schema** (`brand_equivalence`):

| Column | Type |
|--------|------|
| airline | varchar(256) |
| source | varchar(256) |
| brand | varchar(256) |
| discount_economy | boolean |
| confidence_score | double |
| avg_min_price | double |
| cabin | varchar(256) |
| total_count | bigint |

_Partition key: `sales_date` (bigint)_
_S3 Location_: `s3://s3-atp-3victors-{env}-use1-ds-standard-brands/brand_equivalence/v1/`

---

### daily-itins

**Type**: ECS Fargate Task
**Trigger**: EventBridge cron — daily at 12:30 UTC
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Reads raw itinerary data from Redshift (`data_lakes.daily_representative_itinerary_v4` or the configured `dl_dri_input_table`) and common output format data (`common_output.common_output_format`) for the current sales date. Computes enriched daily representative itineraries across all unique points-of-sale for each customer and writes the aggregated results back to the Redshift analytics schema. This output feeds downstream anomaly models.

**Input**:
- Redshift: `data_lakes.daily_representative_itinerary_v4` (configurable), `common_output.common_output_format`

**Output**:
- Redshift analytics tables (daily representative itinerary data)

---

### pax-midt

**Type**: ECS Fargate Task
**Trigger**: EventBridge cron — daily at 23:00 UTC
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Pulls MIDT (Marketing Information Data Transfer) passenger booking volumes by market, carrier, and cabin from an external MIDT source or the Redshift `analytics` schema. Aggregates booking counts at the O&D market level and writes the result to the Redshift analytics database. This MIDT data is consumed by `revenue-score` the following quarter-hour.

**Input**:
- Redshift / MIDT source: passenger booking data per market/carrier/cabin

**Output**:
- Redshift analytics: MIDT pax aggregates

---

### revenue-score

**Type**: ECS Fargate Task
**Trigger**: EventBridge cron — daily at 23:15 UTC
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Joins MIDT pax booking counts (from `pax-midt`) with average price data from the analytics database, normalizing cabin groups between the two sources. Computes a revenue impact score per market/carrier/cabin segment and writes the results as parquet to S3. The revenue score is a weighting factor used by `market-level-generator` when computing anomaly impact scores.

**Input**:
- Redshift analytics: pax_midt data (from `pax-midt` job), average price data
- Configuration: `revenue-score.properties`

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-anomaly-datasets/revenue_score/v1/` (parquet)

---

### competitive-position

**Type**: ECS Fargate Task
**Trigger**: No schedule defined in this repo — triggered externally (event-driven or manual)
**Compute**: 6144 MB, 2 vCPU (ARM64, 200 GiB ephemeral) — largest task in the repo

**What it does**: Reads DCO (Derived Common Output) data in batch chunks from S3 via PyArrow, computing a competitive position classification (e.g., Undercut, Overpriced, Competitive) per market+carrier+advance-purchase+cabin combination. Uses `diff_min_ow` and `pcnt_diff_min_ow` fields to determine whether the customer carrier is overpriced or undercut relative to competitors. Writes output to the Redshift `analytics` schema for use by downstream anomaly models.

**Input**:
- S3: DCO data (`input.bucket`, `input.version` from properties — configurable)
- Reads columns: `customer_observation_date`, `origin_metro`, `destination_metro`, `carrier`, `customer`, `stops`, `length_of_stay`, `cabin`, `brand_group`, `price_inc`, etc.

**Output**:
- Redshift analytics: competitive position table (read by market/segment analysis jobs)

---

### market-level-analysis

**Type**: ECS Fargate Task
**Trigger**: No schedule defined in this repo — triggered externally (event-driven or from upstream pipeline)
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Reads market-level competitive pricing data from Redshift (reading from `common_output_format` or analytics schema), builds aggregated metrics per market (`mkt`) such as frequency of competitive position occurrence and magnitude of price differences over a rolling 22-day window. Writes the aggregated metric output back to Redshift analytics tables so that the `market-level-generator` can run IQR-based anomaly detection on it. Emits a `MarketLevel Task Completed` event on the `data-pipeline` EventBridge bus.

**Input**:
- Redshift analytics: competitive position data, `common_output_format`

**Output**:
- Redshift analytics: market-level metric aggregates
- EventBridge: `Task Completed` / `MarketLevel` event on `data-pipeline` bus (source: `threevictors.ecs.analytics`)

---

### segment-level-analysis

**Type**: ECS Fargate Task
**Trigger**: No schedule defined in this repo — triggered externally (event-driven or from upstream pipeline)
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Same pattern as `market-level-analysis` but at the segment level (a segment being a group of markets sharing a competitive characteristic). Reads Redshift analytics data, builds per-segment metric time series (frequency percentages, magnitude of price differences), and writes back to Redshift. Emits a `SegmentLevel Task Completed` event on the `data-pipeline` bus, which triggers the **DS-Analytics-EventDriven-Jobs** Step Function.

**Input**:
- Redshift analytics: competitive position data, market-level metrics

**Output**:
- Redshift analytics: segment-level metric aggregates
- EventBridge: `Task Completed` / `SegmentLevel` event on `data-pipeline` bus — **this triggers the Step Function**

---

### market-level-generator

**Type**: ECS Fargate Task
**Trigger**: Step Function (`DS-Analytics-EventDriven-Jobs`) — fired when a `SegmentLevel` Task Completed event arrives; receives `customer` and `sales_date` as ARGUMENTS env var
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Reads 22 days of market-level anomaly metrics from Redshift (`analytics` schema), fetches OAG scores (from S3 via analytics reader), revenue scores, and direction scores (weights from `analytics.anomalies_direction_score`). Runs an IQR-based anomaly detection model (`freq_pcnt`, `mag_nominal`, `mag_pcnt`) across `seg_mkt` aggregation for each competitive position dimension. Computes a composite impact score weighted by OAG score, revenue score, direction score, and customer-specific weights from `analytics.anomalies_impact_score_weights`. Writes the final market-level anomaly output as parquet to S3 and registers the partition in the Glue catalog.

**Input**:
- Redshift analytics: 22-day market anomaly metrics (`get_market_anomalies_df`)
- Redshift analytics: OAG scores, revenue scores, direction scores, impact score weights
- MySQL analytics: `anomalies_direction_score`, `anomalies_impact_score_weights`
- Configuration: `market-level-generator.properties` (output bucket/prefix, glue database/table)

**Output**:
- S3: `s3://{output.bucket}/{output.prefix}/{customer}/{YYYY}/{MM}/{DD}/data.parquet`
- Glue: partition registered in the configured `glue.database` / `glue.table`

---

### segment-level-generator

**Type**: ECS Fargate Task
**Trigger**: Step Function (`DS-Analytics-EventDriven-Jobs`) — runs immediately after `market-level-generator`; receives `customer` and `sales_date` as ARGUMENTS env var
**Compute**: 2048 MB, 1 vCPU (ARM64, 200 GiB ephemeral)

**What it does**: Mirrors `market-level-generator` but operates at the segment level. Reads 22 days of segment anomaly metrics from Redshift, runs IQR-based anomaly detection across `freq_pcnt`, `mag_nominal`, and `mag_pcnt` metrics for each segment dimension. Computes impact scores using direction scores, OAG scores, and revenue scores, with an additional outlier replacement step for historical data. Writes segment-level anomaly results as parquet to S3.

**Input**:
- Redshift analytics: 22-day segment anomaly metrics (`get_segment_anomalies_df`)
- Redshift analytics: OAG scores, revenue scores, direction scores, impact score weights
- Configuration: `segment-level-generator.properties`

**Output**:
- S3: `s3://{output.bucket}/{output.prefix}/{customer}/{YYYY}/{MM}/{DD}/data.parquet`

---

### alerts

**Type**: Lambda Function
**Trigger**: EventBridge cron — every hour at :30 past (e.g., 00:30, 01:30, ... 23:30 UTC)
**Compute**: 512 MB, 90 s timeout (ARM64)

**What it does**: Runs hourly but only processes customers whose scheduled UTC delivery hour matches the current hour (schedule is fetched from a MySQL database table). For each eligible customer, queries the Redshift `analytics` schema for the latest segment anomaly data, computes `change_direction` (Improving / Worsening / Mixed / No Change) based on 7-day price diffs, and applies the "Recommended Alerts" filter (impact score thresholds vary by cp and change direction). Fetches the top 3 impacted markets per alert from Redshift, limits to 20 alerts maximum per event, and publishes a consolidated EventBridge alert event to the `data-pipeline` bus. Also uploads filtered alert data as parquet to S3 for audit purposes.

**Input**:
- Redshift analytics: segment anomaly data (`fetch_segment_data.sql`, `fetch_market_data.sql`); table name configured in `alerts.properties`
- MySQL: customer schedule table (`utc_hour` per customer)

**Output**:
- EventBridge: `SegmentLevel` / `Price Anomaly` business event on `data-pipeline` bus (source: `threevictors.testing`)
- S3: `s3://s3-atp-3victors{env}-use1-pe-analytics-audits/alerts/v1/` (parquet audit records)
- Glue table: `alerts_audit_v1` in `glue-atp-3victors{env}-use1-pe_analytics_audits_db`

**Table Schema** (`alerts_audit_v1`):

| Column | Type |
|--------|------|
| alert_id | varchar(32) |
| event_id | varchar(64) |
| customer_code | varchar(16) |
| anomaly_date | int |
| timestamp | varchar(64) |
| segment | varchar(128) |
| competitive_position | varchar(32) |
| impact_score | double |
| change_direction | varchar(32) |
| num_markets | int |
| top_impacted_markets | varchar(128) |
| top_competitors | varchar(256) |
| freq_pcnt_val | double |
| abs_mag_pcnt_val_log_balanced | double |
| avg_fp_7d_diff | double |
| avg_fp_7d_diff_abs_scaled | double |
| avg_mp_7d_diff | double |
| avg_mp_7d_diff_abs_scaled | double |
| direction_score | double |
| any_anomaly | int |
| oag_score | double |

_Partition key: `sales_date` (int)_

---

### ins-dco-generator

**Type**: Lambda Function
**Trigger**: EventBridge S3 Object Created event — fires when a `_SUCCESS` marker is created at `s3://s3-atp-3victors{env}-use1-derived-common-output/v1/B6/*/_SUCCESS`
**Compute**: 1024 MB, 900 s timeout (ARM64)

**What it does**: Triggered when a new hour of B6 (JetBlue) DCO v1 data lands in S3. Reads all parquet files from that hour's prefix under `v1/B6/YYYY/MM/DD/HH/`, applies field transformations to rebrand B6 as INS (InsightAir customer): replaces the `customer` field, renames carrier codes (`B6` → `INS`), normalizes fare families (`MINT*` → `BUSINESS`, `BLUE*` or `VALUE*` → `ECONOMY`), and writes the transformed parquet files with an `-INS` filename suffix to `v1/INS/YYYY/MM/DD/HH/` in the same bucket. Writes a `_SUCCESS` marker when done.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-derived-common-output/v1/B6/YYYY/MM/DD/HH/*.parquet`
- EventBridge event payload: S3 Object Created detail with bucket + key

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-derived-common-output/v1/INS/YYYY/MM/DD/HH/*-INS.parquet`
- S3: `s3://s3-atp-3victors{env}-use1-derived-common-output/v1/INS/YYYY/MM/DD/HH/_SUCCESS`

---

### dco-tsdb-unload

**Type**: Glue ETL Job (Spark, Python 3, Glue 4.0, G.1X × 10 workers)
**Trigger**: Not defined in this repo — likely triggered externally (manually, or by schedule in another stack)

**What it does**: Connects to Redshift analytics via the `Analytics Connection` Glue connection, executes a large SQL query that assembles a denormalized observation record from several analytics tables (`collection_observation`, airport lookups, city code tables, etc.). Columns span all observation layers: trip context (POS, origin/destination cities, trip type, cabin, advance purchase), carrier details (marketing/operating carriers, codeshare, flight numbers), travel dates, price data, and quality indicators. The resulting dataset is exported to S3 (path configured via Glue job parameters). Supports both scheduled full loads and date-range loads via `--SALES_DATE`, `--SALES_DATE_BEGIN`, `--SALES_DATE_END` parameters.

**Input**:
- Redshift analytics: `collection_observation` and related lookup tables (via `Analytics Connection`)
- Glue job parameters: `--SALES_DATE`, `--ENV`, `--MODULE_NAME`, `--RUN_MODE`

**Output**:
- S3: (configured via `--TempDir` and job args — path not hardcoded in CloudFormation)

---

### dcov2-tsdb-upload-b6

**Type**: Lambda Function
**Trigger**: EventBridge S3 Object Created — fires when a `_SUCCESS` is created at `s3://{env}-use1-derived-common-output/v2/B6/YYYY/MM/DD/_SUCCESS`
**Compute**: 2048 MB, 300 s timeout (ARM64)

**What it does**: Triggered by a new day of B6 DCO v2 data arriving in S3. Reads parquet files from the B6 hourly prefix, parses them into a structured DataFrame using the `TABLE_COLUMNS` schema, and bulk-copies the data into a TimescaleDB table (`derived_common_output_v2_{customer}`) via `COPY` from a CSV stream. Creates the TimescaleDB table if it does not exist. Only processes B6 (`ONLY_CUSTOMER = "B6"`).

**Input**:
- S3: `s3://{env}-use1-derived-common-output/v2/B6/YYYY/MM/DD/*.parquet`
- Configuration: `dco-tsdb-config.properties` (TimescaleDB connection, bucket/prefix, table name)

**Output**:
- TimescaleDB: `derived_common_output_v2` table (per-customer)

---

## Glue Databases

| Database | Tables | Notes |
|----------|--------|-------|
| `glue-atp-3victors-{env}-use1-adf_db` | `assembled_data_feed_v1` | ADF data (ADF source); S3: `s3://s3-atp-3victors-{env}-use1-adf/assembled_data_feed_emr/` |
| `glue-atp-3victors-{env}-use1-infare_db` | `assembled_data_feed_v1` | Same schema as adf_db but under the Infare source namespace; same S3 location |
| `glue-atp-3victors-{env}-use1-brands_enrichment_db` | `brand_equivalence` | Written by `brands-equivalence`; S3: `…use1-ds-standard-brands/brand_equivalence/v1/` |
| `glue-atp-3victors{env}-use1-pe_analytics_audits_db` | `alerts_audit_v1` | Written by `alerts`; S3: `…use1-pe-analytics-audits/alerts/v1/` |

### assembled_data_feed_v1 Schema (adf_db and infare_db)

_Partition keys: `customer` (varchar 50), `sales_date` (bigint), `sales_hour` (bigint)_

Key columns (subset): `cxr`, `orig`, `dest`, `fare_class`, `o_r`, `trf`, `rtg`, `fn`, `cur`, `fare_amt`, `ow_amt`, `rt_amt`, `market`, `cabin`, `tax_amt`, `total_price_amt`, `ap`, `min_stay`, `max_stay`, `first_tvl`, `last_tvl`, `nonstop`, `direct`, `rbd`, `outbound_travel_date`, `inbound_travel_date`, `outbound_day_of_week`, `inbound_day_of_week`, `rule_title`, `first_res`, `last_res`, `ref_fare_amt`, `outbound_segment_marketing_flights`, `inbound_segment_marketing_flights`

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| ECS Fargate Task Definitions | 10 |
| Lambda Functions | 3 (`alerts`, `ins-dco-generator`, `dcov2-tsdb-upload-b6`) |
| Glue ETL Jobs | 1 (`dco-tsdb-unload`) |
| Step Functions | 1 (`DS-Analytics-EventDriven-Jobs`) |
| Glue Databases | 4 |
| Glue Tables | 5 (`assembled_data_feed_v1` × 2, `brand_equivalence`, `alerts_audit_v1`, + market/segment tables registered at runtime) |
| EventBridge Rules | 7 (5 cron + 1 Step Function trigger + 1 S3 event for ins-dco-generator) |
| S3 Buckets (defined here) | 1 (`pe-analytics-audits`) |
| CloudWatch Log Groups | 1 per component (7-day retention) |
| CloudWatch Alarms | 2 (timeout alarms on `alerts` and `ins-dco-generator`) |
