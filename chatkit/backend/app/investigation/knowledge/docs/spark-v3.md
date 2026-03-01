# spark-v3

> EMR-based Spark pipeline that produces hourly and daily representative itineraries, demand debot signals, TMO campaign data, tax modeling outputs, and flight schedule data for the PriceEye product.

> **Current branch**: `develop` — this document reflects the `develop` branch. The production branch is `master`; the documented state may differ from what is currently running in production.

---

## Architecture Overview

```
[EventBridge cron: every hour at :10]
          │
          ▼
[Step Function: EMR-Hourly-Cluster]
  (creates EMR cluster only at 04:10, 10:10, 16:10, 22:10 UTC)
          │
          ├──► [EMR-Create-Cluster]  ←── capacity reservation fallback
          │         (r7g.4xlarge → r6g.8xlarge → r6gd.8xlarge → m6g.8xlarge)
          │
          ├──► [EMR-Add-Hourly-Steps]  ── loops over 5-6 hour window
          │         └──► spark-submit: hourly representative itinerary jobs (per hour)
          │                   └──► S3: dataset-hourly-representative-itinerary-parquet/v2/
          │
          ├──► [tmo-replacement]
          │         └──► S3: campaign-specific TMO output (partitioned by campaign)
          │
          ├──► [debot-demand]
          │         └──► S3: ds-demand-debot/v1/  ──► Glue: demand_debot_v1
          │
          └──► [debot-demand-estream]
                    └──► S3: ds-demand-debot-estream/  ──► Glue: demand_debot_estream
                              │
                              ▼  (EMR Step Status Change: "Hourly Step*" COMPLETED)
               [EventBridge: aws.emr step status]
                              │
                              ▼
               [Step Function: EMR-Daily-Cluster]
                 (per geographic zone: Oceania/Asia=24, EMEA=19, Americas=32 instances)
                              │
                              ├──► [daily-representative-itinerary-json]  × 5 zones
                              │         └──► S3: ds-daily-rep-parquet/v4/
                              │                   └──► Glue: daily_representative_itinerary_v4
                              │                   └──► Redshift Spectrum: data_lakes.daily_representative_itinerary_v4
                              │
                              └──► [City Summary]  × 5 zones

Standalone (separate triggers or manual invocation):
  [neo-price-itinerary-loader]  ── hourly, reads estream RawSearch
        └──► S3: dataset-neo-price-itinerary-loader-parquet/v1/
  [flight-schedule-harvest]  ── hourly/daily, reads search-with-itineraries-avro
        └──► S3: dataset-flight-schedule-harvest-parquet/v1/
  [tax-model-data-generator]  ── hourly, reads hourly rep itinerary
        └──► S3: configurable output bucket
  [tax-regression]  ── daily, reads daily rep itinerary + Redshift
        └──► S3: ds-tax-regression/v2/  ──► Glue: tax_regression_v2
  [historical-data]  ── ECS Fargate task (wn-historical), reads price-eye-common-output
        └──► S3: 3v-wn-historical-{env}/
```

---

## Orchestration

### Step Function: EMR-Hourly-Cluster

- **Trigger**: EventBridge rule `EMR-Hourly-Initiate` — cron `cron(10 * ? * * *)` — fires every hour at :10
- **Behavior**: Checks the current hour value; only creates a full EMR cluster at hours 04, 10, 16, and 22 UTC (four times per day). For other hours, adds steps to the existing running cluster.
- **Pipeline**: Check Hour → EMR-Create-Cluster → EMR-Add-Hourly-Steps → tmo-replacement → debot-demand → debot-demand-estream
- **Definition**: `deploy/commonfiles/EMR-Hourly-Cluster.yaml`

### Step Function: EMR-Daily-Cluster

- **Trigger**: EventBridge event — `aws.emr` source, `EMR Step Status Change` detail-type, state `COMPLETED`, step name matching `Hourly Step*`
- **Behavior**: Creates a larger zone-specific EMR cluster (19–32 instances depending on zone) and runs daily representative itinerary + city summary jobs for all 5 geographic zones.
- **Pipeline**: Get Dates → EMR-Create-Cluster (per zone) → EMR-Daily-Cluster-Zone-Add-Steps (Daily Rep Itinerary JSON × zone, City Summary × zone)
- **Definition**: `deploy/commonfiles/EMR-Daily-Cluster.yaml`

### Step Function: EMR-Create-Cluster

- **Trigger**: Called by both EMR-Hourly-Cluster and EMR-Daily-Cluster as a nested step function.
- **Behavior**: Attempts to reserve EC2 capacity with fallback across instance types (`r7g.4xlarge` → `r6g.8xlarge` → `r6gd.8xlarge` → `m6g.8xlarge`) and AZs (`us-east-1a/b/c`). Cancels the reservation after cluster creation.
- **EMR Release**: `emr-7.5.0`
- **Definition**: `deploy/commonfiles/EMR-Create-Cluster.yaml`

### Step Function: EMR-Add-Hourly-Steps

- **Trigger**: Called by EMR-Hourly-Cluster.
- **Behavior**: Iterator loop — submits one `spark-submit` EMR step per hour in the assigned window (5–6 hours per cluster run). Each step uses `--driver-memory 24G`.
- **Definition**: `deploy/commonfiles/EMR-Add-Hourly-Steps.yaml`

### Step Function: EMR-Daily-Cluster-Zone-Add-Steps

- **Trigger**: Called by EMR-Daily-Cluster for each zone.
- **Behavior**: Adds two EMR steps per zone: Daily Representative Itinerary JSON and City Summary.
- **Definition**: `deploy/commonfiles/EMR-Daily-Cluster-Zone-Add-Steps.yaml`

### Supporting Step Functions

| State Machine | Purpose |
|---------------|---------|
| `EMR-Add-Step` | Wrapper that invokes `elasticmapreduce:addStep` for a single Spark job |
| `EMR-Create-Capacity-Reservation` | Loops through instance type/AZ combinations to find available capacity |
| `EMR-Error-Processing` | Checks if EMR cluster terminated with errors vs. success |
| `EMR-Step-Failure` | Handles individual EMR step-level failures |
| `3Victors-General-Failure-Handling` | Sends failure notification to Slack `#Operations` channel |
| `Send-Message-To-Slack-Operations` | HTTP POST to Slack webhook |

---

## Components

_Ordered by position in the pipeline — hourly jobs first, then daily, then standalone._

---

### tmo-replacement

**Type**: EMR Spark Job (Java)
**Trigger**: Step in EMR-Hourly-Cluster, after hourly steps complete
**Compute**: EMR cluster (shared); driver memory 24 GB

**What it does**: Reads the most recent N hours (default 4) of hourly representative itinerary parquet files filtered to US point-of-sale, then loads active campaign definitions (market pairs, length-of-stay constraints, stops, departure days) from an RDS database via AWS Secrets Manager. For each campaign it filters, validates, groups by route+date+carrier+cabin+stops, and keeps the most recent record per group, writing campaign-partitioned parquet to S3.

**Input**:
- S3: `s3://dataset-hourly-representative-itinerary-parquet/v2/{year}/{month}/{day}/{hour}/pos_country_code=US/`
- RDS: campaign definitions table (via `TmoDatabaseAccess` + SecretsManager `RDS/user_code`)

**Output**:
- S3: configurable output path, partitioned by campaign

---

### debot-demand

**Type**: EMR Spark Job (Java)
**Trigger**: Step in EMR-Hourly-Cluster, after tmo-replacement
**Compute**: EMR cluster (shared); driver memory 24 GB

**What it does**: Removes bot and duplicate signals from the daily demand summary data using a multi-stage statistical pipeline: BotFilter, SpikeFilter, VolumeFilter, CarrierFilter, FanoutFilter, and SplitTicketFilter. Reads parquet demand summary files for the given sales date, applies all filters in sequence, and writes deduplicated demand records as parquet partitioned by sales date. Supports S3, JDBC, and CSV input modes.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-dataset-demand-summary-parquet/v1/{year}/{month}/{day}/`
- Glue table: `demand_summary_v1_parquet` from `glue-atp-3victors-{env}-use1-data_lakes_db`

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-demand-debot/v1/{year}/{month}/{day}/`
- Glue table: `demand_debot_v1` (partition registered via partition_creator)

**Table Schema** (demand_debot_v1):

| Column | Type |
|--------|------|
| origin_city_code | char(3) |
| destination_city_code | char(3) |
| depart_date | int |
| return_date | int |
| searches | int |

_Partition keys: `sales_date`_

---

### debot-demand-estream

**Type**: EMR Spark Job (Java)
**Trigger**: Step in EMR-Hourly-Cluster, after debot-demand
**Compute**: EMR cluster (shared); driver memory 24 GB

**What it does**: Applies the same multi-stage deduplication pipeline as `debot-demand` but operates on the estream-sourced demand summary dataset. Reads estream demand summary parquet for the given sales date, runs all deduplication filters, and writes cleaned demand records partitioned by date. Also registers the new Glue partition in `spectrum_db.demand_debot_estream_only_v3`.

**Input**:
- S3: `s3://s3-atp-3victors{env}-use1-dataset-demand-summary-parquet/v1/{year}/{month}/{day}/` (estream path)

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-demand-debot-estream/v1/{year}/{month}/{day}/`
- Glue table: `demand_debot_estream` (partition registered via partition_creator)

**Table Schema** (demand_debot_estream):

| Column | Type |
|--------|------|
| origin_city_code | char(3) |
| destination_city_code | char(3) |
| depart_date | int |
| return_date | int |
| searches | int |

_Partition keys: `sales_date`_

---

### daily-representative-itinerary-json

**Type**: EMR Spark Job (Java)
**Trigger**: Step in EMR-Daily-Cluster (runs after hourly cluster completes), once per geographic zone
**Compute**: EMR cluster (zone-specific: 19–32 instances); driver memory 24 GB

**What it does**: Aggregates a full day of hourly representative itinerary parquet records into daily summaries for a specific geographic zone. Each zone covers a timezone-aware UTC hour range (e.g., zone 1 Oceania/Asia: 12:00 previous day to 15:59 same day). Reads `FlattenedHourlyRepresentativeItineraryJson`, groups by route (origin, destination, dates), reduces hourly records into a single daily entry, and writes GZIP-compressed parquet partitioned by zone and point-of-sale country code.

**Input**:
- S3: `s3://{input.bucket}/{input.version}/{year}/{month}/{day}/{hour}/` (hourly rep itinerary, for zone-specific hour range)

**Zone Hour Ranges** (UTC):
| Zone | Coverage | UTC Window |
|------|----------|-----------|
| 1 | Oceania/Asia (NZ GMT+12, Perth GMT+8) | prev day 12:00 – same day 15:59 |
| 2 | Asia (Japan GMT+9, India GMT+5.5) | prev day 15:00 – same day 18:59 |
| 3 | EMEA (UAE GMT+4, Iceland GMT) | prev day 20:00 – same day 23:59 |
| 4 | Americas (Caribbean GMT-4, Pacific GMT-8) | same day 04:00 – next day 08:00 |
| 5 | Americas (alternate) | same day 04:00 – next day 08:00 |

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-daily-rep-parquet/v4/{year}/{month}/{day}/dr_zone_id={zoneId}/pos_country_code={pos}/`
- Glue table: `daily_representative_itinerary_v4` (partition registered via partition_creator)
- Redshift Spectrum: `data_lakes.daily_representative_itinerary_v4`

**Table Schema** (daily_representative_itinerary_v4, selected key columns):

| Column | Type | Notes |
|--------|------|-------|
| version | int | |
| shop_date | int | YYYYMMDD |
| shop_country_code | char(2) | |
| origin_city_code | char(3) | |
| destination_city_code | char(3) | |
| depart_date | int | |
| return_date | int | |
| advance_purchase | int | |
| length_of_stay | int | |
| carrier_code | char(3) | |
| stops | int | |
| d_min / y_min / p_min / j_min / f_min | float8 | Min price per cabin class |
| price | float8 | Representative price |
| price_points | varchar(65535) | JSON price distribution |
| itinerary_count | bigint | |
| outbound_fare / inbound_fare | float8 | |
| outbound_legs / inbound_legs | int | |
| outbound_mktg_carrier_codes | varchar(32) | |
| timestamp | bigint | |
| _(+ ~70 more leg-detail columns)_ | | |

_Partition keys: `observation_date`, `pos_country_code`_

---

### neo-price-itinerary-loader

**Type**: EMR Spark Job (Java)
**Trigger**: Standalone — defaults to previous hour; can accept `<date> <hour>` arguments or `velocity` test mode
**Compute**: EMR cluster; driver memory 24 GB

**What it does**: Reads raw estream search-with-itinerary parquet records for a specific hour, enriches them with city-to-country mapping from `MetadataReader`, and builds `NeoPriceQuery` objects used for Neo4j price look-ups. Each query captures origin, destination, airline, and pricing context. Writes the resulting query objects as parquet partitioned by date and hour.

**Input**:
- S3: `s3://s3-atp-3victors-3vprod-use1-dataset-ingest/estream/search-with-itineraries/v1/{year}/{month}/{day}/{hour}/`
- Format: Parquet (`RawSearch` objects)

**Output**:
- S3: `s3://dataset-neo-price-itinerary-loader-parquet/v1/{year}/{month}/{day}/{hour}/`
- Format: Parquet (`NeoPriceQuery` objects)

---

### flight-schedule-harvest

**Type**: EMR Spark Job (Java)
**Trigger**: Standalone — accepts `<date>` (daily) or `<date> <time>` (hourly, time rounded to nearest 100)
**Compute**: EMR cluster; driver memory 24 GB

**What it does**: Harvests flight schedule information (routes, departure/arrival times, equipment codes) from raw search-with-itinerary Avro files. When a time argument is provided, processes a single hour's slice; when omitted, aggregates the full 24-hour day. Writes `FlightScheduleHarvest` records as parquet.

**Input**:
- S3: `s3://search-with-itineraries-avro/v1/{year}/{month}/{day}/` (daily) or `.../v1/{year}/{month}/{day}/{hour}/` (hourly)
- Format: Avro (`FlightInfo` records)

**Output**:
- S3: `s3://dataset-flight-schedule-harvest-parquet/v1/{year}/{month}/{day}/` (daily) or `.../v1/{year}/{month}/{day}/{hour}/` (hourly)
- Format: Parquet

---

### tax-model-data-generator

**Type**: EMR Spark Job (Java)
**Trigger**: Standalone, hourly — defaults to current UTC time minus 1 hour
**Compute**: EMR cluster; driver memory 24 GB

**What it does**: Reads hourly representative itinerary parquet files for the previous 24-hour window and generates `FlattenedTaxRow` records containing tax metadata (base fare, tax, surcharge components) used for downstream tax modeling. Writes hourly-partitioned parquet to the configured output bucket.

**Input**:
- S3: `s3://{input.bucket.name}/{input.version}/{year}/{month}/{day}/{hour}/` (24 hours of hourly data)
- Config: `EMR-TaxModelDataSparkJob.properties`

**Output**:
- S3: `s3://{output.bucket.name}/{output.version}/{year}/{month}/{day}/{hour}/`
- Format: Parquet

---

### tax-regression

**Type**: EMR Spark Job (Java)
**Trigger**: Standalone, daily — defaults to current date minus 2 days; observes past 8 days of data
**Compute**: EMR cluster; driver memory 24 GB

**What it does**: Correlates daily representative itinerary data with historical basic-detail records from Redshift (8-day lookback) to fit per-route tax regression models. Produces `TaxRegressionData` records containing regression coefficients (slope `m`, intercept `b`, R², correlation) by point-of-sale, OD pair, cabin, carrier, and stop count. Optionally registers the output partition in the Glue catalog.

**Input**:
- S3: `s3://{input.bucket.name}/{input.version}/{year}/{month}/{day}/` (daily rep itinerary parquet)
- Redshift: `CommonOutputBasicDetail` records via `PriceEyeRedshiftDemoReader` (8-day window)
- Config: `EMR-TaxRegressionSparkJob.properties`

**Output**:
- S3: `s3://s3-atp-3victors{env}-use1-ds-tax-regression/v2/{year}/{month}/{day}/`
- Glue table: `tax_regression_v2` (partition registered via partition_creator)
- Redshift Spectrum: `data_lakes.tax_regression_v2`

**Table Schema** (tax_regression_v2):

| Column | Type | Notes |
|--------|------|-------|
| pos | varchar(3) | Point of sale |
| od | varchar(6) | Origin+destination city pair |
| is_one_way | boolean | |
| search_class | varchar(32) | Cabin class |
| carrier | varchar(3) | |
| currency | varchar(3) | |
| nbr_outbound_stop | smallint | |
| nbr_inbound_stop | smallint | |
| ct | int | Sample count |
| minx / x_bar / maxx | float8 | Independent variable stats (fare) |
| miny / y_bar / maxy | float8 | Dependent variable stats (tax) |
| m | float8 | Regression slope |
| b | float8 | Regression intercept |
| r2 | float8 | R-squared |
| correlation | float8 | |
| added_at | bigint | Timestamp |

_Partition keys: `sales_date`_

---

### historical-data

**Type**: ECS Fargate Task (Java — `wn-historical`)
**Trigger**: Standalone (manual or separate schedule — no EventBridge rule in repo)
**Compute**: 2048 MB memory, 1024 CPU units; Linux ARM64; 200 GB ephemeral storage
**Docker Image**: `3victors/wnhistorical/wn-historical`

**What it does**: Converts PriceEye common output parquet files (Southwest Airlines / WN brand) into gzipped CSV files in the historical itinerary format used for offline reporting and analysis. Reads `PECommonOutput` parquet records, loads airport-to-timezone and site metadata, deduplicates by outbound/inbound flight and fare-family combination within each `RequestId`, formats using `HistoricalPEItineraryFormatter`, and writes one itinerary per line as gzipped CSV. Supports multiple source path variants based on sales date ranges to handle historical schema transitions.

**Input** (date-dependent path):
- `salesDate ≤ 20240912`: `s3://price-eye-common-output/WN/all/{year}/{month}/{day}/`
- `20240912 < salesDate ≤ 20240926`: `s3://price-eye-common-output/WN/WN_oneway/` and `WN_roundtrip/{year}/{month}/{day}/`
- `20240926 < salesDate ≤ 20241009`: `s3://price-eye-common-output/WN/WN_perf_test/{year}/{month}/{day}/`

**Output**:
- S3: `s3://3v-wn-historical-{env}/{year}/{month}/{day}/`
- Format: Gzipped CSV text files (3VFormatV2)

---

## Glue Databases & Tables

All tables live in the `glue-atp-3victors-{env}-use1-data_lakes_db` Glue database (also accessible as `data_lakes` Redshift Spectrum schema). Partitions are registered automatically via the `partition_creator` Aurora RDS service.

| Table | S3 Location | Partition Keys | Producer |
|-------|-------------|----------------|---------|
| `daily_representative_itinerary_v4` | `s3-atp-3victors{env}-use1-ds-daily-rep-parquet/v4/` | `observation_date`, `pos_country_code` | daily-representative-itinerary-json |
| `demand_debot_v1` | `s3-atp-3victors{env}-use1-ds-demand-debot/v1/` | `sales_date` | debot-demand |
| `demand_debot_estream` | `s3-atp-3victors{env}-use1-ds-demand-debot-estream/v1/` | `sales_date` | debot-demand-estream |
| `demand_debot_rollup_v1` | `s3-atp-3victors{env}-use1-ds-demand-debot-rollup/v1/` | `sales_date` | _(rollup, downstream)_ |
| `demand_debot_rollup_estream` | `s3-atp-3victors{env}-use1-ds-demand-debot-rollup-estream-p/v1/` | `sales_date` | _(rollup, downstream)_ |
| `demand_summary_v1_parquet` | `s3-atp-3victors{env}-use1-dataset-demand-summary-parquet/v1/` | `sales_date`, `sales_hour` | _(upstream ingestion)_ |
| `demand_rollup_v2` | `s3-atp-3victors{env}-use1-ds-demand-rollup/v2/` | `sales_date` | _(rollup, downstream)_ |
| `demand_rollup_estream_v1` | `s3-atp-3victors{env}-use1-ds-demand-rollup-estream/v1/` | `sales_date` | _(rollup, downstream)_ |
| `tax_regression_v2` | `s3-atp-3victors{env}-use1-ds-tax-regression/v2/` | `sales_date` | tax-regression |

---

## Infrastructure Summary

| Resource | Count | Details |
|----------|-------|---------|
| EMR Clusters | 2 types | Hourly (16 instances), Daily (19–32 instances by zone) |
| EMR Release | — | `emr-7.5.0` |
| ECS Fargate Tasks | 1 | `wn-historical` (2 GB / 1 vCPU / ARM64) |
| Step Functions | 9 | EMR-Hourly-Cluster, EMR-Daily-Cluster, EMR-Create-Cluster, EMR-Add-Step, EMR-Add-Hourly-Steps, EMR-Daily-Cluster-Zone-Add-Steps, EMR-Create-Capacity-Reservation, EMR-Error-Processing, 3Victors-General-Failure-Handling |
| EventBridge Rules | 2 | `EMR-Hourly-Initiate` (cron), `Daily-Cluster-Initiation` (event-driven) |
| S3 Buckets | 5+ | daily-rep, demand-debot, demand-debot-estream, demand-debot-rollup, demand-debot-rollup-estream-p |
| Glue Tables | 9 | See table above |
| Redshift Spectrum Views | 9 | `data_lakes.*` schema, mirroring Glue tables |
| EC2 Instance Types Used | 4 | r7g.4xlarge, r6g.8xlarge, r6gd.8xlarge, m6g.8xlarge (ARM64) |
| Spark Serialization | — | Kryo (all jobs) |
| EMR Step Driver Memory | — | 24 GB (all jobs) |

---

## Key Notes

- **Zone-Based Daily Processing**: The daily cluster is created once per zone (5 zones) with zone-specific instance counts. Each zone processes a different UTC hour window to capture the full local-midnight-to-midnight shopping day for that region.
- **Capacity Reservation Strategy**: Before creating any EMR cluster, the pipeline loops through 4 instance types × 3 AZs to secure capacity, then releases the reservation after the cluster is running.
- **Demand Filtering**: Both `debot-demand` and `debot-demand-estream` apply the same 6-stage filter chain (bot, spike, volume, carrier, fanout, split-ticket) but operate on different input datasets (batch vs. estream).
- **Campaign-Driven TMO**: `tmo-replacement` is data-driven — campaigns (market pairs, constraints) are loaded from RDS at runtime, making it configurable without code changes.
- **Partition Registration**: Glue partitions for all tables are registered via an external `partition_creator` Aurora RDS service rather than inline by the Spark jobs themselves. The `spectrum-tables.sql` in `docs/` contains the corresponding Redshift Spectrum DDL.
- **Error Handling**: All failures flow through `EMR-Error-Processing` → `3Victors-General-Failure-Handling` → Slack `#Operations` webhook.
- **ARM64 Throughout**: All EMR instance types (Graviton) and the ECS task (`wn-historical`) run on ARM64.
