# emr

> A multi-module Maven/Java library of Apache Spark and Hadoop MapReduce jobs that transform raw airline search and booking data — from Avro and CSV on S3 — into structured Parquet datasets consumed by downstream analytics and pricing pipelines.

> **Production branch**: `master` (this document reflects the master branch, which is what runs in production)

---

## Architecture Overview

```
[External trigger / manual launch]
         │
         ▼
[EmrClusterLauncher (common)]  ←── AutoDeployer ──► s3://3v-emr-deployment/
         │
         │  spins up EMR cluster (emr-5.17.0, Spark 2.4.4 / Hadoop 2.8.4)
         │  instance fleet: c4.2xl / c4.4xl / c4.8xl spot + m5.4xl master
         │
         ├──────────────── PRIMARY PIPELINE (spark module) ────────────────────────────
         │
         │  [AvroToParquet]
         │    reads ◄── s3://search-with-itineraries-avro/v1/{yyyy}/{mm}/{dd}/{hh}/
         │    writes ──► <output>/{yyyy}/{mm}/{dd}/{hh}/  (partitioned by POS country)
         │
         │  [AmadeusCSVToParquet]
         │    reads ◄── s3://3v-amadeus/{yyyy}/{mm}/{dd}/  (^ pipe-delimited CSV)
         │    writes ──► <output>/{yyyy}/{mm}/{dd}/         (parquet, partitioned by POS)
         │
         │  [HourlyCSVToParquet]
         │    reads ◄── hourly itinerary CSV (FlattenedHourlyItinerarySummary format)
         │    writes ──► <output>/  (sorted parquet)
         │
         │  [DailyCSVToParquet]
         │    reads ◄── s3://dataset-daily-itinerary-summary-csv/v3/{yyyy}/{mm}/{dd}/
         │    writes ──► <output>/{yyyy}/{mm}/{dd}/  (partitioned by shop_country_code)
         │
         │  [HourlyToDaily]
         │    reads ◄── s3://dataset-hourly-itinerary-summary-csv/v2/{yyyy}/{mm}/{dd}/{hh}/
         │           +  Redshift metadata (city-country, market-distance maps)
         │    writes ──► <output>/{yyyy}/{mm}/{dd}/  (daily aggregated parquet)
         │
         │  [DemandCSVToParquet]
         │    reads ◄── <input>/{yyyy}/{mm}/{dd}/  (FlattenedDemandV2 CSV)
         │    writes ──► <output>/{yyyy}/{mm}/{dd}/  (partitioned by shop_country_code)
         │
         │  [DedupConverter]
         │    reads ◄── parquet demand (v3/ prefix, 24 hours per day)
         │           or  Redshift: spectrum_schema.demand_v3_all_pos
         │    applies → CarrierFilter → FanoutFilter → SplitTicketFilter
         │            → BotFilter → VolumeFilter → SpikeFilter (STL decomposition)
         │    writes ──► <output>/{yyyy}/{mm}/{dd}/  (DemandSummary parquet)
         │
         │  [FareConstruction]
         │    reads ◄── Avro RawSearch (ESTREAM source only)
         │    writes ──► <output>/  (CSV: route, price, tax, fare construction, tax ladder)
         │
         │  [TimeseriesDriver]
         │    reads ◄── s3://3v-gww/poo-us/  (TimeseriesRawData CSV)
         │    writes ──► <output>/  (parquet partitioned by year/month/day)
         │
         │  [SparkHourlyItinerarySummary]
         │    reads ◄── s3://search-with-itineraries-avro/v1/  (Avro)
         │    writes ──► s3://3v-gww/hourly-itinerary-summary/  (CSV text)
         │
         │  [AvroToParquetSkinny]
         │    reads ◄── s3://3v-polling/dev-2180/gds/{yyyy}/{mm}/{dd}/{hh}/
         │    writes ──► s3://3v-polling/dev-2180/skinny/{yyyy}/{mm}/{dd}/{hh}/
         │
         │  [PopulateCache]
         │    reads ◄── Redshift: spectrum_internal.dev_2180_skinny
         │    writes ──► Redis: itinerary-cache.1doyu3.0001.use1.cache.amazonaws.com:6379
         │
         ├──────────────── LEGACY HADOOP MR MODULES (commented out in root pom) ───────
         │
         │  [AASummaryJob]       reads Avro RawSearch ──► gzipped CSV summaries
         │  [TicketBuildingJob]  reads ARC coupon text ──► ArcTicket Avro
         │  [TicketSummaryJob]   reads ArcTicket Avro  ──► summarized Avro
         │  [RollupHourlyJob]    reads hourly CSVs     ──► rolled-up output
         │  [WNPopulate]         reads Redshift hourly_itinerary + demand tables
         │                       writes ──► s3://3v-wn/  (unload) then reloads to Redshift
         │
         └──────────────────────────────────────────────────────────────────────────────
```

---

## Orchestration

This repository is a **library of EMR job implementations**, not an orchestration layer. Each job is launched directly via its `ClusterLauncher` main class, which:

1. Calls `AutoDeployer` to sync the JAR to `s3://3v-emr-deployment/` (if changed)
2. Provisions an EMR cluster (via AWS SDK) with instance fleets
3. Submits a Hadoop/Spark step and terminates the cluster on completion

Scheduling and sequencing of these jobs is managed externally (see `spark-v3` repo or step function definitions in the orchestrating repo).

**EMR cluster defaults** (from `EmrClusterLauncher`):

| Role | Instance types | Capacity | Provisioning |
|------|---------------|----------|-------------|
| Master | m5.4xlarge | 1 | On-demand |
| Core | c4.2xl / c4.4xl / c4.8xl | 4 (on-demand) | On-demand |
| Task | c4.2xl / c4.4xl / c4.8xl | 400 spot units | Spot (50% bid), terminates after 30m timeout |

EMR release: `emr-5.17.0` · Spark: `2.4.4` · Scala: `2.11` · Hadoop: `2.8.4`

Logs: `s3://3victors-hadoop/logs/{yyyy}/{mm}/{dd}/`

---

## Components

_(Ordered by data pipeline stage — raw ingestion first, enrichment and analytics last.)_

---

### AvroToParquet

**Module**: `source/spark`
**Class**: `com.threevictors.emr.avrotoparquet.AvroToParquet`
**Type**: Spark job (EMR)
**Trigger**: Manual / external orchestration, per date range

**What it does**: Reads raw `RawSearch` Avro records (one folder per hour) from the primary search capture bucket and converts them to GZIP-compressed Parquet, filtered to a configurable set of point-of-sale countries. Partitions output by `pointOfSaleCountryCode` and assigns a random `partitionNumber` per country (US gets up to 256 partitions, HK 128, GB/GR 64, etc.) to control downstream parallelism. Sorts within each partition by `originCityCode`, `destinationCityCode`, `departDate`, `returnDate`.

**Input**:
- S3: `s3://search-with-itineraries-avro/v1/{yyyy}/{mm}/{dd}/{hh}/` (Avro, `RawSearch` schema)

**Output**:
- S3: `<output-path>/{yyyy}/{mm}/{dd}/{hh}/` (Parquet, partitioned by `pointOfSaleCountryCode`)

**Key schema** (`RawSearch`):

| Field | Type |
|-------|------|
| timestamp | long |
| source | string |
| originCityCode | string |
| destinationCityCode | string |
| departDate | int (yyyyMMdd) |
| returnDate | int (yyyyMMdd, 0 = one-way) |
| advancePurchase | int |
| lengthOfStay | int |
| pointOfSaleCountryCode | string |
| itineraries | list of RawItinerary (with RawLeg list) |
| gds | string (nullable) |
| pcc | string (nullable) |
| restricted | boolean (nullable) |

---

### AmadeusCSVToParquet

**Module**: `source/spark`
**Class**: `com.threevictors.emr.amadeus.AmadeusCSVToParquetConverter` (driver: `AmadeusCSVToParquetDriver`)
**Type**: Spark job (EMR)
**Trigger**: Per shop-date, manual / external

**What it does**: Reads Amadeus GDS raw search data from S3 in pipe-delimited CSV format (`^` separator, 133 columns per record, `RawAmadeusRecord` schema). Groups rows by `transactionId` to consolidate multi-leg itineraries into `RawSearch` objects, resolving airport codes to city codes via a broadcast metadata map. Filters to single adult passenger searches, validates origin/destination city consistency, and writes as Parquet partitioned by `pointOfSaleCountryCode`.

**Input**:
- S3: `s3://3v-amadeus/{yyyy}/{mm}/{dd}/` (pipe-delimited CSV, `RawAmadeusRecord` format)
- Redshift metadata (airport-to-city mapping, loaded via `Metadata` class at job startup)

**Output**:
- S3: `<output-path>/{yyyy}/{mm}/{dd}/` (Parquet `RawSearch` format, partitioned by `pointOfSaleCountryCode`)

---

### HourlyCSVToParquet

**Module**: `source/spark`
**Class**: `com.threevictors.emr.parquet.HourlyCSVToParquetConverter` (driver: `HourlyCSVToParquetDriver`)
**Type**: Spark job (EMR)
**Trigger**: Manual / external, per hourly CSV batch

**What it does**: Reads hourly itinerary summary CSV files in `FlattenedHourlyItinerarySummary` schema and converts them to GZIP Parquet. The 39-column schema covers shop metadata (date, hour, country, origin/destination city/airport, cabin, carrier, stops), itinerary counts, cheapest price/tax/surcharge/duration/booking codes, and fastest itinerary equivalents. Sorts within partitions by origin, destination, depart date, return date.

**Input**:
- S3 / local: path to hourly CSV files (`FlattenedHourlyItinerarySummary` schema)

**Output**:
- S3: `<output-path>/` (Parquet, sorted)

**Key columns** (`FlattenedHourlyItinerarySummary`):

| Column | Description |
|--------|-------------|
| shop_date, shop_hour | Shopping date/hour |
| shop_country_code | Point-of-sale country |
| origin_city_code, destination_city_code | Market O&D |
| origin_airport_code, destination_airport_code | Airport O&D |
| carrier_code, cabin_code, stops, codeshare | Itinerary dimensions |
| depart_date, return_date, advance_purchase, length_of_stay | Trip dates |
| itinerary_count, stream_count | Observation counts |
| price, tax, surcharge, duration, booking_codes | Cheapest itinerary |
| fastest_price, fastest_tax, fastest_surcharge, fastest_duration, fastest_booking_codes | Fastest itinerary |

---

### DailyCSVToParquet

**Module**: `source/spark`
**Class**: `com.threevictors.emr.dailycsvtoparquet.DailyCSVToParquetConverter` (driver: `DailyCSVToParquetDriver`)
**Type**: Spark job (EMR)
**Trigger**: Per shop-date range, manual / external

**What it does**: Reads daily itinerary summary CSV files (already aggregated to day granularity) from S3 in `FlattenedDailyItinerarySummary` format and converts them to Parquet. Assigns a random partition number per country code (US: up to 48 partitions, HK: 20, GB/GR: 16, etc.) to balance output file sizes. Partitions the Parquet output by `shop_country_code`.

**Input**:
- S3: `s3://dataset-daily-itinerary-summary-csv/v3/{yyyy}/{mm}/{dd}/` (`FlattenedDailyItinerarySummary` CSV)

**Output**:
- S3: `<output-path>/{yyyy}/{mm}/{dd}/` (Parquet, partitioned by `shop_country_code`)

**Key schema** (`FlattenedDailyItinerarySummary`):

| Column | Type | Description |
|--------|------|-------------|
| version | int | Schema version (3) |
| shop_date | int | Shopping date (yyyyMMdd) |
| shop_country_code | string | Point-of-sale country |
| origin_city_code | string | Origin city |
| destination_city_code | string | Destination city |
| origin_airport_code | string | Origin airport |
| destination_airport_code | string | Destination airport |
| depart_date | int | Departure date |
| return_date | int | Return date (0 = one-way) |
| advance_purchase | int | Days before departure |
| length_of_stay | int | Trip duration in days |
| cabin_code | string | Cabin class |
| carrier_code | string | Marketing carrier |
| stops | int | Number of stops |
| codeshare | string | Codeshare indicator |
| itinerary_count | long | Total itinerary observations |
| origin_country_code | string | Origin country (enriched) |
| destination_country_code | string | Destination country (enriched) |
| distance | int | Route distance in miles |
| price | double | Cheapest total price |
| tax | double | Taxes on cheapest |
| surcharge | double | Surcharges on cheapest |
| duration | string | Duration of cheapest |
| booking_codes | string | Booking codes of cheapest |
| cheapest_stops | string | Stop details of cheapest |
| fastest_price | double | Price of fastest itinerary |
| fastest_tax | double | Taxes of fastest |
| fastest_duration | string | Duration of fastest |
| fastest_booking_codes | string | Booking codes of fastest |
| fastest_stops | string | Stop details of fastest |
| partition_number | int | Internal partition hint |

---

### HourlyToDaily

**Module**: `source/spark`
**Class**: `com.threevictors.emr.hourlytodailyparquet.HourlyToDailyParquetConverter` (driver: `HourlyToDailyParquetDriver`)
**Type**: Spark job (EMR)
**Trigger**: Per shop-date range, manual / external

**What it does**: Aggregates 24 hourly CSV files (one per hour per shop-date) into a single daily Parquet file. For each unique combination of shop dimensions (market, carrier, cabin, stops, etc.), selects the cheapest price (minimum `price`, tie-broken by shortest `duration`) and the fastest itinerary (minimum `duration`, tie-broken by lowest `price`) across all hourly snapshots. Enriches with city-country and market-distance lookup maps broadcast from Redshift at startup. Writes partitioned by `shop_country_code`.

**Input**:
- S3: `s3://dataset-hourly-itinerary-summary-csv/v2/{yyyy}/{mm}/{dd}/{hh}/` (24 hourly CSVs per day)
- Redshift: city-to-country map and market-distance map (loaded once at startup, broadcast)

**Output**:
- S3: `<output-path>/{yyyy}/{mm}/{dd}/` (Parquet `FlattenedDailyItinerarySummary`, partitioned by `shop_country_code`)

---

### DemandCSVToParquet

**Module**: `source/spark`
**Class**: `com.threevictors.emr.demandcsvtoparquet.DemandCSVToParquetConverter` (driver: `DemandCSVToParquetDriver`)
**Type**: Spark job (EMR)
**Trigger**: Per date range, manual / external

**What it does**: Reads demand summary CSV files in `FlattenedDemandV2` format (74+ columns covering per-market shopping counts, price breakdowns by stop count, carrier lists, and cabin type distributions) and converts them to Parquet partitioned by `shop_country_code`. The demand schema captures raw GDS shopping sessions before deduplication; the job normalises boolean hint fields and adds empty marketing carrier and itinerary-per-carrier-stops columns for schema compatibility.

**Input**:
- S3: `<input-path>/{yyyy}/{mm}/{dd}/` (FlattenedDemandV2 CSV, ~74 columns)

**Output**:
- S3: `<output-path>/{yyyy}/{mm}/{dd}/` (Parquet, partitioned by `shop_country_code`)

---

### DedupConverter

**Module**: `source/spark`
**Class**: `com.threevictors.emr.dedup.DedupConverter` (driver: `DedupDriver`)
**Type**: Spark job (EMR)
**Trigger**: Per sales-date, manual / external

**What it does**: Removes duplicate and bot-generated demand observations from the raw shopping data through a multi-stage filter pipeline. Each stage is independently configurable for debugging:

1. **CarrierFilter** — removes duplicate search records within the same origin/destination/carrier key
2. **FanoutFilter** — removes fanout searches (same session querying many markets simultaneously)
3. **SplitTicketFilter** — suppresses split-ticket search artifacts
4. **BotFilter** — identifies and removes bot traffic based on session velocity
5. **VolumeFilter** — removes volume-inflated observations
6. **SpikeFilter** — applies STL (Seasonal-Trend decomposition) to detect and remove price spikes

Supports three input modes: S3 Parquet (`s3`), Redshift JDBC (`jdbc`), and CSV (`csv`). Outputs either a `DemandSummary` Parquet (search count per market/trip) or a `SpikeFilterStatistics` debug CSV.

**Input**:
- S3: `<input-path>/v3/{yyyy}/{mm}/{dd}/{hh}/` (FlattenedDemand Parquet, 24 hours per day)
- or Redshift: `spectrum_schema.demand_v3_all_pos` (via JDBC, for debug)
- or CSV: local/S3 CSV files

**Output**:
- S3: `<output-path>/{yyyy}/{mm}/{dd}/` (Parquet `DemandSummary`, sorted by origin/destination/depart/return)
- or `<output-path>/output.csv` / `spikedebug.csv` (debug modes)

**DemandSummary schema**:

| Column | Type | Description |
|--------|------|-------------|
| origin_city_code | string | Origin city |
| destination_city_code | string | Destination city |
| depart_date | int | Departure date |
| return_date | int | Return date |
| searches | int | Deduplicated search count |

---

### FareConstruction

**Module**: `source/spark`
**Class**: `com.threevictors.emr.fareconstruction.FareConstructionDriver`
**Type**: Spark job (EMR)
**Trigger**: Manual / external

**What it does**: Extracts fare construction details from raw `RawSearch` Avro files, filtering to ESTREAM source searches only. For each itinerary, extracts the origin/destination airports, depart/return dates, validating carrier, total price, taxes, combined outbound+inbound fare construction text, and tax ladder breakdown. Writes the result as CSV.

**Input**:
- S3: `<input-path>/{hh}/` (24 hourly Avro `RawSearch` files per day)

**Output**:
- S3: `<output-path>/` (CSV: `originAirportCode`, `destinationAirportCode`, `departDate`, `returnDate`, `carrierCode`, `price`, `taxes`, `fareConstruction`, `taxLadder`)

---

### TimeseriesDriver

**Module**: `source/spark`
**Class**: `com.threevictors.emr.timeseries.TimeseriesDriver`
**Type**: Spark job (EMR)
**Trigger**: Manual / external

**What it does**: Builds a historical pricing time-series dataset by reading raw pricing CSV files from S3, grouping by market+carrier+trip key (origin airport, destination airport, depart date, return date, carrier, stops), and collecting the price indexed by shop date into a map. Enriches each record with calendar columns (year, month, day of depart date) and writes as Parquet partitioned by year/month/day.

**Input**:
- S3: `s3://3v-gww/poo-us/` (all files with prefix, `TimeseriesRawData` CSV schema)

**Output**:
- S3: `<output-path>/` (Parquet, partitioned by `year`/`month`/`day`)

**TimeseriesData schema**:

| Column | Type | Description |
|--------|------|-------------|
| originAirportCode | string | Origin airport |
| destinationAirportCode | string | Destination airport |
| departDate | int | Departure date |
| returnDate | int | Return date |
| carrierCode | string | Carrier |
| stops | int | Stops |
| pricing | map<int,double> | shop_date → price |

---

### SparkHourlyItinerarySummary

**Module**: `source/spark`
**Class**: `com.threevictors.emr.spark.SparkHourlyItinerarySummary` (driver: `SparkClusterDriver`)
**Type**: Spark job (EMR)
**Trigger**: Manual / external (one Avro hour at a time)

**What it does**: Reads raw `RawSearch` Avro files using the `SpecificAvroKeyInputFormat` and builds hourly itinerary summaries using `ItineraryHourlySummaryBuilder`. The builder flattens each search into key-value pairs keyed by market+itinerary dimensions, then reduces by summing counts and taking the minimum price. Outputs as text CSV (itinerary summary format).

**Input**:
- S3: `s3://search-with-itineraries-avro/v1/{yyyy}/{mm}/{dd}/{hh}/*.avro`

**Output**:
- S3: `s3://3v-gww/hourly-itinerary-summary/<timestamp>/` (text CSV)

---

### AvroToParquetSkinny

**Module**: `source/spark`
**Class**: `com.threevictors.emr.avrotoparquet.AvroToParquetSkinny`
**Type**: Spark job (EMR)
**Trigger**: Manual / experimental

**What it does**: Reads polling GDS Avro data and creates a "skinny" flattened view with one row per itinerary (rather than per search session). Explodes the itinerary list, extracts outbound/inbound leg details, builds a route key string, and records price, tax, tax ladder, and duration. Writes as Parquet sorted by airport O&D and trip dates. This appears to be a development/research job based on the hardcoded date range.

**Input**:
- S3: `s3://3v-polling/dev-2180/gds/{yyyy}/{mm}/{dd}/{hh}/` (Avro, `RawSearch` schema)

**Output**:
- S3: `s3://3v-polling/dev-2180/skinny/{yyyy}/{mm}/{dd}/{hh}/` (Parquet `Skinny` schema)

**Skinny schema**:

| Column | Type |
|--------|------|
| originAirportCode | string |
| destinationAirportCode | string |
| departDate | int |
| returnDate | int |
| routeKey | string |
| duration | int |
| totalPrice | float |
| taxes | float |
| taxLadder | string |
| outboundLegs | string (JSON) |
| inboundLegs | string (JSON) |

---

### PopulateCache

**Module**: `source/spark`
**Class**: `com.threevictors.emr.itinerarycache.PopulateCache`
**Type**: Standalone Java program (not an EMR Spark job)
**Trigger**: Manual

**What it does**: Reads skinny itinerary records from Redshift (`spectrum_internal.dev_2180_skinny`) via JDBC and streams Redis `SET` commands to stdout in RESP protocol format. The intent is to pipe the output into `redis-cli --pipe` to bulk-load itinerary data into the ElastiCache Redis cluster. Connects to Redshift with a fetch size of 100,000 rows.

**Input**:
- Redshift: `demo.3victorsaws.com:5439/demo` → `spectrum_internal.dev_2180_skinny`

**Output**:
- Redis (stdout/pipe): `itinerary-cache.1doyu3.0001.use1.cache.amazonaws.com:6379/0`
  Key: `{originAirportCode}{destinationAirportCode}{departDate}{returnDate}{routeKey}`
  Value: JSON `SkinnyItinerary` object

---

## Legacy Components (commented out in root `pom.xml`)

These modules exist in the source tree but are not built by the root POM. They represent earlier Hadoop MapReduce implementations that have been superseded by the Spark jobs above.

### AASummaryJob (`source/aa-data`)

**Type**: Hadoop MapReduce
**What it does**: Processes `RawSearch` Avro files for American Airlines demand analysis. Reads from S3 Avro input (date-range path expansion) and outputs gzipped CSV summaries keyed by `AASummaryKey` (shop date, origin, destination, depart date, advance purchase, time-of-day, cabin). Task fleet: 3,600 spot units; Core fleet: 16.

**Input**: S3 Avro `RawSearch` path(s)
**Output**: Gzipped CSV (`AASummaryData` format)

---

### TicketBuildingJob + TicketSummaryJob (`source/booking-data`)

**Type**: Hadoop MapReduce
**What it does**: Processes ARC (Airline Reporting Corporation) coupon text files. `TicketBuildingJob` parses raw coupon text (mapper: `TicketBuildingMapper`) and groups by ticket number (reducer: `TicketBuildingReducer`) to produce `ArcTicket` Avro output. `TicketSummaryJob` further reduces to ticket-level summaries. 300 reducers. Task fleet: 100 spot units.

**Input**: S3 text files (ARC coupon format)
**Output**: Avro `ArcTicket` records

---

### RollupHourlyJob / MonthlyLosJob / UnitedOneWayNonstopExtractJob / KoreaJob (`source/shopping-data`)

**Type**: Hadoop MapReduce
**What it does**: Various market-specific shopping data processing jobs. `RollupHourlyJob` aggregates hourly itinerary data. `MonthlyLosJob` computes monthly length-of-stay distributions. `UnitedOneWayNonstopExtractJob` extracts United Airlines one-way nonstop itineraries. `KoreaJob` processes Korean market fare data with validation and pruning logic. Task fleet: 1,200 spot units; Core fleet: 16.

---

### WNPopulate (`source/wn`)

**Type**: Standalone Java JDBC program
**What it does**: Populates Southwest Airlines pricing analysis tables in Redshift. For each date: truncates staging tables (`wn_sample`, `wn_demand`, `wn_cheapest`, `wn_fastest`), pulls hourly itinerary data for WN routes filtered by US POS and specific block times/product codes, inserts demand counts, calculates cheapest and fastest price variants, UNLOAD the joined result to `s3://3v-wn/{yyyy}/{mm}/{dd}/`, then COPYs into `wn_7_day_history` in the southwest database.

**Input**:
- Redshift `demand.3victorsaws.com:5439/demand`: `spectrum_schema.hourly_itinerary`, `demand_ow_201810`
- Reference tables: `wn_routes`, `wn_carriers`, `wn_blocktime`, `wn_product`

**Output**:
- S3: `s3://3v-wn/{yyyy}/{mm}/{dd}/` (intermediate unload)
- Redshift `demand.3victorsaws.com:5439/southwest`: `wn_7_day_history`

---

### ML Preprocess (`source/ml`)

**Type**: Standalone local Java program
**What it does**: Experimental preprocessing for ML model input. Reads a local CSV file of itinerary data, builds a booking-code-to-price map indexed by carrier/departure date/advance purchase, constructs fare ladders, and outputs four CSV files: `bycode.csv`, `byprice.csv`, `byladder.csv`, `bydifferential.csv`. Used for analysing booking code ladder ordinals vs. price movements.

**Input**: Local CSV file (itinerary format with booking codes and pricing)
**Output**: Local CSV files (`bycode`, `byprice`, `byladder`, `bydifferential`)

---

## Common Infrastructure (`source/common`)

### EmrClusterLauncher

Abstract base class for all EMR cluster launcher programs. Handles cluster provisioning (instance fleet configuration for master, core, task nodes), Spark/YARN configuration, and job submission via the AWS EMR SDK. Subclasses implement `getStepConfigs()` to define the Hadoop/Spark steps to run. Supports a `--noautodeploy` flag to skip JAR upload to S3.

### AutoDeployer

Compares the local JAR against the version in `s3://3v-emr-deployment/` by size and MD5 hash. If they differ, deletes the old version and uploads the new JAR. Provides the JAR name used in EMR step configs.

---

## Key Data Stores

| Store | Endpoint | Usage |
|-------|----------|-------|
| S3 (primary Avro input) | `s3://search-with-itineraries-avro/v1/` | Raw search Avro files |
| S3 (Amadeus input) | `s3://3v-amadeus/` | Amadeus GDS CSV |
| S3 (hourly summary CSV) | `s3://dataset-hourly-itinerary-summary-csv/v2/` | Hourly itinerary summaries |
| S3 (daily summary CSV) | `s3://dataset-daily-itinerary-summary-csv/v3/` | Daily itinerary summaries |
| S3 (polling/GDS) | `s3://3v-polling/dev-2180/gds/` | Polling Avro |
| S3 (skinny output) | `s3://3v-polling/dev-2180/skinny/` | Skinny parquet |
| S3 (GWW) | `s3://3v-gww/` | Hourly summaries, timeseries |
| S3 (WN) | `s3://3v-wn/` | Southwest unload data |
| S3 (deployment) | `s3://3v-emr-deployment/` | Deployed JARs |
| Redshift (demo) | `demo.3victorsaws.com:5439/demo` | Demand v3, skinny spectra |
| Redshift (demand) | `demand.3victorsaws.com:5439/demand` | WN demand, hourly itinerary |
| Redshift (southwest) | `demand.3victorsaws.com:5439/southwest` | WN 7-day history |
| Redis (ElastiCache) | `itinerary-cache.1doyu3.0001.use1.cache.amazonaws.com:6379` | Itinerary cache |
| Config server | `http://config-server.3victorsaws.com/configuration` | AWS credentials, properties |

---

## Infrastructure Summary

| Resource | Details |
|----------|---------|
| Active Maven modules | 1 (`source/spark`) |
| Legacy Maven modules | 5 (`booking-data`, `shopping-data`, `aa-data`, `ml`, `wn`, commented out) |
| Spark jobs (active) | 9 (`AvroToParquet`, `AmadeusCSVToParquet`, `HourlyCSVToParquet`, `DailyCSVToParquet`, `HourlyToDaily`, `DemandCSVToParquet`, `DedupConverter`, `FareConstruction`, `TimeseriesDriver`) |
| Utility Spark jobs | 3 (`SparkHourlyItinerarySummary`, `AvroToParquetSkinny`, `PopulateCache`) |
| Hadoop MR jobs (legacy) | 6 (`AASummaryJob`, `TicketBuildingJob`, `TicketSummaryJob`, `RollupHourlyJob`, `KoreaJob`, `WNPopulate`) |
| EMR release | emr-5.17.0 |
| Spark version | 2.4.4 (Scala 2.11) |
| Hadoop version | 2.8.4 |
| S3 buckets touched | 8+ |
| Redshift clusters | 2 (`demo`, `demand`) |
| Redis clusters | 1 (ElastiCache) |
| JAR deployment bucket | `s3://3v-emr-deployment/` |
