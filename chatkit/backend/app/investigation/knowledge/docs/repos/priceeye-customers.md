# priceeye-customer

> Customer-specific extensions to PriceEye — metadata enrichment, output post-processing, compare reporting, and customer-tailored delivery for vacation package pricing data.

> **Current branch**: `develop` _(this document reflects the `develop` branch; verify against `master` for what is currently running in production)_

---

## Architecture Overview

```
[EventBridge cron (configured at deploy)]
        │
        ▼
[Step Function: vacation-post-processing-step-function]
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 1. vacation-metadata (ECS Fargate)                      │
│    reads  ◄── S3: price-eye-customer-delivery/          │
│    writes ──► MySQL: hotels, rooms, cars, mappings      │
│    writes ──► S3: 3v-upload-bucket/ (metadata export)   │
│    writes ──► S3: pe-vacation-room-type-vectors/        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼ (Parallel)
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────────┐   ┌──────────▼────────────────┐
│ 2a. vacation-hotel-  │   │ 2b. vacation-local-fees    │
│     enrich (ECS)     │   │     (ECS)                  │
│ reads  ◄── delivery  │   │ reads  ◄── delivery bucket │
│ writes ──► MySQL:    │   │ writes ──► MySQL:           │
│ refund/inv category  │   │ vacation_local_fees         │
└───────┬──────────────┘   └──────────┬─────────────────┘
        └──────────────┬──────────────┘
                       │ (both complete)
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 3. vacation-output-rewrite (ECS Fargate)                │
│    reads  ◄── S3: price-eye-customer-delivery/          │
│    reads  ◄── MySQL: curated hotel metadata             │
│    writes ──► S3: delivery-archive/ (enriched CSV)      │
│    publishes ─► SQS (delivery notifications)            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼ (Wait 300 seconds)
┌─────────────────────────────────────────────────────────┐
│ 4. vacation-compare (ECS Fargate)                       │
│    reads  ◄── S3: delivery-archive/ (rewritten files)   │
│    reads  ◄── MySQL: hotel/car mappings                 │
│    reads  ◄── Redis (intermediate state)                │
│    writes ──► S3: delivery-archive/…/vacation_compare/  │
│    publishes ─► SQS (delivery notifications)            │
└─────────────────────────────────────────────────────────┘

─── Independent / Event-Driven ────────────────────────────

[S3 event: pe-curate-hotels/*.csv]
        │
        ▼
[curate-hotels Lambda] ──► MySQL: hotel/room updates + mappings
        └── uses: Google Places API + Gemini AI

[S3 event: priceeye-alaska/*.csv.gz]
        │
        ▼
[alaska-delivery Lambda] ──► S3: per-market CSV files
        └── writes: MySQL delivery_type_queue (GDrive)

[S3 event: TUI source bucket]
        │
        ▼
[tui-delivery Lambda] ──► S3: top-20 itineraries per market (CSV/Parquet)
        └── writes: MySQL delivery_type_queue (S3 delivery)

─── Scheduled Standalone ──────────────────────────────────

[EventBridge cron (hourly)]
        │
        ▼
[advito-delivery ECS] ──► S3 archive + SQS delivery
        └── filter: cheapest-earliest per source, Advito timezone check

[EventBridge cron (hourly)]
        │
        ▼
[sk-report ECS] ──► SFTP: daily SK formatted report
        └── runs at 7am & 8am Europe/Stockholm + 12:00 UTC
```

---

## Orchestration

### Step Function: `vacation-post-processing-step-function`

- **Trigger**: EventBridge scheduled rule (cron expression passed at deploy time via `Schedule` parameter)
- **Pipeline**:
  1. `vacation-metadata` _(serial)_
  2. `vacation-hotel-enrich` ‖ `vacation-local-fees` _(parallel)_
  3. `vacation-output-rewrite` _(serial, after parallel completes)_
  4. **Wait 300 seconds**
  5. `vacation-compare` _(serial)_
- **Definition**: `source/deploy/yaml/vacation-post-processing.yaml` (inline definition, no `.asl.json`)
- **Notes**: All ECS tasks in this Step Function are launched via a shared `RunTask` Step Function state machine (which wraps `ecs:runTask.sync`). The `ResultPath: $.null` pattern is used throughout; after deployment these must be changed to `ResultPath: null`.

---

## Components

_(Ordered by when they run in the pipeline — earliest first.)_

---

### vacation-metadata

**Type**: ECS Fargate Task (ARM64)
**Trigger**: Step Function step — first in `vacation-post-processing-step-function`
**Compute**: 1024 MB RAM, 1024 CPU units (1 vCPU)

**What it does**: Orchestrates the ingestion and enrichment of all vacation metadata for the PriceEye Customer system. Reads customer-delivered CSV/TSV vacation package files from the packager archive S3 bucket and builds/updates the canonical metadata store for hotels, rooms, and cars. Uses Google Places API and Gemini AI (configurable model, default `gemini-3-flash-preview` for hotels; `gemini-2.5-flash-lite` for rooms) to enrich hotels with official names, addresses, and curation status; uses vector embeddings (stored in S3) for fuzzy room-name similarity matching across sources. Generates cross-source hotel/room/car mappings so that the same real-world entity across different vacation suppliers shares a canonical ID. Exports updated metadata tables to S3 and publishes delivery notifications via SQS.

**Input**:
- S3: `price-eye-customer-delivery/{customer}/{collection}/{date}/` (CSV/TSV vacation package files, optionally gzip)
- MySQL: existing vacation hotel/room/car metadata and source definitions
- Google Places API (hotel enrichment)
- Gemini AI API (hotel/room classification and curation)
- AWS Secrets Manager: `GooglePlaces/apikey`

**Output**:
- MySQL: `vacation_hotel`, `vacation_room`, `vacation_car` tables; cross-source hotel/room/car mapping tables
- S3: `3v-upload-bucket/` (metadata export files)
- S3: `s3-atp-3victors{env}-use1-pe-vacation-room-type-vectors/` (room name vector embeddings)
- SQS: delivery notifications for downstream consumers

**Config**: `vacation-metadata.properties`
- `packager.archive` — source S3 bucket (default: `price-eye-customer-delivery`)
- `delivery.archive` — delivery archive bucket (default: `price-eye-delivery-archive`)
- `export.bucket` — metadata export bucket (default: `3v-upload-bucket`)
- `vector.bucket` — vector DB bucket
- `matches.room.number` — max candidate room matches (default: 20)
- `matches.room.distance` — max vector distance for room match (default: 0.10)
- `classify.model` / `classify.room.model` — Gemini model names

---

### vacation-hotel-enrich

**Type**: ECS Fargate Task (ARM64)
**Trigger**: Step Function step — parallel branch A in `vacation-post-processing-step-function`
**Compute**: 1024 MB RAM, 1024 CPU units (1 vCPU)

**What it does**: Enriches the canonical hotel metadata with two properties derived from daily customer delivery files. (1) **Refundability category** — scans delivered CSV files for `hotel_refundable` field values on observations with advance purchase ≥ 15 days; updates the hotel's `refundability_category` (rfnd / nonrfnd / unknown). (2) **Inventory category** — compares competitor (comp) and OJT room pricing across matched hotel pairs using canonical room mappings; classifies hotels as `same` (comp and OJT carry the same room types) or `different`. Updates both the deep source and its shallow source counterpart in MySQL. Uses an 8-thread parse pool for parallel file processing.

**Input**:
- S3: `price-eye-customer-delivery/{customer}/{collection}/{date}/` (CSV, optionally gzip)
- MySQL: `vacation_hotel`, `vacation_room`, `vacation_source`, hotel/room mapping tables
- Config: `vacation-hotel-enrich.properties` → `packager.archive`

**Output**:
- MySQL: `vacation_hotel.refundability_category`, `vacation_hotel.inventory_category`

---

### vacation-local-fees

**Type**: ECS Fargate Task (ARM64)
**Trigger**: Step Function step — parallel branch B in `vacation-post-processing-step-function`
**Compute**: 1024 MB RAM, 1024 CPU units (1 vCPU)

**What it does**: Reads customer-delivered vacation package CSV files and extracts local fees data for each hotel. For each record that has both `price_inc` (total with local fees) and `total_without_local_fees` fields, computes the local fee as the difference, normalizes it to a per-night percentage of both the with-fees and without-fees totals, then batch-inserts `LocalFeesRecord` rows into MySQL. One record per unique (source_id, cityCode, normalizedHotelName) per run. Also propagates fees to shallow source counterparts.

**Input**:
- S3: `price-eye-customer-delivery/{customer}/{collection}/{date}/` (CSV, optionally gzip)
- MySQL: `vacation_hotel`, `vacation_source` (for hotel lookups and airport→city mapping)
- Config: `vacation-local-fees.properties` → `packager.archive`

**Output**:
- MySQL: `vacation_local_fees` (source_id, city_code, normalized_name, pct_per_day_with, pct_per_day_without)

---

### vacation-output-rewrite

**Type**: ECS Fargate Task (ARM64)
**Trigger**: Step Function step — after parallel vacation-hotel-enrich + vacation-local-fees complete
**Compute**: 1024 MB RAM, 1024 CPU units (1 vCPU), 24 write threads

**What it does**: Reads packaged vacation CSV files from the packager archive, enriches each record by looking up the hotel in the curated metadata store, and overwrites two fields in-place: `hotel_inventory_category` and `hotel_refundable_category`. Records for hotels not present in the curated metadata are dropped entirely. Supports both CSV (with optional gzip) and Parquet output formats. Uploads the rewritten files to the delivery archive bucket and publishes SQS delivery messages for the downstream delivery pipeline.

**Input**:
- S3: `{packager.archive}/{customer}/{collection}/{date}/` (CSV/Parquet, optionally gzip)
- MySQL: curated hotel metadata (inventory_category, refundability_category, source map)
- Config: `vacation-output-rewrite.properties` → `packager.archive`, `delivery.archive`, `write.pool.threads`

**Output**:
- S3: `{delivery.archive}/{customer}/{collectionId}/{date}/` (enriched CSV/Parquet, same compression as input)
- SQS: `PEDeliveryMessage` objects (up to 25 S3 keys per message)

---

### vacation-compare

**Type**: ECS Fargate Task (ARM64)
**Trigger**: Step Function step — after 300-second wait following vacation-output-rewrite
**Compute**: 1024 MB RAM, 1024 CPU units (1 vCPU)

**What it does**: Joins two vacation package data sources — a competitor (comp) source and an OJT (Own Journey Tracking) source — to produce match and exception reports. Reads rewritten delivery files from S3, normalizes origins and destinations via co-terminal maps, then correlates records by composite match keys (origin, destination, dates, hotel/car via canonical mappings, timebands, board/meal plan, occupancy). Writes separate output files for matching pairs and non-matching exceptions, partitioned by compare type (Hotels, Cars, or Combined hotel+car). Uses Redis for intermediate state. Delivers the compare files to customers via SQS.

**Input**:
- S3: `{delivery.archive}/{customer}/{collection}/{date}/` (rewritten CSV/Parquet from vacation-output-rewrite)
- MySQL: `vacation_hotel_mapping`, `vacation_car_mapping` (canonical cross-source mappings); customer/packaging configs
- Redis: intermediate compare state
- Config: `vacation-compare.properties` → `packager.archive`, output settings, co-terminal maps

**Output**:
- S3: `{delivery.archive}/{customer}/vacation_compare/{collectionId}/` (match + exception CSVs, optionally gzip), partitioned by `sales_date`
- SQS: `PEDeliveryMessage` delivery notifications

**Note**: The `docs/redshift/vacation-spectrum.sql` file defines Redshift Spectrum external tables (`swav.comp_ojt`, `swav.comp_comp`) over the `vacation_compare` output in the production delivery archive. These tables are not created by this repo but reflect its output schema.

---

### curate-hotels

**Type**: AWS Lambda Function (arm64)
**Trigger**: EventBridge S3 event — `.csv` file created in `s3-atp-3victors{env}-use1-pe-curate-hotels`
**Compute**: 2048 MB RAM, 315 second timeout

**What it does**: An operational Lambda for manually curating the hotel metadata catalog. An operator uploads a CSV to the curate-hotels bucket with rows of: `[hotel_id, curate_flag, google_name, google_address, canonical_room]`. Each field supports three modes: a literal value (to set it), `RESET` (to clear it), or `LOOKUP` (to auto-classify using Google Places API or Gemini AI). The Lambda loads the existing hotel and room records, parses the CSV using a 4-thread pool, applies the requested changes, writes updated hotels and rooms to MySQL, then regenerates hotel mappings (if curation status or Google name/address changed) and room mappings (if canonical room tier/space type changed). Requires `GooglePlaces/apikey` from Secrets Manager.

**Input**:
- S3: `s3-atp-3victors{env}-use1-pe-curate-hotels/{key}.csv` (triggering object)
- Google Places API (when `LOOKUP` in google_name/google_addr column)
- Gemini AI (hotel model: `gemini-3-flash-preview`; room model: `gemini-2.5-flash-lite`) (when `LOOKUP` in curate or canonical_room column)
- MySQL: existing `vacation_hotel`, `vacation_room` records

**Output**:
- MySQL: updated `vacation_hotel` (curated flag, google_name, google_addr); updated `vacation_room` (room_tier, space_type); regenerated hotel/room mapping records

---

### alaska-delivery

**Type**: AWS Lambda Function (arm64)
**Trigger**: EventBridge S3 event — `.csv.gz` file created in `s3-atp-3victors{env}-use1-priceeye-alaska`
**Compute**: 2048 MB RAM, 315 second timeout

**What it does**: Post-processes the Alaska Airlines vacation delivery output. When a single consolidated CSV.GZ file lands in the Alaska bucket, this Lambda reads it, splits records by market key (tokens[4] + tokens[5]), writes one CSV file per market — with a header — to the destination bucket under a date-partitioned key, then inserts a `delivery_type_queue` record per market file (status: `ready`) into MySQL so the delivery scheduler picks the files up for transfer to Google Drive. A 5-minute job-lock prevents duplicate processing if the S3 event fires twice.

**Input**:
- S3: `s3-atp-3victors{env}-use1-priceeye-alaska/{key}.csv.gz` (triggering object)
- Config: `delivery-alaska.properties` → `destinationBucket`, `destinationKeyPrefix`, `delivery_id`

**Output**:
- S3: `{destinationBucket}/{destinationKeyPrefix}/{YYYY}/{MM}/{DD}/{market}.csv` (one file per market)
- MySQL: `delivery_type_queue` (one record per market file, status: ready; used by GDrive delivery scheduler)

---

### tui-delivery

**Type**: AWS Lambda Function
**Trigger**: EventBridge S3 event — file created in the configured TUI source bucket

**What it does**: Post-processes TUI vacation package delivery output. Reads the incoming file (CSV.GZ or Parquet), groups itineraries by market key (origin, destination, outbound/inbound dates), then for each market keeps only the top 20 cheapest unique itineraries (by the composite itin key of origin + destination + dates + carriers + flight numbers), including any tied with the 20th price. Writes the filtered output as a single CSV or Parquet file to the customer delivery bucket, then queues it in `delivery_type_queue` for S3-based delivery.

**Input**:
- S3: TUI source bucket (triggering CSV.GZ or Parquet file)
- Config: `price-eye-tui-delivery.properties` → `destinationBucket`, `destinationKeyPrefix`, `delivery_id`, `customer`, `refGroup`

**Output**:
- S3: `{destinationBucket}/{destinationKeyPrefix}/{YYYY}/{MM}/{DD}/{filename}.csv` or `.parquet`
- MySQL: `delivery_type_queue` (status: ready; used by S3 delivery scheduler)

---

### advito-delivery

**Type**: ECS Fargate Task (ARM64, scheduled)
**Trigger**: EventBridge cron (hourly) — schedule expression configured at deploy time
**Compute**: 1024 MB RAM, 1024 CPU units (1 vCPU)

**What it does**: A scheduled custom delivery job for the Advito customer. Each hour it checks whether the current time matches the customer's configured delivery hour and frequency (cron expression stored in `customer.frequency`, evaluated in the customer's timezone). When it matches, it loads packaged vacation pricing data from the delivery archive via `PackagedLoader`, applies a `CheapestEarliestPerSourceFilter` to keep only the cheapest and earliest available option per source, writes the result as an uncompressed CSV (stripping the first `id` field), archives the file to S3, and publishes a `PEDeliveryMessage` to SQS for downstream delivery.

**Input**:
- S3: `{archive.bucket}/{Advito}/{deliveryName}/{date}/` (packaged vacation CSV files)
- MySQL: `pe_customer`, `pe_customer_delivery_config`, `pe_customer_packaging` (customer/delivery configs)
- Config: `advito-delivery.properties` → `archive.bucket`

**Output**:
- S3: `{archive.bucket}/Advito/all/{YYYY}/{MM}/{DD}/{HH}/{filename}.csv` (filtered delivery file)
- SQS: `PEDeliveryMessage` for downstream delivery

---

### sk-report

**Type**: ECS Fargate Task (ARM64, scheduled)
**Trigger**: EventBridge cron (hourly) — schedule expression configured at deploy time
**Compute**: 1024 MB RAM, 1024 CPU units (1 vCPU)

**What it does**: Generates a daily formatted pricing report for the SK (Scandinavian Airlines) customer. Each hour it checks whether the current Stockholm time is 7am or 8am (Europe/Stockholm) or 12:00 UTC, and if so, runs `PESKReportDaily` to read SK packaged data from the S3 packager archive, format it into `SkDailyReportFormat`, write it to a local file, and deliver it via SFTP.

**Input**:
- S3: packager archive bucket (SK vacation package data in `SkFormat`)
- Config: sk-report properties → S3 report bucket, SFTP credentials (via Secrets Manager)

**Output**:
- SFTP: daily SK formatted report file
- S3: `{reportBucket}/` (report archive)

---

## S3 Buckets Defined in this Repo

_(Defined in `source/deploy/yaml/priceeye-customer-buckets.yaml`)_

| Bucket Name | Purpose | EventBridge Enabled |
|-------------|---------|---------------------|
| `s3-atp-3victors-{env}-use1-pe-vacation-room-type-vectors` | Stores S3 Vector bucket for vacation room name embeddings used by vacation-metadata | Yes |
| `s3-atp-3victors-{env}-use1-pe-curate-hotels` | Drop zone for operator hotel curation CSVs; events trigger curate-hotels Lambda | Yes |

**S3 Buckets referenced but defined elsewhere:**

| Bucket | Used by |
|--------|---------|
| `s3-atp-3victors{env}-use1-priceeye-alaska` | alaska-delivery (trigger source) |
| `price-eye-customer-delivery` | vacation-metadata, vacation-hotel-enrich, vacation-local-fees, vacation-output-rewrite (packager archive read) |
| `pe-delivery-archive` / `price-eye-delivery-archive` | vacation-output-rewrite, vacation-compare (delivery write) |
| `3v-upload-bucket` | vacation-metadata (metadata export) |

---

## ECS Clusters

| Environment | Cluster Name |
|-------------|-------------|
| 3vdev | `ecs-3vdev-use1-price-eye-customer` |
| 3vgold | `ecs-3vgold-use1-price-eye-customer` |
| 3vprod | `ecs-3vprod-use1-price-eye-customer` |

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| ECS Fargate Tasks (Step Function pipeline) | 5 (`vacation-metadata`, `vacation-hotel-enrich`, `vacation-local-fees`, `vacation-output-rewrite`, `vacation-compare`) |
| ECS Fargate Tasks (scheduled standalone) | 2 (`advito-delivery`, `sk-report`) |
| Lambda Functions | 3 (`alaska-delivery`, `curate-hotels`, `tui-delivery`) |
| Step Functions | 1 (`vacation-post-processing-step-function`) |
| EventBridge Rules (S3 event-driven) | 2 (`alaska-delivery`, `curate-hotels`) |
| EventBridge Rules (cron-scheduled) | 3 (`vacation-post-processing-step-function`, `advito-delivery`, `sk-report`) |
| S3 Buckets (defined here) | 2 (`pe-vacation-room-type-vectors`, `pe-curate-hotels`) |
| CloudWatch Alarms | 1 per Lambda (timeout alarm → `HighPriorityAlarm` SNS) |
| External Dependencies | MySQL (vacation metadata DB), Redis (vacation-compare state), Google Places API, Gemini AI |
