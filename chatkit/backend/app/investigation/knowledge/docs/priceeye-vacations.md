# priceeye-vacations

> Event-driven pipeline that submits vacation-package (air+hotel, air+car, air+hotel+car) pricing requests to external providers (Hitech, QL2, OJT), ingests and parses their responses, and routes normalised itinerary data into the PriceEye global downstream pipeline.

> **Production branch**: `master` (this document reflects the `develop` branch, which is the current HEAD at time of writing — 2026-02-28)

---

## Architecture Overview

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                     HITECH PROVIDER FLOW                        │
  │                                                                 │
  │  S3: pe-batch-providers/                                        │
  │    Hitech/requests/*.json   (written by upstream scheduler)     │
  │         │                                                       │
  │         ▼  EventBridge S3-ObjectCreated                         │
  │  [provider-hitech-request]  Lambda / 624 MB / 120 s            │
  │    reads  JSON request bundle from s3-pe-batch-providers        │
  │    writes CSV job files ──► S3: pe-hitech-transfer-bucket/      │
  │               Hitech partner (external) places results:         │
  │    ┌─────────────────────────────────────┐                      │
  │    │  *-output_*.csv  (success results)  │                      │
  │    │  *-outputError_*.csv (error marker) │                      │
  │    └──────────────────┬──────────────────┘                      │
  │                       │  EventBridge S3-ObjectCreated           │
  │          ┌────────────┴──────────────┐                          │
  │          ▼                           ▼                          │
  │ [provider-hitech-response-handler]  [provider-hitech-error-     │
  │   Lambda / 1024 MB / 900 s           response-checker]          │
  │   reads CSV, parses itineraries      Lambda / 2048 MB / 270 s   │
  │   writes raw archive to:            publishes S3 key to:        │
  │     pe-hitech-provider-archive/      SQS: PEProviderHitechError │
  │   sends to PEGlobalFilter.fifo                FileResponse.fifo │
  │                                              │                  │
  │                                              ▼                  │
  │                               [provider-hitech-error-file-      │
  │                                response-handler]                │
  │                                Lambda / 512 MB / 60 s           │
  │                                SQS-triggered (batch=10)         │
  │                                processes error files, sends     │
  │                                audit records                    │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                      QL2 VACATION FLOW                          │
  │                                                                 │
  │  SQS: PEQL2Vacation.fifo   (upstream request queue)            │
  │         │                                                       │
  │         ▼  SQS EventSourceMapping                               │
  │  [provider-ql2-vacation-request-generator]                      │
  │    ECS Fargate / 8192 MB / arm64                                │
  │    reads request bundles from PEQL2Vacation.fifo                │
  │    POSTs search jobs to QL2 API (HTTP)                          │
  │    QL2 partner places results:                                  │
  │    ┌──────────────────────────────────────┐                     │
  │    │  *_out.csv     (success results)     │                     │
  │    │  *_error.csv   (error results)       │                     │
  │    │  *_rerun.csv   (re-run marker)       │                     │
  │    └──────────────────┬───────────────────┘                     │
  │           S3: pe-ql2-vacation-transfer-bucket/                  │
  │                       │  EventBridge S3-ObjectCreated           │
  │          ┌────────────┴──────────────┐                          │
  │          ▼ (_out, _rerun)            ▼ (_error)                 │
  │ [provider-ql2-vacation-response-handler]  [provider-ql2-        │
  │   Lambda / 1024 MB / 900 s              vacation-error-         │
  │   parses success CSV, builds            response-checker]       │
  │   PEVacationItinerary objects           Lambda / 2048 MB / 270s │
  │   sends to PEGlobalFilter.fifo          publishes to SQS:       │
  │   archives raw to:                        PEProviderQL2Vacation  │
  │     pe-ql2-vacation-provider-output/        ErrorFileResponse.fifo
  │                                              │                  │
  │                                              ▼                  │
  │                               [provider-ql2-vacation-error-     │
  │                                file-response-handler]           │
  │                                Lambda / 512 MB / 60 s           │
  │                                SQS-triggered (batch=10)         │
  │                                processes error files, sends     │
  │                                retry + audit records            │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                      OJT (Open Jaw Tech) FLOW                   │
  │                                                                 │
  │  SQS: PEProviderOJTRequest.fifo   (upstream request queue)     │
  │         │                                                       │
  │         ▼  SQS EventSourceMapping (max concurrency 32)          │
  │  [provider-ojt-request]  Lambda / 624 MB / 60 s                │
  │    reads request bundles from PEProviderOJTRequest.fifo         │
  │    fetches OAuth2 token from OJT auth endpoint                  │
  │    POSTs JSON search request to OJT PricingInsight API          │
  │    enqueues HTTP work on SQS: PEProviderOJTHttpService.fifo     │
  │                       │                                         │
  │                       ▼  SQS EventSourceMapping (max concurr 32)│
  │  [provider-ojt-response]  Lambda / 624 MB / 120 s              │
  │    reads HTTP responses from PEProviderOJTHttpService.fifo      │
  │    parses OJT JSON response, maps rental-car codes              │
  │    sends parsed itineraries to PEGlobalFilter.fifo              │
  └─────────────────────────────────────────────────────────────────┘

  Common downstream:
  PEGlobalFilter.fifo ──► (packager / global-filter pipeline, outside this repo)
  PERetry.fifo        ──► (retry pipeline, outside this repo)
```

---

## Orchestration

There are **no AWS Step Functions** defined in this repository. Each provider sub-pipeline is independently triggered by EventBridge S3 object-creation events or SQS queue messages. Orchestration of *when* request files are placed into S3 (for Hitech) or *when* messages are written to `PEQL2Vacation.fifo` / `PEProviderOJTRequest.fifo` is owned by the upstream PriceEye request-scheduling system (a separate repository).

### EventBridge Rules

| Rule (logical name) | Trigger | Target Lambda |
|---|---|---|
| `provider-hitech-request` forwarding | S3 object created in `pe-batch-providers`, key `Hitech/requests/*.json` | `provider-hitech-request` |
| `provider-hitech-response-handler` forwarding | S3 object created in `pe-hitech-transfer-bucket`, key `files_for_3Victors/*-output_*.csv` | `provider-hitech-response-handler` |
| `provider-hitech-error-response-checker` forwarding | S3 object created in `pe-hitech-transfer-bucket`, key `files_for_3Victors/*-outputError_*.csv` | `provider-hitech-error-response-checker` |
| `provider-ql2-vacation-response-handler` forwarding (success) | S3 object created in `pe-ql2-vacation-transfer-bucket`, key suffix `_out.csv` | `provider-ql2-vacation-response-handler` |
| `provider-ql2-vacation-response-handler` forwarding (rerun) | S3 object created in `pe-ql2-vacation-transfer-bucket`, key suffix `_rerun.csv` | `provider-ql2-vacation-response-handler` |
| `provider-ql2-vacation-error-response-checker` forwarding | S3 object created in `pe-ql2-vacation-transfer-bucket`, key suffix `_error.csv` | `provider-ql2-vacation-error-response-checker` |

---

## Components

_(ordered by pipeline sequence within each provider sub-flow)_

---

### provider-hitech-request

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: EventBridge S3 `Object Created` — bucket `s3-atp-3victors{env}-use1-pe-batch-providers`, key pattern `Hitech/requests/*.json`
**Compute**: 624 MB memory, 120 s timeout, VPC-attached (FMSSecuritygroupApp)
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.hitech.request.PEProviderLambdaHitechRequestGenerator::handleRequest`
**Source**: `source/provider-lambdas/provider-hitech/request/`

**What it does**: Reads a JSON array of `PEVacationExpandedInputRequest` objects from the batch-providers S3 bucket. It groups requests by vacation type (AirHotel, AirCar, AirHotelCar), converts each group into a CSV row formatted for the Hitech (3Victors) search engine, and uploads the resulting CSV batch files to the Hitech transfer bucket under the configured `inputFolderName`. After a successful file write, the original JSON file is archived to an `archive/` prefix in the same bucket. A 5-minute distributed job lock prevents duplicate event processing.

**Input**:
- S3: `s3-atp-3victors{env}-use1-pe-batch-providers/Hitech/requests/*.json` — JSON array of vacation expanded input requests
- DB (Aurora): `PECabinUtil` reads cabin mappings; `JobLockDAO` reads/writes lock state

**Output**:
- S3: `s3-atp-3victors{env}-use1-pe-hitech-transfer-bucket/{inputFolderName}/{jobName}.csv` — CSV batch files for the Hitech provider
- S3 (archive): `s3-atp-3victors{env}-use1-pe-batch-providers/archive/Hitech/requests/{filename}.json`

**CSV output columns**: `id, crawl_date_utc, shop_type, site, marketed_airline, number_of_stops, origin_airport_code, destination_airport_code, depart_day, return_day, outbound_depart_window, inbound_depart_window, product_type, flight_selection_order, rooms, occupancy, hotels_per_search, hotel_search_order, car_brand, car_type, car_model, job_name, crawl_run_hour_UTC, priority, dropdead_datetime_utc`

**CloudWatch Alarm**: `alarm-timeout` fires to `HighPriorityAlarm` SNS topic if Lambda duration >= 120,000 ms.

---

### provider-hitech-response-handler

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: EventBridge S3 `Object Created` — bucket `s3-atp-3victors{env}-use1-pe-hitech-transfer-bucket`, key pattern `files_for_3Victors/*-output_*.csv`
**Compute**: 1024 MB memory, 900 s timeout, VPC-attached
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.hitech.response.handler.PEProviderHitechResponseHandler::handleRequest`
**Source**: `source/provider-lambdas/provider-hitech/response-handler/`, `source/common-hitech-response-handler/`

**What it does**: Receives an EventBridge notification that Hitech has placed a success CSV file on the transfer bucket. It delegates to `HitechResponseHandler`, which reads the CSV row-by-row via `ProcessHitechOutputFile`, converts each row into a `PEVacationItinerary`, and publishes parsed itinerary bundles to the `PEGlobalFilter.fifo` SQS queue for downstream packaging. On completion the raw CSV is moved (archived) to the `pe-hitech-provider-archive` bucket under a date-partitioned `v1/YYYY/MM/DD/Success/` path; error files land under `.../Error/`. Application statistics are persisted to the PriceEye report database.

**Input**:
- S3: `s3-atp-3victors{env}-use1-pe-hitech-transfer-bucket/files_for_3Victors/*-output_*.csv`
- DB (Aurora): `PriceEyeVacationsAuditReader` for request lookup, `PriceEyeReportReader`

**Output**:
- SQS: `PEGlobalFilter.fifo` — `PERequestResponse` objects (itinerary + request bundle)
- SQS: `PERetry.fifo` — retry messages for failed/unparseable responses
- S3 (archive): `s3-atp-3victors{env}-use1-pe-hitech-provider-archive/v1/YYYY/MM/DD/Success/`
- Kinesis/Firehose: provider response audit records (via `CommonAuditDataStreamPublisher`)

**CloudWatch Alarm**: `alarm-timeout` at 900,000 ms threshold.

---

### provider-hitech-error-response-checker

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: EventBridge S3 `Object Created` — bucket `s3-atp-3victors{env}-use1-pe-hitech-transfer-bucket`, key pattern `files_for_3Victors/*-outputError_*.csv`
**Compute**: 2048 MB memory, 270 s timeout, VPC-attached
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.hitech.error.response.checker.PEProviderHitechErrorResponseChecker::handleRequest`
**Source**: `source/provider-lambdas/provider-hitech/error-response-checker/`

**What it does**: Acts as a thin fan-out stage that receives the S3 notification when Hitech delivers an error response CSV. It wraps the bucket name and S3 key into an `ErrorFileResponseHandlerMessage` and publishes it to the FIFO SQS queue `PEProviderHitechErrorFileResponse.fifo` (15-minute delay). This decouples the EventBridge invocation from the heavier error-file parsing performed by `provider-hitech-error-file-response-handler`.

**Input**:
- S3 key (from EventBridge event): `files_for_3Victors/*-outputError_*.csv`

**Output**:
- SQS: `PEProviderHitechErrorFileResponse.fifo` — `ErrorFileResponseHandlerMessage` (bucket + key)

---

### provider-hitech-error-file-response-handler

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: SQS `PEProviderHitechErrorFileResponse.fifo` (batch size 10, `ReportBatchItemFailures` enabled, 15-minute delay, DLQ: `FAILED-PEProviderHitechErrorFileResponse.fifo`)
**Compute**: 512 MB memory, 60 s timeout, VPC-attached
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.hitech.errorfile.response.handler.PEProviderHitechErrorFileResponseHandler::handleRequest`
**Source**: `source/provider-lambdas/provider-hitech/error-file-response-handler/`, `source/common-hitech-response-handler/`

**What it does**: Consumes batches of `ErrorFileResponseHandlerMessage` from the FIFO queue. For each message it reads the corresponding Hitech error CSV from S3 and calls `HitechErrorResponseHandler.processResponse()`, which parses error records, categorises them (timeout vs. failed), sends audit records via `CommonAuditDataStreamPublisher`, and flushes any pending audit arrays. Failed SQS items are reported individually so only successfully processed messages are deleted.

**Input**:
- SQS: `PEProviderHitechErrorFileResponse.fifo`
- S3: error CSV files from `s3-atp-3victors{env}-use1-pe-hitech-transfer-bucket/`

**Output**:
- Kinesis/Firehose: provider response audit records
- SQS: `PERetry.fifo` (retry messages for timed-out requests, via `HitechErrorResponseHandler`)

---

### provider-ql2-vacation-request-generator

**Type**: ECS Fargate Task (arm64, Linux)
**Trigger**: SQS `PEQL2Vacation.fifo` — queue-driven; task is launched when messages are present (managed by PEQueueApplicationRunner framework)
**Compute**: 8192 MB memory, 2048 CPU units (2 vCPU), configurable thread pool
**Source**: `source/provider-ql2-vacation-request-generator/`

**What it does**: Consumes `PERequestBundle` messages from the upstream `PEQL2Vacation.fifo` SQS queue. It queues all incoming requests in memory and, on shutdown, groups them into QL2 search "jobs" (using `PEQl2VacationJobBuilder`) keyed by site code and vacation type. For each job it builds a QL2 POST request (CSV body) via `PEQl2VacationRequestBuilder`, authenticates using credentials from Secrets Manager (`provider/QL2`), and submits the job to the QL2 Vacation API via `PEQl2VacationCommunicator`. Successful job submissions optionally archive the raw request body to a configured S3 bucket. Failed submissions emit `PEProviderResponseAudit` records.

**Input**:
- SQS: `PEQL2Vacation.fifo` — `PERequestBundle` (vacation expanded input requests)
- DB (Aurora): site map via `PriceEyeReader`
- Secrets Manager: `provider/QL2` — API credentials

**Output**:
- HTTP POST: QL2 Vacation API (external) — search job submissions
- S3 (optional archive): configured `archive.bucket` — raw request CSVs
- Kinesis/Firehose: `CommonAuditDataStreamPublisher` — provider response audits on error

**CSV request header sent to QL2**: `Site_ID, Origin, Destination, Departure, Return, Adults, Children, Rooms, Hotel_Address, Reference, Search_Type, Carrier, Flight_Number, Rental_Car_Agency, Car_Type, Depart_Time, Return_Time, Property_Name, Length_of_Stay, Stars, Max_Properties, Max_Properties_applies_to_each_Star_Rating, Board_Basis, Sort_by_Price, Room_Type, Property_Id, POS, Rates_Per_Hotel, DOW_filter_depart, DOW_filter_return, Custom, Geo, Sort`

---

### provider-ql2-vacation-response-handler

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: EventBridge S3 `Object Created` — bucket `s3-atp-3victors{env}-use1-pe-ql2-vacation-transfer-bucket`, key suffix `_out.csv` or `_rerun.csv`
**Compute**: 1024 MB memory, 900 s timeout, VPC-attached
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.ql2.vacation.response.handler.PEProviderQL2VacationResponseHandler::handleRequest`
**Source**: `source/provider-lambdas/provider-ql2/vacation-response-handler/`, `source/common-ql2-vacation-response-handler/`

**What it does**: Triggered when QL2 delivers a response file to the vacation transfer bucket. The `QL2VacationResponseHandler` reads success (`_out.csv`) files line-by-line (expecting 62+ CSV fields per line), constructs `PEVacationItinerary` objects via `PEVacationItineraryBuilder` (using airport-to-timezone metadata), sorts itineraries by total price then duration, and publishes bundles to `PEGlobalFilter.fifo`. For `_rerun.csv` files, it simply deletes the file. Successfully parsed raw files are archived to `pe-ql2-vacation-provider-output` under `v1/YYYY/MM/DD/success/`. Response statistics are recorded via `PEResponseStatsGenerator`.

**Input**:
- S3: `s3-atp-3victors{env}-use1-pe-ql2-vacation-transfer-bucket/*_out.csv` or `*_rerun.csv`
- DB (Aurora): `PriceEyeReader` (site map), `PriceEyeReportReader` (expanded input request lookup, audit details), `MetadataReader` (airport-timezone map)

**Output**:
- SQS: `PEGlobalFilter.fifo` — `PERequestResponse` (itineraries + request bundle)
- SQS: `PERetry.fifo` — retry messages for failed/unparseable responses
- S3 (archive): `s3-atp-3victors{env}-use1-pe-ql2-vacation-provider-output/v1/YYYY/MM/DD/success/`
- Kinesis/Firehose: provider response audits

---

### provider-ql2-vacation-error-response-checker

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: EventBridge S3 `Object Created` — bucket `s3-atp-3victors{env}-use1-pe-ql2-vacation-transfer-bucket`, key suffix `_error.csv`
**Compute**: 2048 MB memory, 270 s timeout, VPC-attached
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.ql2.vacation.error.response.checker.PEProviderQL2VacationErrorResponseChecker::handleRequest`
**Source**: `source/provider-lambdas/provider-ql2/vacation-error-response-checker/`

**What it does**: Mirror of the Hitech error-response-checker for the QL2 Vacation flow. Wraps the S3 bucket and key of an error CSV into an `ErrorFileResponseHandlerMessage` and publishes to `PEProviderQL2VacationErrorFileResponse.fifo` (15-minute delay). This prevents the EventBridge Lambda from timing out on large error files and allows retry semantics via SQS.

**Input**:
- S3 key (from EventBridge): `*_error.csv` in `pe-ql2-vacation-transfer-bucket`

**Output**:
- SQS: `PEProviderQL2VacationErrorFileResponse.fifo` — `ErrorFileResponseHandlerMessage`

---

### provider-ql2-vacation-error-file-response-handler

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: SQS `PEProviderQL2VacationErrorFileResponse.fifo` (batch size 10, 15-minute delay, DLQ: `FAILED-PEProviderQL2VacationErrorFileResponse.fifo`)
**Compute**: 512 MB memory, 60 s timeout, VPC-attached
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.ql2.vacation.errorfile.response.handler.PEProviderQL2VacationErrorFileResponseHandler::handleRequest`
**Source**: `source/provider-lambdas/provider-ql2/vacation-error-file-response-handler/`, `source/common-ql2-vacation-response-handler/`

**What it does**: Processes batches of `ErrorFileResponseHandlerMessage` from the FIFO queue. For each message it reads the QL2 error CSV from S3 using `QL2VacationErrorResponseHandler`, parses each error record, categorises responses as `timeout` (if "Search Aborted") or `failed`, sends retry messages to `PERetry.fifo`, and emits audit records. The handler uses `ReportBatchItemFailures` semantics to individually fail only unprocessable SQS records.

**Input**:
- SQS: `PEProviderQL2VacationErrorFileResponse.fifo`
- S3: error CSV files from `pe-ql2-vacation-transfer-bucket`

**Output**:
- SQS: `PERetry.fifo` — retry messages
- Kinesis/Firehose: provider response audit records

---

### provider-ojt-request

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: SQS `PEProviderOJTRequest.fifo` (batch size 10, max concurrency 32, DLQ: `FAILED-PEProviderOJTRequest.fifo`)
**Compute**: 624 MB memory, 60 s timeout, VPC-attached
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.ojt.request.PEProviderLambdaOJTRequestGenerator::handleRequest`
**Source**: `source/provider-lambdas/provider-ojt/request/`

**What it does**: Consumes `PERequestBundle` messages from the OJT request queue. For each bundle it resolves airport codes to OJT `location_code` values (read from Aurora table `priceeye.ojt_hotel_locations`), determines the vacation type connection string (`HF`, `VF`, or `HFV`), and renders a Velocity template (`open-jaw-technologies.json.vm`) into a JSON POST body for the OJT PricingInsight API. OAuth2 bearer tokens are obtained from the OJT auth endpoint (credentials from Secrets Manager `provider/OJT`) with a 1-hour cache. The HTTP request (with auth) is enqueued onto `PEProviderOJTHttpService.fifo` for asynchronous execution.

**Input**:
- SQS: `PEProviderOJTRequest.fifo` — `PERequestBundle`
- DB (Aurora): `priceeye.ojt_hotel_locations` (airport → location code mapping)
- Secrets Manager: `provider/OJT` — `client-id`, `client-secret`, `auth-url`, `request-url`

**Output**:
- SQS: `PEProviderOJTHttpService.fifo` — `PEHttpServiceRequest` (serialised HTTP request + auth config)

---

### provider-ojt-response

**Type**: AWS Lambda Function (container image, arm64)
**Trigger**: SQS `PEProviderOJTResponse.fifo` (batch size 10, max concurrency 32, DLQ: `FAILED-PEProviderOJTResponse.fifo`)
**Compute**: 624 MB memory, 120 s timeout, VPC-attached
**Handler**: `com.threevictors.aws.priceeye.provider.lambdas.provider.ojt.response.PEProviderLambdaOJTResponseParser::handleRequest`
**Source**: `source/provider-lambdas/provider-ojt/response/`

**What it does**: Consumes HTTP responses from `PEProviderOJTResponse.fifo` (the HTTP service queue result). The `OJTResponseJsonConverter` parses the OJT JSON response, resolves rental-car agency shorthand codes to full names (from Aurora `priceeye.rental_agencies`), and builds `PEOJTItinerary` objects. Error responses are classified as `noresult` (no flight supplier results), `timeout`, or `failed`. Successful parse results are published upstream as `PEProviderParserResponse` objects routed to `PEGlobalFilter.fifo`.

**Input**:
- SQS: `PEProviderOJTResponse.fifo` — HTTP response payloads
- DB (Aurora): `priceeye.rental_agencies` (shorthand → agency name)

**Output**:
- SQS: `PEGlobalFilter.fifo` — parsed itinerary responses
- Kinesis/Firehose: provider response audits (noresult, timeout, failed classifications)

---

## Shared Library Modules

These modules are not deployed independently but are packaged into the Lambda/ECS images that use them.

### common-hitech-response-handler (`source/common-hitech-response-handler/`)

Shared library used by `provider-hitech-response-handler` and `provider-hitech-error-file-response-handler`. Contains `HitechResponseHandler`, `HitechErrorResponseHandler`, `HitechToItineraryConverter`, `ProcessHitechOutputFile`, and `LegParser`. Responsible for the core business logic of reading Hitech CSV files, converting rows into `PEVacationItinerary` flight legs, and publishing to downstream queues.

### common-ql2-vacation-response-handler (`source/common-ql2-vacation-response-handler/`)

Shared library used by `provider-ql2-vacation-response-handler` and `provider-ql2-vacation-error-file-response-handler`. Contains `QL2VacationResponseHandler`, `QL2VacationErrorResponseHandler`, `PEVacationItineraryBuilder`, `PESuccessFileReader`, `PEErrorFileReader`, `PEResponseStatsGenerator`, `PEAirportTimezoneCache`, and related data classes. Core logic for parsing QL2 CSV files (62-field success format) into normalised vacation itineraries.

### packager-vacation-output (`source/packager-vacation-output/`)

`VacationOutputFormatter` — an `OutputFormatter` plugin loaded dynamically by the PriceEye packager (separate repository). Extends `ItineraryOutputFormatter` with vacation-specific output fields: hotel details (id, name, check-in/check-out dates, room type, refundable status, inventory category), car rental details (company, type, model), passenger count, destination fees, region/area lookup (from `swav.region_area` table), overnight flight indicator, carrier type classification (Legacy/LCC/ULCC/MULTI/UNKNOWN), and `total_without_local_fees`. Also handles room price increment expansion — each room type in `RoomPriceIncrements` generates a separate output row.

### packager-ojt-output (`source/packager-ojt-output/`)

`OJTOutputFormatter` — extends `VacationOutputFormatter` with OJT-specific output fields covering detailed flight pricing breakdowns (per-component: package, rooms, car, flight), nightly room rates (up to 14 nights), car pickup/dropoff details (date, time, depot), SIPP codes, hotel chain/location codes, and promotional discount fields. Emits one output row per room type per itinerary.

### packager-vacation-hooks (`source/packager-vacation-hooks/`)

Three `PEHook` plugin implementations:
- `VacationLocalFeesHook` — enriches `PEVacationItinerary.total_wolocaltfc` (total without local taxes and fees) using a fee ratio table loaded from Aurora (`PriceEyeVacationMetadataReader.getVacationLocalFees()`). Uses hotel name normalisation and airport-to-city mapping to match fee records.
- `VacationHotelRefundableCategoryHook` — populates hotel refundable category.
- `VacationHotelInventoryCategoryHook` — populates hotel inventory category.

### dao (`source/dao/`)

`PriceEyeVacationsReader` — Aurora database reader providing:
- `getRegionAreas()` — reads `swav.region_area` (airportCode → region/area)
- `loadOJTLocationsMap()` — reads `priceeye.ojt_hotel_locations` (airport_code → location_code)
- `getRentalAgencyCodeToNameMap()` — reads `priceeye.rental_agencies` (shorthand → agency)
- `getQL2VacationSiteCodes()` — reads `priceeye.ql2_vacation_site` (site_code → ql2_site_code)

### data (`source/data/`)

Domain data classes: `PEOJTItinerary` (extends `PEVacationItinerary` with OJT-specific car/flight pricing structures), `RoomPricingNightlyRates`, `RoomPricingPromotions`, `CarPricingPromotions`, `PERegionArea`.

### data-serde-vacations (`source/data-serde-vacations/`)

Kryo serialisation registration for vacation data classes, loaded dynamically as a serde JAR from `s3-atp-3victors{env}-use1-pe-injection-jars/serde/` at Lambda startup.

---

## Glue Databases

No `AWS::Glue::Table` resources are defined in this repository. The Glue catalog tables for vacation data are defined in a separate repository (likely `priceeye-analytics` or `spark-v3`).

---

## S3 Buckets

| Bucket (pattern) | Purpose | EventBridge enabled |
|---|---|---|
| `s3-atp-3victors{env}-use1-pe-batch-providers` | Incoming Hitech JSON request files from scheduler | No (bucket policy grants external partner write) |
| `s3-atp-3victors{env}-use1-pe-hitech-transfer-bucket` | Hitech bidirectional transfer: CSV requests written by `provider-hitech-request`; success/error CSVs placed by Hitech (external, account 694352152025) | Yes |
| `s3-atp-3victors{env}-use1-pe-hitech-provider-archive` | Archive of processed Hitech response files | Yes |
| `s3-atp-3victors{env}-use1-pe-ql2-vacation-transfer-bucket` | QL2 bidirectional transfer: jobs submitted by request-generator; `_out.csv`, `_error.csv`, `_rerun.csv` placed by QL2 (external, account 288192894589) | Yes |
| `s3-atp-3victors{env}-use1-pe-ql2-vacation-provider-output` | Archive of processed QL2 response files | Yes |

---

## SQS Queues

| Queue | Type | Delay | DLQ | Consumer |
|---|---|---|---|---|
| `PEProviderHitechErrorFileResponse.fifo` | FIFO | 15 min | `FAILED-PEProviderHitechErrorFileResponse.fifo` | `provider-hitech-error-file-response-handler` |
| `FAILED-PEProviderHitechErrorFileResponse.fifo` | FIFO | 0 s | — | Manual |
| `PEProviderOJTRequest.fifo` | FIFO | 0 s | `FAILED-PEProviderOJTRequest.fifo` | `provider-ojt-request` |
| `FAILED-PEProviderOJTRequest.fifo` | FIFO | 0 s | — | Manual |
| `PEProviderOJTHttpService.fifo` | FIFO | 0 s | `FAILED-PEProviderOJTHttpService.fifo` | OJT HTTP service (part of framework) |
| `FAILED-PEProviderOJTHttpService.fifo` | FIFO | 0 s | — | Manual |
| `PEProviderOJTResponse.fifo` | FIFO | 0 s | `FAILED-PEProviderOJTResponse.fifo` | `provider-ojt-response` |
| `FAILED-PEProviderOJTResponse.fifo` | FIFO | 0 s | — | Manual |
| `PEProviderQL2VacationErrorFileResponse.fifo` | FIFO | 15 min | `FAILED-PEProviderQL2VacationErrorFileResponse.fifo` | `provider-ql2-vacation-error-file-response-handler` |
| `FAILED-PEProviderQL2VacationErrorFileResponse.fifo` | FIFO | 0 s | — | Manual |
| `PEQL2Vacation.fifo` | FIFO | 0 s | — | `provider-ql2-vacation-request-generator` (ECS) |
| `PEGlobalFilter.fifo` | FIFO | 0 s | — | Global filter pipeline (external repo) |
| `PERetry.fifo` | FIFO | 0 s | — | Retry pipeline (external repo) |

---

## Infrastructure Summary

| Resource | Count |
|---|---|
| AWS Lambda Functions | 8 (provider-hitech-request, provider-hitech-response-handler, provider-hitech-error-response-checker, provider-hitech-error-file-response-handler, provider-ql2-vacation-response-handler, provider-ql2-vacation-error-response-checker, provider-ql2-vacation-error-file-response-handler, provider-ojt-request, provider-ojt-response) |
| ECS Fargate Tasks | 1 (provider-ql2-vacation-request-generator) |
| S3 Buckets | 5 |
| EventBridge Rules | 6 |
| SQS FIFO Queues | 13 (including DLQs) |
| CloudWatch Log Groups | 9 (one per Lambda/ECS stack, 7-day retention) |
| CloudWatch Alarms | 8 (alarm-timeout per Lambda) |
| IAM Roles | 9 (one per component) |
| Secrets Manager Secrets | 2 (`provider/OJT`, `provider/QL2`) |
| External Provider Accounts | 2 (Hitech: 694352152025, QL2: 288192894589) |

---

## Key Design Patterns

1. **Error-checker / error-file-handler split**: For both Hitech and QL2, an EventBridge-triggered Lambda (checker) immediately acknowledges the S3 event and publishes the S3 key to a 15-minute-delayed FIFO SQS queue. A separate SQS-triggered Lambda (handler) then does the heavy lifting. This prevents EventBridge Lambda timeouts on large error files and allows SQS retry semantics.

2. **Job-lock deduplication**: All EventBridge-triggered Lambdas use `JobLockDAO.obtain5MinTimedLock()` to prevent duplicate processing when EventBridge delivers the same S3 notification more than once.

3. **Dynamic serde loading**: Lambda containers load Kryo serialisation JARs at startup from `s3-atp-3victors{env}-use1-pe-injection-jars/serde/` via the `run.sh` entrypoint, allowing serde evolution without rebuilding images.

4. **Output plugin architecture**: `VacationOutputFormatter` and `OJTOutputFormatter` are loaded by the packager via Java `ServiceLoader` (META-INF/services), making vacation formatting concerns fully isolated from the core packager engine.

5. **Room price increment expansion**: A single QL2/OJT itinerary with multiple `roomPriceIncrements` entries is exploded at packaging time into one output row per room type, each with adjusted total prices.
