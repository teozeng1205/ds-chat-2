# priceeye-providers

> The provider integration layer for PriceEye: queries airline and travel-agency APIs (or internal caches) for flight pricing data, parses the responses, and forwards standardized itinerary records to the downstream Global Filter pipeline.

> **Branch note**: Documented from the `develop` branch. The `master`/`main` branch represents what is currently running in production; this document may reflect in-progress changes.

---

## Architecture Overview

Each provider follows one of three patterns:

### Pattern A — Real-time HTTP providers (majority of providers)

```
[Scheduler / upstream system]
        │
        ▼ PEProvider{CODE}Request.fifo
[Request Generator Lambda]   ← builds HTTP request headers/auth/body
        │
        ▼ PEProvider{CODE}HttpService.fifo
[HTTP Service (ECS Fargate)]  ← long-running, makes actual HTTP call to provider API
        │
        ▼ PEProvider{CODE}Response.fifo
[Response Handler Lambda]    ← parses provider-specific XML/JSON, extracts itineraries
        │
        ├──► PEGlobalFilter.fifo   (success — downstream pricing pipeline)
        └──► PERetry.fifo          (failures — retry queue)

     DLQs: FAILED-PEProvider{CODE}HttpService.fifo
           ↓
     [dlq-processor Lambda]       ← reprocesses dead-lettered HTTP requests
```

### Pattern B — Batch file providers (AI, QL2, PIT)

```
[External batch system writes files to S3]
        │
        ▼ S3 object-created event (EventBridge)
[Request Lambda]  (AI/QL2/PIT)   ← registers the batch job
        │
        ▼  [External batch executes, writes output CSV/JSON to S3]
        │
        ▼ S3 object-created event (EventBridge, e.g. "_out.csv")
[Response Handler Lambda]         ← reads & parses CSV/JSON output files
        │
        ├──► PEGlobalFilter.fifo
        └──► PERetry.fifo

Error paths:
  S3: output/*/Error* or _error.csv
        ↓
  [Error Response Checker Lambda]
        ↓
  [Error File Response Handler Lambda]
```

### Pattern C — Internal ingest providers (DL, WN/Estream, Southwest)

```
[Scheduler] ──► PEProviderIngest{CODE}.fifo
        │
        ▼
[Ingest Lambda]   ← looks up search results from internal Redis cache
        │
        ├──► PEGlobalFilter.fifo   (found results)
        └──► PERetry.fifo          (not found / expired)
```

### Scheduled maintenance tasks

```
[EventBridge cron: Mondays 01:25 UTC]
        ▼
[provider-atlas-routes (ECS)]   ← refreshes Atlas carrier route data in MySQL

[EventBridge cron: every 15 min]
        ▼
[provider-ql2-response-replay (ECS)]  ← reprocesses failed QL2 S3 response files

[EventBridge cron: Mondays 09:00 UTC]
        ▼
[site-capabilities (ECS)]       ← probes provider sites, records detected capabilities in MySQL
```

---

## Provider Registry

The table below maps every provider code to its airline/system, HTTP-service variant, and SQS queue prefix.

| Code | Airline / System | HTTP Service Type | Queue Prefix |
|------|-----------------|-------------------|--------------|
| AA | American Airlines | `aa-http-service` (65 GB / 56 tasks) | `PEProviderAA` |
| AAPTS | American Airlines PTS | `aapts-http-service` | `PEProviderAAPts` |
| AI | Air India *(batch)* | EventBridge/S3 | — |
| AMP | Amadeus | `amp-http-service` | `PEProviderAMP` |
| AS | Alaska Airlines | `as-http-service` (65 GB / 56 tasks) | `PEProviderAS` |
| ATLAS | Atlas | `atlas-http-service` (65 GB / 56 tasks) | `PEProviderAtlas` |
| BA | British Airways | `ba-http-service` (65 GB / 56 tasks) | `PEProviderBA` |
| DL | Delta *(ingest)* | `PEProviderIngestDelta.fifo` | — |
| HA | Hawaiian Airlines | `ha-http-service` | `PEProviderHA` |
| LA | LATAM Airlines | `la-http-service` | `PEProviderLA` |
| LHG | Lufthansa Group | `lhg-http-service` | `PEProviderLHG` |
| MTC | MTC provider | `mtc-http-service` | `PEProviderMTC` |
| PIT | PITOBI *(batch)* | EventBridge/S3 | — |
| QL2 | Galileo/Amadeus QL2 *(batch)* | EventBridge/S3 | — |
| SP | SP provider | `sp-http-service` | `PEProviderSP` |
| SS | Skyscanner *(polling)* | `ss-http-polling-service` | `PEProviderSS` |
| TP | Travelport | `tp-http-service` | `PEProviderTP` |
| TS | TS provider | `ts-http-service` | `PEProviderTS` |
| UA | United Airlines | `ua-http-service` | `PEProviderUA` |
| VY | Vueling | `vy-http-service` | `PEProviderVY` |
| WL | WyndLabs | `wl-http-service` | `PEProviderWL` |
| WN | Southwest *(ingest/estream)* | `PEProviderIngestEstream` / `PEProviderIngestSouthwest` | — |

---

## Components

Components are organized by layer, from upstream trigger to downstream output.

---

### Request Generator Lambdas (one per real-time provider)

**Type**: AWS Lambda Function (arm64, image-based)
**Trigger**: SQS queue `PEProvider{CODE}Request.fifo` (batch size 10, max concurrency 32)
**Compute**: 624 MB, 60 s timeout
**Image pattern**: `3victors/priceeyev2/provider-lambdas-provider-{code}-request`

**What it does**: Each request generator lambda wakes when a `PEExpandedInputRequest` lands on its `PEProvider{CODE}Request.fifo` queue (placed there by the upstream scheduler). It builds a provider-specific `PEHttpServiceRequest` — including URL, headers, authentication tokens, and cookie setup — and publishes it to `PEProvider{CODE}HttpService.fifo` for the HTTP Service to execute. It also publishes an audit record to the Kinesis audit stream.

**Input**:
- SQS: `PEProvider{CODE}Request.fifo` — `PEExpandedInputRequest` objects from the scheduler
- S3 (optional): Large payloads via `PayloadS3Pointer` offloading

**Output**:
- SQS: `PEProvider{CODE}HttpService.fifo` — `PEHttpServiceRequest` objects
- Kinesis/Firehose: provider request audit records

**Deployed instances** (provider → image module):

| Lambda name | Queue trigger | Memory |
|-------------|--------------|--------|
| `provider-aa-request` | `PEProviderAARequest.fifo` | 624 MB |
| `provider-aapts-request` | `PEProviderAAPtsRequest.fifo` | 624 MB |
| `provider-amp-request` | `PEProviderAMPRequest.fifo` | 624 MB |
| `provider-as-request` | `PEProviderASRequest.fifo` | 624 MB |
| `provider-atlas-request` | `PEProviderAtlasRequest.fifo` | 624 MB |
| `provider-ba-request` | `PEProviderBARequest.fifo` | 624 MB |
| `provider-dl-request` | `PEProviderDLRequest.fifo` | 624 MB |
| `provider-ha-request` | `PEProviderHARequest.fifo` | 624 MB |
| `provider-la-request` | `PEProviderLARequest.fifo` | 624 MB |
| `provider-lhg-request` | `PEProviderLHGRequest.fifo` | 624 MB |
| `provider-mtc-request` | `PEProviderMTCRequest.fifo` | 624 MB |
| `provider-skyscanner-request` | `PEProviderSSRequest.fifo` | 624 MB |
| `provider-sp-request` | `PEProviderSPRequest.fifo` | 624 MB |
| `provider-tp-request` | `PEProviderTPRequest.fifo` | 624 MB |
| `provider-ts-request` | `PEProviderTSRequest.fifo` | 624 MB |
| `provider-ua-request` | `PEProviderUARequest.fifo` | 624 MB |
| `provider-vy-request` | `PEProviderVYRequest.fifo` | 624 MB |
| `provider-wn-request` | `PEProviderWNRequest.fifo` | 624 MB |
| `provider-wyndlabs-request` | `PEProviderWLRequest.fifo` | 624 MB |

**Source**: `source/provider-lambdas/provider-{code}/request/`

---

### http-service (ECS Fargate Service)

**Type**: AWS ECS Fargate long-running service
**Trigger**: Continuously polls `PEProvider{CODE}HttpService.fifo` queue(s); starts processing when messages are present, shuts down when the queue is empty
**Compute**: 32,768 MB RAM / 16,384 CPU units per task; 28 task instances per provider deployment
**Image**: `3victors/priceeyev2/http-service`

**What it does**: The core HTTP execution engine. Reads `PEHttpServiceRequest` messages from the provider's HttpService queue, performs optional token-based authentication (first HTTP request), then issues the main content request to the provider's API. Handles TPS rate limiting via bucket4j (rates stored in MySQL `priceeye.transaction_rates`). Packages raw HTTP responses into `PEHttpServiceResponse` objects and publishes them to `PEProvider{CODE}Response.fifo`. Also emits per-request audit records.

**Input**:
- SQS: `PEProvider{CODE}HttpService.fifo` — `PEHttpServiceRequest`
- MySQL: `priceeye.transaction_rates` (TPS limits per provider per hour)
- AWS Secrets Manager: provider credentials / API keys

**Output**:
- SQS: `PEProvider{CODE}Response.fifo` — `PEHttpServiceResponse` (raw provider response)
- Kinesis/Firehose: audit records (success/failure, latency)
- DLQ on failure: `FAILED-PEProvider{CODE}HttpService.fifo`

**Per-provider deployments** (defined as ECS task-definition instances of the base service):

| ECS definition | Memory | Tasks |
|---------------|--------|-------|
| `aa-http-service` | 64 GB | 56 |
| `as-http-service` | 64 GB | 56 |
| `atlas-http-service` | 64 GB | 56 |
| `ba-http-service` | 64 GB | 56 |
| `aapts-http-service` | 32 GB | 28 |
| `amp-http-service` | 32 GB | 28 |
| `dl-http-service` | 32 GB | 28 |
| `ha-http-service` | 32 GB | 28 |
| `la-http-service` | 32 GB | 28 |
| `lhg-http-service` | 32 GB | 28 |
| `mtc-http-service` | 32 GB | 28 |
| `ojt-http-service` | 32 GB | 28 |
| `ql2-http-service` | 32 GB | 28 |
| `sp-http-service` | 32 GB | 28 |
| `tp-http-service` | 32 GB | 28 |
| `ts-http-service` | 32 GB | 28 |
| `ua-http-service` | 32 GB | 28 |
| `vy-http-service` | 32 GB | 28 |
| `wl-http-service` | 32 GB | 28 |
| `wn-http-service` | 32 GB | 28 |

**Source**: `source/http-service/`

---

### http-polling-service (ECS Fargate Service)

**Type**: AWS ECS Fargate long-running service
**Trigger**: Continuously polls `PEProvider{CODE}HttpService.fifo`; same start/stop behavior as http-service
**Compute**: 32,768 MB RAM / 16,384 CPU units per task; 28 task instances
**Image**: `3victors/priceeyev2/http-polling-service`

**What it does**: Variant of http-service for providers (currently Skyscanner) that use session-based search: makes an initial request to begin a search session, receives a completion token, then polls repeatedly until results are ready. Manages token caching between poll cycles, handles session teardown, and enforces TPS rate limits. Publishes the completed response to `PEProviderSSResponse.fifo`.

**Input**:
- SQS: `PEProviderSSHttpPollingService.fifo` — `PEHttpServiceRequest` with polling configuration
- MySQL: `priceeye.transaction_rates`

**Output**:
- SQS: `PEProviderSSResponse.fifo` — `PEHttpServiceResponse`
- Kinesis/Firehose: audit records
- DLQ: `FAILED-PEProviderSSHttpPollingService.fifo`

**Per-provider deployments**:

| ECS definition | Memory | Tasks |
|---------------|--------|-------|
| `ss-http-polling-service` | 32 GB | 28 |

**Source**: `source/http-polling-service/`

---

### Response Handler Lambdas (one per real-time provider)

**Type**: AWS Lambda Function (arm64, image-based)
**Trigger**: SQS queue `PEProvider{CODE}Response.fifo` (batch size 10, max concurrency up to 256)
**Compute**: 624–1,536 MB, 120–300 s timeout
**Image pattern**: `3victors/priceeyev2/provider-lambdas-provider-{code}-response`

**What it does**: Triggered when `PEHttpServiceResponse` messages land on the response queue. Extracts the raw HTTP body, calls the provider-specific response parser (XML for TP/SP, JSON for most others), converts results into `PEItinerary` objects, and publishes a `PERequestResponse` to `PEGlobalFilter.fifo` for downstream pricing. Failed or errored responses are published to `PERetry.fifo` with retry metadata. Audit records are emitted to Kinesis/Firehose.

**Input**:
- SQS: `PEProvider{CODE}Response.fifo` — `PEHttpServiceResponse`
- S3 (optional): Large payload offloading via `PayloadS3Pointer`
- MySQL: provider site mappings, request correlation data

**Output**:
- SQS: `PEGlobalFilter.fifo` — `PERequestResponse` (parsed itineraries, consumed by downstream Global Filter)
- SQS: `PERetry.fifo` — `PERetryMessage` (for failures / retries)
- Kinesis/Firehose: response audit records (success/failed/timeout/no-result)

**Deployed instances**:

| Lambda name | Queue trigger | Memory | Timeout |
|-------------|--------------|--------|---------|
| `provider-aa-response` | `PEProviderAAResponse.fifo` | 624 MB | 300 s |
| `provider-aapts-response` | `PEProviderAAPtsResponse.fifo` | 624 MB | 120 s |
| `provider-amp-response` | `PEProviderAMPResponse.fifo` | 624 MB | 120 s |
| `provider-as-response` | `PEProviderASResponse.fifo` | 624 MB | 300 s |
| `provider-atlas-response` | `PEProviderAtlasResponse.fifo` | 624 MB | 300 s |
| `provider-ba-response` | `PEProviderBAResponse.fifo` | 624 MB | 120 s |
| `provider-dl-response` | `PEProviderDLResponse.fifo` | 624 MB | 120 s |
| `provider-ha-response` | `PEProviderHAResponse.fifo` | 624 MB | 120 s |
| `provider-la-response` | `PEProviderLAResponse.fifo` | 624 MB | 120 s |
| `provider-lhg-response` | `PEProviderLHGResponse.fifo` | 624 MB | 300 s |
| `provider-mtc-response` | `PEProviderMTCResponse.fifo` | 624 MB | 120 s |
| `provider-skyscanner-response` | `PEProviderSSResponse.fifo` | 624 MB | 120 s |
| `provider-sp-response` | `PEProviderSPResponse.fifo` | 624 MB | 120 s |
| `provider-tp-response` | `PEProviderTPResponse.fifo` | 928 MB | 120 s |
| `provider-ts-response` | `PEProviderTSResponse.fifo` | 624 MB | 120 s |
| `provider-ua-response` | `PEProviderUAResponse.fifo` | 624 MB | 300 s |
| `provider-vy-response` | `PEProviderVYResponse.fifo` | 624 MB | 120 s |
| `provider-wn-response` | `PEProviderWNResponse.fifo` | 624 MB | 120 s |
| `provider-wyndlabs-response` | `PEProviderWLResponse.fifo` | 624 MB | 120 s |

**Source**: `source/provider-lambdas/provider-{code}/response/`

---

### provider-tp-persistence (ECS Fargate Task)

**Type**: AWS ECS Fargate Task
**Trigger**: SQS queue `PETPPersistence.fifo`
**Compute**: 1,024 MB RAM / 1,024 CPU (1 vCPU), ARM64

**What it does**: Dedicated persistence task for Travelport (TP) data. Reads processed TP pricing data from its queue, optionally invokes a SageMaker endpoint for model inference, and writes partitioned results to Glue-cataloged S3 storage. Has additional IAM permissions for SageMaker `InvokeEndpoint` and Glue `GetPartitions`/`CreatePartition`.

**Input**:
- SQS: `PETPPersistence.fifo`

**Output**:
- S3 + Glue: partitioned Travelport pricing data
- Glue: creates/updates partitions

**Source**: `source/` (ECS task, no dedicated source subfolder; configured via `source/deploy/commonfiles/provider-tp-persistence.yaml`)

---

### Batch Provider: AI (Air India)

**Type**: AWS Lambda Function + S3/EventBridge
**Trigger (request)**: S3 object-created event on `s3-atp-3victors{env}-use1-pe-batch-providers` with key `AI/requests/*.json`
**Trigger (response)**: S3 object-created event on `s3-atp-3victors{env}-use1-pe-aggregate-intelligence` with key `output/*/Success*.gz`
**Compute**: Request lambda 624 MB / 120 s; Response handler 4,096 MB / 315 s

**What it does**: The AI provider uses a file-based batch model rather than real-time HTTP. When a request JSON is deposited in the batch-providers S3 bucket under `AI/requests/`, the `provider-ai-request` lambda registers the batch job. After the external AI batch system processes the job and writes compressed output to `s3-pe-aggregate-intelligence/output/*/Success*.gz`, the `provider-ai-response-handler` lambda is triggered, reads and decompresses the file, parses itineraries, and forwards them to `PEGlobalFilter.fifo`. Processed files are archived to `output/{date}/success/`.

**Input**:
- S3: `s3-atp-3victors{env}-use1-pe-batch-providers/AI/requests/*.json` (request trigger)
- S3: `s3-atp-3victors{env}-use1-pe-aggregate-intelligence/output/*/Success*.gz` (response trigger)
- MySQL: provider request/audit correlation

**Output**:
- SQS: `PEGlobalFilter.fifo` — parsed itineraries
- SQS: `PERetry.fifo` — failures
- S3 archive: `s3-pe-aggregate-intelligence/output/{date}/success/` or `error/`
- Kinesis/Firehose: audit records

**Error handling**:
- `provider-ai-error-response-checker` (2,048 MB / 270 s): triggered by `output/*/Error*` in the aggregate-intelligence bucket; analyzes error files
- `provider-ai-error-file-response-handler` (512 MB / 60 s): triggered by `PEProviderAIErrorFileResponse.fifo`; processes individual error file messages

**Source**: `source/common-ai-response-handler/`, `source/provider-lambdas/provider-ai/`

---

### Batch Provider: QL2 (Galileo/Amadeus)

**Type**: AWS Lambda Function + S3/EventBridge
**Trigger (request)**: S3 object-created event on `s3-atp-3victors{env}-use1-pe-batch-providers` with key `QL2/requests/*.json`
**Trigger (response)**: S3 object-created event on `s3-atp-3victors{env}-use1-pe-ql2-transfer-bucket` with suffix `_out.csv` or `_rerun.csv`
**Compute**: Request lambda 624 MB / 120 s; Response handler 1,024 MB / 315 s

**What it does**: Similar batch model to AI but uses CSV output from Galileo/Amadeus's QL2 system. The `provider-ql2-request` lambda registers the job when a request JSON appears in S3. Once QL2 writes a `_out.csv` (success) or `_rerun.csv` (rerun) to the QL2 transfer bucket, the `provider-ql2-response-handler` lambda parses the CSV — extracting flight itineraries, fares, carriers, and routing — and publishes to `PEGlobalFilter.fifo`. Error CSVs trigger the error checker flow.

**Input**:
- S3: `s3-atp-3victors{env}-use1-pe-batch-providers/QL2/requests/*.json`
- S3: `s3-atp-3victors{env}-use1-pe-ql2-transfer-bucket/*_out.csv` or `*_rerun.csv`
- MySQL: site mappings, request correlation

**Output**:
- SQS: `PEGlobalFilter.fifo` — `PERequestResponse` with itineraries
- SQS: `PERetry.fifo` — failed/timeout requests
- S3 archive: `ql2-transfer-bucket/v1/{year}/{month}/{day}/success/` or `error/`
- Kinesis/Firehose: audit records

**Error handling**:
- `provider-ql2-error-response-checker` (2,048 MB / 270 s): EventBridge triggered on S3 error files
- `provider-ql2-error-file-response-handler` (512 MB / 60 s — extended to 600 s in artifacts): SQS triggered via `PEProviderQL2ErrorFileResponse.fifo`

**Source**: `source/common-ql2-response-handler/`, `source/provider-lambdas/provider-ql2/`

---

### Batch Provider: PIT (PITOBI)

**Type**: AWS Lambda Function + S3/EventBridge
**Trigger (request)**: S3 object-created event on `s3-atp-3victors{env}-use1-pe-batch-providers` with key `PIT/requests/*.json`
**Trigger (response)**: S3 object-created event on `s3-atp-3victors{env}-use1-pe-pitobi` with key `output/*.csv`
**Compute**: Request lambda 624 MB / 120 s; Response handler 4,096 MB / 315 s (6 GB in artifacts)

**What it does**: Batch pricing integration with the PITOBI system. Request files landing in the batch-providers S3 bucket under `PIT/requests/` trigger the `provider-pit-request` lambda to register the job. When PITOBI deposits output CSV files in the pitobi S3 bucket, the `provider-pit-response-handler` lambda reads and parses the CSV, constructs itinerary objects, and forwards to `PEGlobalFilter.fifo`.

**Input**:
- S3: `s3-atp-3victors{env}-use1-pe-batch-providers/PIT/requests/*.json`
- S3: `s3-atp-3victors{env}-use1-pe-pitobi/output/*.csv`

**Output**:
- SQS: `PEGlobalFilter.fifo`
- SQS: `PERetry.fifo`
- Kinesis/Firehose: audit records

**Source**: `source/provider-lambdas/provider-pit/`

---

### Ingest Lambdas: Delta, Southwest, Estream

**Type**: AWS Lambda Function (arm64, image-based)
**Trigger**: SQS queue — `PEProviderIngestDelta.fifo`, `PEProviderIngestSouthwest.fifo`, or `PEProviderIngestEstream.fifo` (batch size 10)
**Compute**: 2,048 MB / 120 s (per artifacts manifest)

**What it does**: These providers integrate with an internal Redis cache ("ingest cache") rather than external APIs. The scheduler places `PERequestBundle` search requests on the ingest queue. The lambda deserializes the bundle, performs a Redis lookup (`startSearch` / `finishSearch` lifecycle), and if results are found, parses them into `PERequestResponse` objects and publishes to `PEGlobalFilter.fifo`. Missing or expired cache entries go to `PERetry.fifo`. Supports TPFC (Trifecta Pricing and Fare Cache) verification via a second Redis instance. Private-market carrier mappings are loaded from MySQL.

**Input**:
- SQS: `PEProviderIngest{CODE}.fifo` — `PERequestBundle`
- S3 (optional): `PayloadS3Pointer` offloading for large bundles
- Redis: two instances (with TPFC and without TPFC)
- MySQL: provider config, private-market carrier maps

**Output**:
- SQS: `PEGlobalFilter.fifo` — `PERequestResponse`
- SQS: `PERetry.fifo` — `PERetryMessage`
- Kinesis/Firehose: audit records

**Deployed instances**:

| Lambda | Queue |
|--------|-------|
| `provider-ingest-delta` | `PEProviderIngestDelta.fifo` |
| `provider-ingest-southwest` | `PEProviderIngestSouthwest.fifo` |
| `provider-ingest-estream` | `PEProviderIngestEstream.fifo` |

**Source**: `source/provider-ingest-lambdas/`

---

### dlq-processor (Dead Letter Queue Processor)

**Type**: AWS Lambda Function (arm64, image-based)
**Trigger**: Multiple DLQs (batch size 10, max concurrency 8 per queue)
**Compute**: 624 MB / 120 s

**What it does**: Centralized handler for messages that exhausted their retry budget on the HttpService queues. Reads from the `FAILED-PEProvider{CODE}HttpService.fifo` queues, inspects each failed message, and performs recovery actions (re-queuing, logging, or discarding based on error type).

**Input** (DLQs monitored):
- `FAILED-PEProviderAAHttpService.fifo`
- `FAILED-PEProviderAAPtsHttpService.fifo`
- `FAILED-PEProviderQL2HttpService.fifo`
- `FAILED-PEProviderSPHttpService.fifo`
- `FAILED-PEProviderTPHttpService.fifo`
- `FAILED-PEProviderTSHttpService.fifo`

**Source**: `source/deploy/commonfiles/dlq-processor.yaml`

---

### provider-dummy-lambda

**Type**: AWS Lambda Function
**Trigger**: SQS queue `PEDummy.fifo` (batch size 10, max concurrency 32)
**Compute**: 624 MB / 120 s

**What it does**: Test/dummy provider used for integration testing and pipeline validation. Reads `PERequestBundle` messages, generates synthetic itinerary responses, and publishes to `PEGlobalFilter.fifo` and `PERetry.fifo` without making any real API calls.

**Source**: `source/provider-dummy/`

---

### provider-atlas-routes (Scheduled ECS Task)

**Type**: AWS ECS Fargate Task
**Trigger**: EventBridge cron — `cron(25 1 ? * MON *)` = **every Monday at 01:25 UTC**
**Compute**: 4,096 MB RAM / 1,024 CPU, 2 task instances

**What it does**: Weekly maintenance job that refreshes Atlas carrier route data in MySQL. Reads valid routes (where `scheduleEnd >= today`) from the `atlas_routes` table, selects the best route per carrier (preferring most-recent `scheduleEnd`, preferring direct routes), then updates `origin_airport_code` and `destination_airport_code` in `priceeye.site_dictionary_requests` for all Atlas-provider rows where the site code matches a valid carrier.

**Input**:
- MySQL: `atlas_routes` table (valid routes by carrier)
- MySQL: `priceeye.site_dictionary_requests` (Atlas provider entries)

**Output**:
- MySQL: updated `origin_airport_code` / `destination_airport_code` in `site_dictionary_requests`

**Source**: `source/atlas-routes/`

---

### provider-ql2-response-replay (Scheduled ECS Task)

**Type**: AWS ECS Fargate Task
**Trigger**: EventBridge cron — `cron(*/15 * * * ? *)` = **every 15 minutes**
**Compute**: 4,096 MB RAM / 1,024 CPU, 2 task instances

**What it does**: Continuously polls the `FAILED-ql2-response-handler` SQS DLQ for messages containing S3 bucket/key references to QL2 response files that previously failed processing. For each message, it retrieves the referenced S3 object and runs it back through the `QL2ResponseHandler` — re-parsing the CSV and re-publishing results to `PEGlobalFilter.fifo` or `PERetry.fifo`.

**Input**:
- SQS: `FAILED-ql2-response-handler` DLQ
- S3: QL2 CSV response files (`_out.csv`, `_error.csv`, `_rerun.csv`)

**Output**:
- SQS: `PEGlobalFilter.fifo` / `PERetry.fifo` (via QL2ResponseHandler)
- S3: archives processed files to `v1/{year}/{month}/{day}/success/` or `error/`

**Source**: `source/provider-ql2/response-replay/`

---

### site-capabilities (Scheduled ECS Task)

**Type**: AWS ECS Fargate Task
**Trigger**: EventBridge cron — `cron(0 9 ? * MON *)` = **every Monday at 09:00 UTC**
**Compute**: 4,096 MB RAM / 1,024 CPU, 2 task instances
**Default schedule** (from site-capabilities.yaml): `cron(0,25 * * * ? *)` = every hour at :00 and :25

**What it does**: Probes provider booking websites to detect their supported search capabilities (cabin class selection, refundable fares, advance purchase limits, etc.). Loads test search configurations from `priceeye.site_dictionary_requests` in MySQL, executes test searches against each provider/site combination using the appropriate `PEHttpCommunicator` (or `PEHttpPollingServiceApplication` for Skyscanner), analyzes the response for `CapabilityType` signals, and writes `PESiteCapabilityDetection` records back to MySQL. Sends a Slack summary report on completion.

**Input**:
- MySQL: `priceeye.site_dictionary_requests` (test request configs)
- MySQL: site capability detection metadata
- HTTP: provider APIs (direct)

**Output**:
- MySQL: `priceeye.site_capability_detections` — detected capability records
- Slack: summary report of detection results

**Source**: `source/site-capabilities/`

---

## Shared Libraries

| Module | Purpose |
|--------|---------|
| `source/provider-common/` | Abstract base classes: `PERequestBuilder`, `PEResponseParser`, `PEHttpCommunicator`, `PELifecycle` |
| `source/provider-tp-common/` | Shared Travelport XML response parsing (`ProviderTPResultHandler`, `TravelportXMLReader`) |
| `source/common-ai-response-handler/` | Shared AI batch response processing (`AIResponseHandler`) |
| `source/common-ql2-response-handler/` | Shared QL2 CSV parsing logic (`QL2ResponseHandler`, `PESuccessFileReader`, `PEErrorFileReader`, `PEItineraryBuilder`) |
| `source/dao/` | Database access objects: `PriceEyeProvidersReader`, `PriceEyeProvidersMetadataReader`, `PriceEyeProvidersTrackingReader`, `PEProvidersCommonOutputReader` (MySQL + Redshift) |
| `source/data/` | Domain model: `PESiteCapabilityDetection`, `PESiteDictionaryRequestRow`, and other DTOs |
| `source/atpco-fare-lookup/` | ATPCO fare lookup utilities |
| `source/external-data/` | External data integration utilities |

---

## SQS Queue Inventory

All queues are **FIFO** with content-based deduplication, 24-hour message retention, 20-second long-polling, and `maxReceiveCount: 4` before DLQ routing (QL2HttpService uses 5).

### Per-provider queues (standard HTTP providers)

Each real-time HTTP provider has three queues:

| Queue | Purpose | Visibility Timeout |
|-------|---------|-------------------|
| `PEProvider{CODE}Request.fifo` | Scheduler → Request Lambda | 900 s |
| `PEProvider{CODE}HttpService.fifo` | Request Lambda → HTTP Service | 900 s |
| `PEProvider{CODE}Response.fifo` | HTTP Service → Response Lambda | 900 s |

Plus corresponding `FAILED-*` DLQs for each.

### Special queues

| Queue | Purpose |
|-------|---------|
| `PEProviderIngestDelta.fifo` | Delta ingest requests |
| `PEProviderIngestEstream.fifo` | Estream/Southwest ingest requests |
| `PEProviderIngestSouthwest.fifo` | Southwest ingest requests |
| `PEProviderAIErrorFileResponse.fifo` | AI error file processing |
| `PEProviderQL2ErrorFileResponse.fifo` | QL2 error file processing |
| `PEQL2Vacation.fifo` | QL2 vacation/package pricing |
| `PETPPersistence.fifo` | Travelport persistence task |
| `PEDummy.fifo` | Dummy/test provider |

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| Lambda Functions | ~52 (request + response lambdas for 19 providers + batch/error handlers + dummy) |
| ECS Fargate Services (http-service) | 20 per-provider deployments of `http-service` |
| ECS Fargate Services (http-polling-service) | 1 (`ss-http-polling-service`) |
| ECS Scheduled Tasks | 3 (`provider-atlas-routes`, `provider-ql2-response-replay`, `site-capabilities`) |
| ECS Ad-hoc Tasks | 1 (`provider-tp-persistence`) |
| SQS FIFO Queues (main) | ~65 (3 per standard provider + ingest + special) |
| SQS DLQs | ~30+ (`FAILED-*` queues per provider) |
| EventBridge Rules | ~10 (S3 triggers for batch providers AI, QL2, PIT + cron schedules) |
| Glue Databases | 0 (no Glue tables in this repo; data is forwarded to PEGlobalFilter) |
| Step Functions | 0 |

---

## Key S3 Buckets

| Bucket | Usage |
|--------|-------|
| `s3-atp-3victors{env}-use1-pe-batch-providers` | Input: AI (`AI/requests/`), QL2 (`QL2/requests/`), PIT (`PIT/requests/`) request files |
| `s3-atp-3victors{env}-use1-pe-aggregate-intelligence` | AI response output (`output/*/Success*.gz`, `output/*/Error*`) |
| `s3-atp-3victors{env}-use1-pe-ql2-transfer-bucket` | QL2 output CSVs (`*_out.csv`, `*_rerun.csv`) |
| `s3-atp-3victors{env}-use1-pe-pitobi` | PIT/PITOBI output CSVs (`output/*.csv`) |

---

## Key MySQL Tables

| Table | Usage |
|-------|-------|
| `priceeye.transaction_rates` | TPS limits per provider per hour (used by http-service / http-polling-service) |
| `priceeye.site_dictionary_requests` | Provider test request configurations (site-capabilities, atlas-routes) |
| `priceeye.site_capability_detections` | Detected site capabilities (written by site-capabilities) |
| `atlas_routes` | Atlas carrier valid route data (read/written by provider-atlas-routes) |

---

## Downstream Output

All parsed itinerary results from every provider converge on two SQS queues consumed by a separate downstream system:

| Queue | Description |
|-------|-------------|
| `PEGlobalFilter.fifo` | Successful `PERequestResponse` objects containing parsed `PEItinerary` lists — fed into the PriceEye Global Filter / pricing pipeline |
| `PERetry.fifo` | `PERetryMessage` objects for requests that failed or timed out — consumed by the retry scheduler |
