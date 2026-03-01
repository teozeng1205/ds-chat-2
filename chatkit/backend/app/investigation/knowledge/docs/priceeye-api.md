# priceeye-api

> Serverless backend API platform for the PriceEye airline pricing intelligence product, providing REST endpoints for customer configuration, scheduling, segmentation, reporting, and proof-of-concept management.

> **Current branch**: `develop` — this document reflects the `develop` branch. The `master`/`main` branch represents what is running in production; the documented state may differ.

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────────────┐
                        │  Client / PriceEye UI (browser or internal tool) │
                        └──────────────────────┬──────────────────────────┘
                                               │ HTTPS
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
              ┌───────────────────┐ ┌──────────────────┐ ┌──────────────────┐
              │ API Gateway       │ │ API Gateway       │ │ API Gateway       │
              │ PE-MDW-API        │ │ PE-BCK-API        │ │ Health-Deployment │
              │ (Middleware tier) │ │ (Backend/DAO tier)│ │ (/pulse)          │
              └────────┬──────────┘ └────────┬──────────┘ └────────┬─────────┘
                       │                     │                      │
              JWT auth via ums-authorizer    │              API-key auth
                       │                     │
              ┌────────▼──────────┐ ┌────────▼──────────┐ ┌────────▼──────────┐
              │ Middleware Lambdas │ │ Backend Lambdas    │ │ Health Lambda      │
              │ (~22 functions)   │ │ (~25 functions)    │ │ pulse-lambda       │
              │ (business logic + │ │ (DAO layer direct  │ │ checks MySQL +     │
              │  proxying)        │ │  MySQL/Redshift)   │ │ Redshift health    │
              └────────┬──────────┘ └────────┬──────────┘ └───────────────────┘
                       │  HTTP               │
                       └─────────────────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
              ┌─────────────┐       ┌────────────────┐
              │ MySQL/Aurora│       │    Redshift     │
              │(transact.)  │       │  (analytics,    │
              └─────────────┘       │   read-only)    │
                                    └────────────────┘

  ┌────────────────────────────────────────────────────────────────────────┐
  │  Standalone scheduled / event-driven Lambdas                           │
  │                                                                        │
  │  [EventBridge cron: daily 01:15 UTC]                                   │
  │       │                                                                │
  │       ▼                                                                │
  │  [sales-poc-expiry Lambda]  ──► MySQL (expire POC records)             │
  │                                                                        │
  │  [EventBridge: NotificationSystemBus (source prefix "pe-api-*")]       │
  │       │                                                                │
  │       ▼                                                                │
  │  [notification-processor Lambda] ──► SES (send emails)                │
  │                                   ──► SQS: PENotificationRequest.fifo  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## API Layers

The system is split into two independently deployed API Gateway + Lambda tiers that work together:

### Middleware API (`PE-MDW-API`)
- **Purpose**: The public-facing tier consumed by the PriceEye UI. Handles JWT-based authorization via the `ums-authorizer` Lambda custom authorizer. Applies business logic, RBAC enforcement (injects the caller's customer code for non-admin users), and proxies upstream calls to the Backend API.
- **Auth**: Bearer JWT token decoded in each handler. Roles checked: `role.api_priceeye.admin` (full access) vs. `role.api_priceeye.customer` (own org only).
- **Throttle**: Burst 1000 / Rate 1000 req/s.
- **~22 Lambda functions** — one per domain (see Components section).

### Backend API (`PE-BCK-API`)
- **Purpose**: Internal DAO-tier API. Receives proxied requests from Middleware Lambdas and talks directly to MySQL/Redshift via the DAO module. Also exposed as a private API Gateway (VPC endpoint only) secured with WAFv2.
- **Auth**: API key header + UMS Authorizer.
- **Throttle**: Burst 1000 / Rate 1000 req/s.
- **~25 Lambda functions** — one per domain.

### Health API (`Health-Deployment`)
- **Purpose**: Liveness/readiness check. Single route `GET /pulse`. Checks row counts in Redshift materialized views and MySQL collections. Returns HTTP 200 (OK), 204 (unexpected), or 503 (error).
- **Auth**: API key header only.
- **Throttle**: Burst 5 / Rate 5 req/s (intentionally very low).

---

## Components

All Lambda functions share these infrastructure characteristics unless noted:
- **Runtime**: Java 17 container image on ARM64 (Graviton), deployed from ECR (`732267085676.dkr.ecr.us-east-1.amazonaws.com`)
- **Memory**: 624 MB (default)
- **VPC**: Connected to VPC subnets (`SubnetApp0/1/2`) with `FMSSecuritygroupApp`
- **Logging**: CloudWatch Logs group `priceeyeapi/<stack-name>`, retention 7 days
- **Alarms**: CloudWatch alarm fires to `HighPriorityAlarm` SNS topic on timeout
- **IAM**: S3 Full, SQS Full, Kinesis Full, Lambda VPC execution, SecretsManager, SES

---

### Backend API Lambdas (`priceeye-api-api-*`)

Each backend Lambda implements `RequestStreamHandler`, extending the `APIHandlerSkeleton` base class. They receive `APIGatewayProxyRequestEvent` from API Gateway, decode the JWT, and dispatch to the appropriate DAO method. All functions connect to MySQL (via connection pool) and/or Redshift.

| Lambda Function | API Routes | Purpose |
|----------------|-----------|---------|
| `priceeye-api-api-customers` | `/customers`, `/customer_site_codes` | Customer CRUD, billing, site codes |
| `priceeye-api-api-customer-defaults` | `/customer_defaults` | Customer feature flags (analytics, monitoring, channel_comparison) |
| `priceeye-api-api-customer-collections` | `/customer_collections` | Per-customer data collection configurations |
| `priceeye-api-api-customer-delivery` | `/customer_delivery` | Customer delivery configuration |
| `priceeye-api-api-customer-packaging` | `/customer_packaging` | Customer packaging options |
| `priceeye-api-api-inputs` | `/inputs`, `/input_actions`, `/volume_check`, `/capacity_check` | Data input definitions, volume/capacity checks |
| `priceeye-api-api-input-formats` | `/input_formats` | Input file format specifications |
| `priceeye-api-api-output-formats` | `/output_formats` | Report output format specifications |
| `priceeye-api-api-scheduling` | `/site_metrics`, `/market_blacklist_summary`, `/site_mapping`, `/retry_rates`, `/cache_rates`, `/import_rates`, `/date_range` | Price update scheduling, cache/retry rate configuration |
| `priceeye-api-api-segmentation` | `/cabin_group`, `/carrier_group`, `/geography`, `/geography_entry`, `/region`, `/segment`, `/cabins` | Market segmentation — cabin groups, carrier groups, geographic segments |
| `priceeye-api-api-provider-configuration` | `/provider`, `/provider_ss_config`, `/provider_cabin_mapping`, `/provider_pos_sitemap`, `/transformation_rules`, `/capability_definition`, `/site_capabilities`, `/site_carriers`, `/travelport_pccs`, `/retry_substitution`, `/valid_substitution_sites`, `/enrichment_sites`, `/joint_business` | Airline provider setup, POS mappings, transformation rules |
| `priceeye-api-api-dashboards` | `/dashboards` | Dashboard configuration CRUD; triggers `Generate-Dashboard` Step Function on create |
| `priceeye-api-api-reporting` | `/channel_comparison`, `/chnl_comp_download` | Reporting queries against Redshift analytics |
| `priceeye-api-api-poc` | `/poc_requests`, `/poc_delivery` | Proof-of-concept request management |
| `priceeye-api-api-poc-wizard` | `/poc_wizard` | POC wizard workflow backend |
| `priceeye-api-api-autoscheduler` | `/auto_schedule_generations`, `/site_dictionary_requests` | Auto-scheduler configuration |
| `priceeye-api-api-autoscheduler-details` | `/autoscheduler_details` | Auto-scheduler detail records |
| `priceeye-api-api-site-hierarchy` | `/site_hierarchy` | Hierarchical site/airport organization |
| `priceeye-api-api-site-dictionary` | `/sites` | Site dictionary (airport/city lookup) |
| `priceeye-api-api-form-filters` | `/filters` | UI form filter definitions |
| `priceeye-api-api-help-docs` | `/help_docs` | Help documentation retrieval; reads from S3 `s3-atp-3victors{env}-use1-help-docs-assets` |
| `priceeye-api-api-release-notes` | `/release-notes` | Release notes management |
| `priceeye-api-api-system` | `/enrichment_sites`, `/retry_substitution`, `/travelport_pccs`, `/joint_business`, `/valid_substitution_sites`, `/customer_imports` | System-level reference data |
| `priceeye-api-api-vacations` | `/vacations_compare`, `/vacations_upload`, `/vacations_source`, `/vacations_source_input` | Vacation scheduling data upload and comparison |

---

### Middleware API Lambdas (`priceeye-api-middleware-*`)

Middleware Lambdas mirror the backend layer. Each one:
1. Receives the UI request with the user's JWT
2. Validates the JWT and enforces RBAC
3. Optionally injects the caller's org code (for non-admin users)
4. Proxies the request to the corresponding Backend API Lambda via HTTP (`ProxyBackendClient`)

| Lambda Function | Backed by Backend Lambda |
|----------------|------------------------|
| `priceeye-api-middleware-customers` | `priceeye-api-api-customers` |
| `priceeye-api-middleware-customer-collections` | `priceeye-api-api-customer-collections` |
| `priceeye-api-middleware-customer-delivery` | `priceeye-api-api-customer-delivery` |
| `priceeye-api-middleware-customer-packaging` | `priceeye-api-api-customer-packaging` |
| `priceeye-api-middleware-scheduling` | `priceeye-api-api-scheduling` |
| `priceeye-api-middleware-segmentation` | `priceeye-api-api-segmentation` |
| `priceeye-api-middleware-provider-configuration` | `priceeye-api-api-provider-configuration` |
| `priceeye-api-middleware-dashboards` | `priceeye-api-api-dashboards` |
| `priceeye-api-middleware-reporting` | `priceeye-api-api-reporting` |
| `priceeye-api-middleware-poc` | `priceeye-api-api-poc` |
| `priceeye-api-middleware-poc-wizard` | `priceeye-api-api-poc-wizard` |
| `priceeye-api-middleware-autoscheduler` | `priceeye-api-api-autoscheduler` |
| `priceeye-api-middleware-autoscheduler-details-mw` | `priceeye-api-api-autoscheduler-details` |
| `priceeye-api-middleware-site-hierarchy-mw` | `priceeye-api-api-site-hierarchy` |
| `priceeye-api-middleware-site-dictionary` | `priceeye-api-api-site-dictionary` |
| `priceeye-api-middleware-form-filters` | `priceeye-api-api-form-filters` |
| `priceeye-api-middleware-help-docs` | `priceeye-api-api-help-docs` |
| `priceeye-api-middleware-release-notes-mw` | `priceeye-api-api-release-notes` |
| `priceeye-api-middleware-system` | `priceeye-api-api-system` |
| `priceeye-api-middleware-input-formats` | `priceeye-api-api-input-formats` |
| `priceeye-api-middleware-output-formats` | `priceeye-api-api-output-formats` |
| `priceeye-api-middleware-vacations` | `priceeye-api-api-vacations` |

---

### Standalone / Scheduled Lambdas

#### `priceeye-api-health-pulse-lambda`

**Type**: Lambda Function
**Trigger**: `GET /pulse` on Health API Gateway
**Compute**: 624 MB, 29 s timeout

**What it does**: Performs a composite health check against the two primary data stores. Queries Redshift for the count of materialized views and MySQL for the count of active customer collections. Returns HTTP 200 (OK), 204 (data count unexpected), or 503 (query failed/error).

**Input**: Internal DB queries
**Output**: JSON health status response

---

#### `priceeye-api-sales-poc-expiry`

**Type**: Lambda Function
**Trigger**: EventBridge cron — daily at 01:15 UTC (`cron(15 1 * * ? *)`)
**Compute**: 624 MB, 60 s timeout

**What it does**: Scheduled job that scans open sales proof-of-concept records in MySQL and marks expired ones as closed based on their configured expiration date. Runs nightly to keep POC lifecycle state current.

**Input**: MySQL `poc_requests` table (via DAO)
**Output**: MySQL writes (status updates)

---

#### `notification-processor-lambda`

**Type**: Lambda Function
**Trigger**: EventBridge rule on `NotificationSystemBus` — matches any event with source prefix `pe-api-*`
**Compute**: 624 MB, 60 s timeout
**IAM Extra**: `ses:SendEmail`, `ses:ListIdentities`

**What it does**: Processes notification events emitted by the PriceEye API layer (e.g., when a customer configuration changes or a POC is approved). Extends the shared `AbstractNotificationIntakeLambda` framework. On success, routes notifications via SES (email) and writes to the `PENotificationRequest.fifo` SQS queue. Failed messages are sent to the `FAILED-PENotificationRequest.fifo` dead-letter queue after 4 retries.

**Input**: EventBridge events on `NotificationSystemBus` (source: `pe-api-*`)
**Output**:
- SES: email delivery
- SQS: `PENotificationRequest.fifo` (FIFO, per-message-group deduplication)
- SQS DLQ: `FAILED-PENotificationRequest.fifo`

---

#### `ums-authorizer`

**Type**: Lambda Function (API Gateway Custom Authorizer)
**Trigger**: API Gateway authorizer invocation on both Middleware and Backend API Gateways
**Compute**: Not a Maven module — built separately with its own Dockerfile and Makefile (`source/ums-authorizer/`)

**What it does**: Validates JWT Bearer tokens presented to the Middleware API Gateway. Extracts claims (`sub`, `email`, `pe_3v_ccode`, `roles`, `scp`) and returns an IAM policy to API Gateway allowing or denying the request. Acts as the authentication gate for the entire middleware tier.

**Input**: JWT from `Authorization: Bearer <token>` header
**Output**: IAM policy document (Allow/Deny) returned to API Gateway

---

## Shared Library Modules

These are Maven JAR modules consumed by all Lambda functions, not deployed independently.

| Module | Purpose |
|--------|---------|
| `source/common` | Base classes: `APIHandlerSkeleton`, `Router`, `ProxyBackendClient`, `KnownRoles`, `BodyValidator`, `APIResponse` — the foundation for every Lambda handler |
| `source/data` | All DTOs/domain objects: `PECustomer`, `PEInput`, `PESchedule`, `AuthPayload`, `RequestPermission`, `ApiErrorException`, etc. |
| `source/dao` | Database access: `PriceEyeReader` (~6,500 lines), `PriceEyeAPIWriter` (~4,700 lines), `RedshiftAnalyticsReader`, `NotificationReader/Writer`. Manages MySQL + Redshift connection pools |
| `source/data-parsing` | Pricing data parsing and transformation logic, wrapping the `threevictors-priceeye-data` library |
| `source/input-common` | Common input validation and processing shared across input-related API modules |
| `source/plan-summary-reports` | Report generation logic — produces plan summaries, writes output to S3 |
| `source/website-price-eye` | Legacy Apache Struts 2 web application (WAR) for browser-based access; includes provider-specific request handling for 20+ airlines |

---

## Authentication & Authorization

```
Request
  │
  ├──[Middleware API Gateway]
  │      │
  │      ▼
  │  UMS Authorizer Lambda
  │      │  decode JWT Bearer token
  │      │  validate signature
  │      │  return IAM Allow/Deny
  │      │
  │      ▼  (if Allow)
  │  Middleware Lambda
  │      │  extract RequestPermission from JWT payload:
  │      │    orgCode   = pe_3v_ccode claim
  │      │    roles     = roles claim
  │      │    subUUID   = sub claim
  │      │
  │      │  RBAC check:
  │      │    admin role   → full access, any orgCode
  │      │    customer role→ inject caller's orgCode, restrict to own data
  │      │
  │      └──► HTTP proxy to Backend API
  │
  └──[Backend API Gateway]
         │  API key + UMS Authorizer
         ▼
     Backend Lambda → DAO → MySQL / Redshift
```

**Roles**:
- `role.api_priceeye.admin` — full access; can operate on any customer
- `role.api_priceeye.customer` — scoped to own organization code

---

## Data Persistence

### MySQL / Aurora (Transactional)

Primary database for all configuration and operational data. Connection pooling via `threevictors-common-database-data-access-*` with separate reader/writer pools.

**Key tables** (from `docs/priceeye.sql`):

| Table | Description |
|-------|-------------|
| `customer` | Customer master records |
| `customer_billing` | Billing and subscription details |
| `customer_defaults` | Per-customer feature flags (analytics, monitoring, channel_comparison) |
| `customer_collection` | Customer-specific data collection configurations |
| `customer_delivery` | Data delivery settings |
| `customer_packaging` | Output packaging options |
| `input` | Pricing data input definitions |
| `input_details` | Input metadata and parameters |
| `file_formats` | Input file format specifications |
| `output_file_formats` | Report output format specifications |
| `provider` | Airline provider definitions |
| `site` | Airport/city site definitions |
| `provider_pos_sitemap` | Provider → Point-of-Sale site mappings |
| `scheduling` | Price update scheduling rules |
| `cache_rates` | Cached pricing rate configurations |
| `retry_rates` | Retry logic parameters |
| `transformation_rules` | Data transformation rules |
| `cabin_group` | Cabin class groupings |
| `site_hierarchy` | Hierarchical site organization |
| `contacts` | User contact records |
| `dashboard_details` | Dashboard configurations |
| `poc_requests` | Proof-of-concept requests and status |

### Redshift (Analytics, read-only)

Used for analytics queries served by the reporting endpoint (`/channel_comparison`, `/chnl_comp_download`) and health checks. Accessed via `RedshiftAnalyticsReader`.

---

## S3 Buckets

| Bucket | Purpose |
|--------|---------|
| `s3-atp-3victors{env}-use1-notification-templates` | Notification email templates; EventBridge notifications enabled |
| `s3-atp-3victors{env}-use1-help-docs-assets` | Help documentation static assets served by `help-docs` Lambda |
| `s3-atp-3victors{env}-use1-cloudformation` | CloudFormation artifacts and OpenAPI specs (deployment use) |

---

## SQS Queues

| Queue | Type | Purpose |
|-------|------|---------|
| `PENotificationRequest.fifo` | FIFO, 15 min visibility, 24h retention | Incoming notification requests; max 4 delivery attempts |
| `FAILED-PENotificationRequest.fifo` | FIFO (DLQ) | Dead-letter queue for notifications that exceeded 4 retries |

---

## Infrastructure Summary

| Resource | Count | Notes |
|----------|-------|-------|
| API Gateways | 3 | PE-BCK-API (private, WAFv2), PE-MDW-API (public), Health-Deployment |
| Lambda Functions | ~50 | ~25 backend + ~22 middleware + 3 standalone + 1 authorizer |
| S3 Buckets | 2 operational | notification-templates, help-docs-assets |
| SQS Queues | 2 | PENotificationRequest.fifo + DLQ |
| EventBridge Rules | 2 | sales-poc-expiry (cron) + notification-processor (event pattern) |
| CloudWatch Alarms | ~50 | One per Lambda, fires on timeout to HighPriorityAlarm SNS topic |
| WAFv2 WebACL | 1 | Associated with PE-BCK-API (Backend API Gateway) |
| ECR Repository | 1 | `732267085676.dkr.ecr.us-east-1.amazonaws.com/3victors/priceeyev2/` |

---

## Build & Deployment

**Stack**: Java 17, Maven multi-module (`pom.xml` at root, version `0.24-SNAPSHOT`).

**Key build plugins**:
- `maven-shade-plugin` — fat JARs for Lambda deployment
- `maven-assembly-plugin` — packaged JARs with manifest
- `fabric8 docker-maven-plugin` — build and push Lambda container images to ECR

**Deployment scripts** (`source/common-scripts/`):

| Script | Purpose |
|--------|---------|
| `deploy.sh` | Main deployment: builds Maven, pushes Docker image, updates CloudFormation stacks |
| `deploy_lambda_version.sh` | Promote a specific Lambda version/alias |
| `refresh_lambda_image.sh` | Update a Lambda function's container image URI in place |
| `upload-api-gateway.sh` | Upload OpenAPI spec YAML to S3 then redeploy API Gateway stage |

**CloudFormation templates** (`source/common-scripts/`):

| Template | Purpose |
|----------|---------|
| `yaml/backend-api-{dev,prod}.yaml` | API Gateway for Backend API (private + WAFv2 on prod) |
| `yaml/middleware-api-{dev,prod}.yaml` | API Gateway for Middleware API |
| `yaml/health-api-{dev,prod}.yaml` | API Gateway for Health endpoint |
| `yaml/priceeye-api-buckets.yaml` | S3 bucket definitions |
| `yaml/priceeye-api-queues.yaml` | SQS FIFO queue definitions |
| `commonfiles/lambda.yaml` | Reusable Lambda function template (API-connected) |
| `commonfiles/priceeye-api-sales-poc-expiry.yaml` | Sales POC expiry scheduled Lambda |
| `commonfiles/priceeye-api-notifications-notification-processor-lambda.yaml` | Notification processor event-driven Lambda |
| `commonfiles/scheduledtaskv2.yaml` | Reusable ECS Fargate scheduled task template (used for heavier workloads) |

---

## External Integrations

### Airline Providers (in `website-price-eye`)
The legacy Struts web application contains provider-specific request handling for 20+ carriers:
American Airlines (AA/AAP/AAPTS), United (UA), Delta (DL), Southwest (WN), Hawaiian (HA), Alaska (AS), Spirit (SP), Vueling (VY), TAP Portugal (TP), LATAM (LA), Tiger Air (TS), Amplitude (AMP), and others.

### AWS Services Used

| Service | Usage |
|---------|-------|
| Lambda | All compute (API handlers, scheduled jobs, authorizer) |
| API Gateway | Three gateways exposing REST endpoints |
| MySQL/Aurora | Transactional data storage |
| Redshift | Analytics queries and health checks |
| S3 | Notification templates, help docs assets, CloudFormation artifacts |
| SQS (FIFO) | Notification request queue with DLQ |
| SES | Email delivery for notifications |
| SecretsManager | Database credentials and API keys |
| CloudWatch Logs | Lambda log groups (7-day retention) |
| CloudWatch Alarms | Timeout alarms per Lambda → SNS |
| EventBridge | Cron trigger (sales-poc-expiry) + event bus (notifications) |
| ECR | Container image registry for Lambda functions |
| WAFv2 | Web ACL on Backend API Gateway (prod) |
| VPC | All Lambdas run inside VPC with private subnets |
