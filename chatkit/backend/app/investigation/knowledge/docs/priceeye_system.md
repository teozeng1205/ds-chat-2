# PriceEye Data Science — System Architecture & Process Reference

> Source: DS_DOCUMENTATION.md (2026-02-27). Use this doc for understanding what each process does, what tables it produces, and where to look when investigating issues.

---

## Data Flow Overview

```
priceeye-v2 (Java Lambda/ECS)
    → Raw data: S3 Parquet, MySQL audits, Redshift external tables
          │
          ├─► priceeye-analytics (Spark)
          │     → DCO (Derived Common Output): normalized price observations
          │     → Market/Segment anomaly Spark jobs (intermediate)
          │
          ├─► ds-priceeye-analytics (Python/ECS)
          │     → Competitive position analysis
          │     → Market-level anomaly scores (market_level_anomalies_v4)
          │     → Segment-level anomaly scores
          │     → Revenue score, OAG score, PAX/MIDT, daily itins
          │     → Alerts (EventBridge events)
          │
          ├─► ds-priceeye-data-collection (Python/Glue)
          │     → Delta SWIA input (collection_optimizer.delta_swia_input_v1)
          │     → Ingest TTL (collection_optimizer.ingest_ttl_v1)
          │     → YQYR cache (yqyr_cache.yqyr_cache_v1)
          │     → Site metrics (capacity, retry, cache, import, TPS)
          │
          ├─► ds-internal-monitoring (Python/Glue)
          │     → Deduped audit tables (monitoring_db.deduped_*)
          │     → Master combined_audit (full request lifecycle)
          │     → Provider-centric view (provider_combined_audit)
          │     → Customer-centric view (customer_combined_audit_v2)
          │
          ├─► ds-priceeye-enrichment (Python/Glue)
          │     → Tax regression coefficients (tax_reg.tax_reg_output_v1)
          │
          └─► ds-customer-monitoring (Python/Glue)
                → Billing metrics (billing_db.customer_daily_requests_v1/v2/v3)
                → Customer-centric monitoring views
```

---

## Processes & Their Tables

### 1. priceeye-v2 (Core System)

**What it does:** Accepts customer flight search requests, polls 20+ providers (airlines, GDS, OTAs), collects pricing responses, caches in Redis, writes audit trail.

**Produced Tables:**

| Table | Where | Description |
|-------|-------|-------------|
| `prod.monitoring.combined_audit` | Redshift (core) | Master joined audit: full request lifecycle (request → response → retry → cache → enrichment → packager → delivery) with error classification |
| `prod.monitoring.provider_combined_audit` | Redshift (core) | Provider-centric aggregated view from combined_audit |
| `prod.common_output.common_output_format` | Redshift (analytics) | Normalized price observations per customer/date — the source of all analytics |
| `priceeye_audits_db.*` | Glue/S3 | Long-term raw audit tables (provider_request_audit, provider_response_audit, packager_audit, delivery_audit, etc.) |

**S3 Raw Data:**
- `s3-atp-3victors-3vprod-use1-pe-common-output/{customer}/{YYYY}/{MM}/{DD}/{HH}/` — Parquet, per customer hourly
- `s3-atp-3victors-{env}-use1-dataset-ingest/estream/search-with-itineraries/v1/{YYYY}/{MM}/{DD}/` — Raw SWIA Avro
- `s3-atp-3victors-{env}-use1-dataset-ingest/delta/search-with-itineraries/v1/{YYYY}/{MM}/{DD}/` — Delta SWIA Avro

**Key columns in `combined_audit`:**
`id`, `inputrequestid`, `customer`, `customercollectionid`, `customercollectionname`, `reference`, `sitecategory`, `customersitecode`, `customerpos`, `providercode`, `sitecode`, `pos`, `carriercodes`, `originairportcode`, `destinationairportcode`, `departdate`, `returndate`, `triptype`, `cabin`, `passengercount`, `filterreason`, `response_status`, `response_itinerarycount`, `response_lastupdated`, `issue_source`, `issue_reason`, `itins_after_filtering`, `retry_response_status`, `retry_response_timestamp`, `retry_site`, `packager_recordcount`, `packager_substituteused`, `packager_timestamp`, `delivery_status`, `delivery_type`, `delivery_lastupdated`, `customer_salesdate`, `scheduledate`, `scheduletime`, `observationtimestamp`, `sales_date`
**NOTE:** `combined_audit` uses singular `issue_source` / `issue_reason`. `provider_combined_audit` uses plural `issue_sources` / `issue_reasons`. Do NOT mix them up.

**Combined audit is built by LEFT JOINing all 9 deduped tables on `providerrequestauditid`, with `deduped_provider_request_audit_detail` as the base.**

**Key columns in `provider_combined_audit`:**
`providercode`, `sitecode`, `sales_date`, `pos`, `carriercodes`, `issue_sources`, `issue_reasons`, `filterreason`, `response_status`, `itinerarycount`, `origin`, `destination`, `ap` (advance purchase), `los` (length of stay)

---

### 2. priceeye-analytics (DCO + Spark Anomalies)

**What it does:** Reads raw common output from S3, normalizes into DCO schema, detects anomalies via Spark jobs. Triggers per-customer per-hour.

**DCO (Derived Common Output):**
- S3: `s3-atp-3victors{env}-use1-derived-common-output/v1/{customer}/{YYYY}/{MM}/{DD}/{HH}/`
- Redshift: `analytics.derived_common_output` (also written to)
- Key fields: `customer_observation_date`, `origin`, `destination`, `pos`, `carrier`, `cabin`, `price_exc`, `price_inc`, `tax`, `sales_date`, `customer`

**Anomaly Spark Output:**
- S3: `s3-atp-3victors{env}-use1-anomaly-datasets/market-level/v4/{customer}/{YYYY}/{MM}/{DD}/`
- S3: `s3-atp-3victors{env}-use1-anomaly-datasets/segment-level/v4/{customer}/{YYYY}/{MM}/{DD}/`
- S3: `s3-atp-3victors{env}-use1-competitive-position/v2/{customer}/{YYYY}/{MM}/{DD}/`

---

### 3. ds-priceeye-analytics (Python Anomaly Scoring & Alerts)

**What it does:** Reads DCO and Spark outputs, applies statistical models, scores anomalies, publishes to Redshift and Alerts.

**Processes & Outputs:**

| Process | Trigger | Input | Output Table |
|---------|---------|-------|-------------|
| Competitive Position | Daily | DCO S3 Parquet (2 days) | S3 competitive-position/v2/... |
| Market-Level Analysis | Daily | Competitive position S3 | `analytics.market_level_analysis_v2` (Redshift) |
| Segment-Level Analysis | Daily | Competitive position S3 | `analytics.segment_level_analysis_v2` (Redshift) |
| Market-Level Generator | Step Functions (after ML Analysis) | `market_level_analysis_v2` (22-day rolling) | `analytics.market_level_anomalies_v4` (Redshift) |
| Segment-Level Generator | Step Functions (after ML Generator) | `segment_level_analysis_v2` + `market_level_anomalies_v4` | `analytics.segment_level_anomalies_v2` (Redshift) |
| Daily Itins (daily-itins) | Daily | `data_lakes.daily_representative_itinerary_v4` | `analytics.daily_itins_prices_v2` (Redshift) |
| OAG Score | Daily | OAG flight data + DCO | `analytics.oag_score_v2` (Redshift) |
| Revenue Score | Daily | `pax_midt` + `daily_itins_prices_v2` | `analytics.revenue_score_v1` (Redshift) |
| PAX MIDT | Daily | `midt_external.midt_daily_booking_summary` | `analytics.pax_midt` (Redshift) |
| Alerts Lambda | Hourly per customer | `analytics.segment_level_anomalies_v3` | EventBridge events |

**Key columns in `market_level_anomalies_v4`:**
`observation_date`, `customer`, `region_name`, `depart_period`, `carrier_group`, `cabin_group`, `seg_mkt`, `segment_name`, `competitive_position`, `freq_pcnt`, `mag_nominal`, `mag_pcnt`, `avg_freq_pcnt_7d`, `avg_mag_nom_7d`, `impact_score`, `estimated_revenue`, `carrier_contribution`, `sales_date`

**Key columns in `market_level_analysis_v2`:**
`sales_date`, `customer`, `mkt`, `seg`, `carrier`, `cabin`, `competitive_position`, `comparison_type`, `customer_brand`, `competitor_brand`, `top_offenders`, `cp_score`, `freq_pcnt`, `mag_nominal`

**Key columns in `segment_level_anomalies_v2`:**
`observation_date`, `customer`, `segment`, `cp` (competitive_position), `impacted_markets`, `impacted_mkt_pcnt`, `freq_pcnt_val`, `mag_pcnt_val`, `impact_score`, `top_offenders`, `estimated_revenue`, `avg_freq_pcnt_7d`, `avg_mag_pcnt_7d`, `sales_date`

---

### 4. ds-priceeye-data-collection (Collection Optimization, YQYR, Site Metrics)

**What it does:** Optimizes collection scheduling, computes YQYR tax predictions, measures site performance.

**Processes & Outputs:**

| Process | Description | Output Table |
|---------|-------------|-------------|
| Delta SWIA Input Unload | Reads raw SWIA Avro, extracts min prices | `collection_optimizer.delta_swia_input_v1` |
| Ingest TTL | Computes 25th-percentile hours-between-price-changes per carrier/POS/OD | `collection_optimizer.ingest_ttl_v1` |
| YQYR Cache Unload | Multi-level YQ/YR tax bucket cache from SWIA data | `yqyr_cache.yqyr_cache_v1` |
| YQYR Cache Inference | Predicts YQ/YR taxes for common_output records | `yqyr_cache.yqyr_predictions` |
| Provider TPS Unload | TPS validation metrics per provider | `site_metrics.provider_tps_validate_v1` |
| Provider TPS by Intervals | TPS in time-interval buckets | `site_metrics.provider_tps_by_intervals_v1` |
| Capacity Metrics | 14-day capacity with IQR outlier filtering | `site_metrics.capacity_final` |
| Cache Metrics | Cache hit/miss rates (4-week window) | `site_metrics.cache_metrics_v1` |
| Retry Metrics | Retry rate percentages (4-week window) | `site_metrics.retry_metrics_v1` |
| Import Metrics | Customer import counts by provider/site/collection/hour | `site_metrics.import_metrics_v1` |
| AS Dashboard Generator | AutoSchedule plan vs. actual comparison CSVs | S3 as-scheduled-comparison/ |
| Sales POC — Market Generator | Top-route market data from Redshift + MySQL | S3 ds-sales-poc/market_data/ |
| Sales POC — Input Generator | Flight search inputs for sales POC | `sales_poc.input_request` (MySQL) |

**Key columns in `delta_swia_input_v1`:**
`sales_date`, `customer`, `origin`, `destination`, `cabin`, `airline`, `brand`, `ap` (advance purchase), `min_price`, `pos`

**Key columns in `ingest_ttl_v1`:**
`sales_date`, `airline`, `carrier`, `pos`, `origin`, `destination`, `travel_period`, `cabin`, `ttl_hours` (25th-pct hours between price changes)

**Key columns in `capacity_final`:**
`sales_date`, `providercode`, `capacity_tph` (transactions per hour, IQR-filtered), `floor_applied` (whether provider floor patch was applied)

**Key columns in `cache_metrics_v1`:**
`sales_date`, `providercode`, `sitecode`, `cache_hit_rate`, `cache_miss_rate`, `total_requests`

**Key columns in `retry_metrics_v1`:**
`sales_date`, `providercode`, `retry_rate_pct`, `total_requests`, `retried_requests`

---

### 5. ds-internal-monitoring (Deduped Audit Pipeline)

**What it does:** Deduplicates raw priceeye-v2 audit data (24-hour rolling window), joins into a master combined_audit table, produces provider/customer-centric views.

**Step Functions order:**
1. Stage 1 (parallel): 5 dedup jobs
2. Stage 2 (parallel): 5 more dedup jobs
3. Stage 3: combined_audit (today + yesterday)
4. Stage 4: ValidateTablesAndRefreshViews
5. Stage 5 (parallel): provider_centric, customer_centric, response_dupes

**Key output tables:**

| Table | Description |
|-------|-------------|
| `monitoring_db.combined_audit` | Full lifecycle: every request with request+response+retry+cache+enrichment+packager+delivery merged + `issue_source`/`issue_reason` from error_mapping |
| `monitoring_db.provider_combined_audit` | Provider-centric view with airport/city metadata, AP and LOS fields |
| `monitoring_db.customer_combined_audit_v2` | Per-customer daily totals: total_reqs, successful, site_issues, no_response, substitute_used, packaged, delivered |
| `monitoring_db.response_dupes` | Requests with duplicate responses (response_count > 1) |
| `monitoring_db.deduped_*` | 9 individual deduped audit tables for granular analysis |

**Available at Redshift as:**
- `prod.monitoring.combined_audit` — the combined_audit table (surfaced via Redshift Spectrum)
- `prod.monitoring.provider_combined_audit` — the provider_combined_audit table

---

### 6. ds-priceeye-enrichment (Tax Regression)

**What it does:** Every Tuesday computes linear regression coefficients (slope m, intercept b) mapping total price → price excluding tax, for each market/carrier/cabin/POS combination.

**Output tables:**
- `tax_reg.tax_reg_output_v1` — Main coefficients (m, b, r2, correlation) per market
- `tax_reg.tax_reg_output_com_v1` — Coefficients for MCLA .COM carriers (Volaris Y4, VivaAerobus VB)
- `taxregression.tax_regression_v1` (MySQL) — Current merged coefficients (overwritten weekly)

**Key columns:** `pos`, `od`, `is_one_way`, `search_class`, `carrier`, `currency`, `nbr_outbound_stop`, `m` (slope), `b` (intercept), `r2`, `correlation`

---

### 7. ds-customer-monitoring (Billing & Customer Monitoring)

**What it does:** Produces billing metrics and customer-facing monitoring views. Primary source for billing data.

**Billing tables:**

| Table | Description |
|-------|-------------|
| `billing_db.customer_daily_requests_v1` | Daily billing per customer |
| `billing_db.customer_daily_requests_v2` | V1 + site code enrichment |
| `billing_db.customer_daily_requests_v3` | Most granular: broken down by providercode, customersitecode, customercollectionname, reference |

**Exact metric definitions (from billing SQL):**
- `GDS_scheduled` = sitecode = `'1G'`
- `OTA_scheduled` = sitecode IN `('EXP','DES','BKG','OBZ','PLN','TCY','EDR')`
- `MSE_scheduled` = sitecode IN `('SKYS','GGL','KYK')`
- `polled` = filterreason = `''` (sent to provider)
- `cached` = filterreason = `'Cache'` (served from cache)
- `filtered` = filterreason NOT IN `('', 'Cache')` (blocked: OAG, cabin, blacklist, etc.)
- `success` = response_status LIKE `'success%'` OR filterreason = `'Cache'`
- `site_failed` = failed AND issue_source = `'site'`
- `bad_requests` = failed AND issue_source = `'request'`
- `true_site_issues` = response_status=`'failed'` AND issue_source=`'site'` AND filterreason≠`'Cache'` AND (retry_response_status=`'failed'` OR IS NULL)
- `billable_requests` = `requested_by_customers` − `true_site_issues`

**Monitoring views (Redshift materialized views):**
- `monitoring_metadata_prod.customer_site_rollup_granularity` — Per-site metrics with airport+cabin granularity
- `monitoring_metadata_prod.customer_site_otp_rollup_granularity` — OTP request/response metrics

---

## Common Investigation Scenarios

### "Why did a request fail?"
→ Query `prod.monitoring.combined_audit` with `sales_date` + `providercode`/`sitecode`
→ Look at `issue_source` (request/site), `issue_reason`, `filterreason`, `delivery_status`

### "What is the collection health for a provider/site?"
→ Query `prod.monitoring.provider_combined_audit` — `issue_sources`, `issue_reasons`, `filterreason`, compute issue_rate_pct
→ Cross-reference `site_metrics.cache_metrics_v1` and `retry_metrics_v1` for deeper analysis

### "What are the pricing anomalies for customer X?"
→ Market level: `prod.analytics.market_level_anomalies_v3` (sales_date + customer)
→ Segment level: `prod.analytics.segment_level_anomalies_v3` (sales_date + customer)
→ Competitive position: `prod.analytics.competitive_position` (sales_date + customer)

### "Show me the price outlook / common output for customer X"
→ `prod.common_output.common_output_format`  partitioned by customer and sales date
→ Or `prod.analytics.derived_common_output`


### "What is the billing for customer X?"
→ `billing_db.customer_daily_requests_v3` for most detail (sales_date required)

### "What is the capacity of provider X?"
→ `site_metrics.capacity_final` — `capacity_tph` (IQR-filtered throughput per hour)

### "What is the retry/cache performance of provider X?"
→ `site_metrics.retry_metrics_v1` — retry_rate_pct
→ `site_metrics.cache_metrics_v1` — cache_hit_rate, cache_miss_rate

### "What sites / markets is customer X configured to collect from?"
→ `federated_priceeye.site_hierarchy` — customer site configuration (prod MySQL via Redshift federation)
→ Filter by `customer = 'X'`, returns sitecode, providercode, market, etc.

### "What is the active collection schedule for customer X?"
→ `federated_scheduling.as_hourly_collection_plans` (redshift_core only)
→ Filter by `customer = 'X'`; shows hourly AutoSchedule plans per market

### "What does error code Y mean?"
→ `federated_priceeye.error_mapping` — maps error codes to human-readable descriptions

### "What currency conversion rate was used on date D?"
→ `federated_metadata.currencyexchangerates` — daily exchange rates (used by daily-process)
→ Filter by `rate_date` and `target_currency = 'USD'`

### "What are the anomaly detection configuration / weights for customer X?"
→ `federated_analytics.anomalies_impact_score_weights` — impact score weights
→ `federated_analytics.anomalies_direction_score` — direction scoring weights
→ `federated_analytics.cabin_group`, `carrier_group`, `segment` — grouping definitions
