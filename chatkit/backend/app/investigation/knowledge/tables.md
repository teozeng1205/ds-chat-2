# DS-* Reader Properties Reference

## Table Namespace Tiers

| Tier | Meaning | Examples |
|------|---------|---------|
| **prod** | Always production data — use `prod.*` prefix | `prod.monitoring.*`, `prod.common_output.*` |
| **local-only** | Dev/env data — no prod equivalent | `local.site_metrics.*`, `local.federated_*.*`, `local.scheduling.*` |
| **analytics-env** | Dev/analytics cluster only — may be empty in prod | `analytics.market_level_anomalies_v4`, `analytics.pax_midt` |
| **mysql** | PriceEye MySQL — environment-independent lookup tables | `priceeye.*`, most MySQL-side `analytics.*` |



## Reader Properties Files

### 1. `database-analytics-redshift-serverless-reader.properties`

**Type:** Redshift (Analytics serverless cluster)

#### Used in repos / modules

| Repo | Module |
|---|---|
| `ds-channel-comparison` | `daily-process` |
| `ds-channel-comparison` | `pdf-generation` |
| `ds-priceeye-analytics` | `brands-equivalence` |
| `ds-priceeye-analytics` | `anomalies/revenue-score` |
| `ds-priceeye-analytics` | `anomalies/pax-midt` (midt_reader) |
| `ds-priceeye-analytics` | `anomalies/market-level-generator` (analytics_reader) |
| `ds-priceeye-analytics` | `anomalies/segment-level-generator` (analytics_reader) |
| `ds-priceeye-analytics` | `alerts` (analytics_reader) |
| `ds-priceeye-data-collection` | `sales-poc-input-generator` (midt_reader + redshift_reader) |
| `ds-priceeye-data-collection` | `sales-poc-market-generator` (redshift_reader) |

#### Tables queried

| Table | Tier | Notes |
|---|---|---|
| `analytics.pax_midt` | **analytics-env** | |
| `analytics.daily_itins_prices_v2` | **analytics-env** | |
| `analytics.market_level_analysis_v2` | **analytics-env** | |
| `analytics.market_level_anomalies_v3` | **analytics-env** | |
| `analytics.market_level_anomalies_v4` | **analytics-env** | |
| `analytics.oag_score_v2` | **analytics-env** | |
| `analytics.revenue_score_v1` | **analytics-env** | |
| `analytics.segment_level_analysis_v2` | **analytics-env** | |
| `analytics.anomalies_impact_score_weights` | **analytics-env** | |
| `metadata.carrier` | **analytics-env** | Airline metadata (pdf-generation) |
| `{schema}.channel_availability` | **analytics-env** | Dynamic; schema from config (pdf-generation) |
| `prod.common_output.common_output_format` | **prod** | Via `daily_process.input_table` config (daily-process) |
| `federated_metadata.currencyexchangerates` | **analytics-env** | Via `daily_process.currency_exchange_table` config (daily-process) |

---

### 2. `database-priceeye-reader.properties`

**Type:** MySQL (PriceEye database)

#### Used in repos / modules

| Repo | Module |
|---|---|
| `ds-priceeye-analytics` | `anomalies/pax-midt` (priceeye_reader) |
| `ds-priceeye-analytics` | `anomalies/segment-level-analysis` (priceeye_reader) |
| `ds-priceeye-analytics` | `anomalies/market-level-analysis` (priceeye_reader) |
| `ds-priceeye-analytics` | `anomalies/market-level-generator` (priceeye_reader) |
| `ds-priceeye-analytics` | `anomalies/segment-level-generator` (priceeye_reader) |
| `ds-priceeye-analytics` | `anomalies/oag-score` (priceeye_reader) |
| `ds-priceeye-analytics` | `anomalies/daily-itins` (priceeye_reader) |
| `ds-priceeye-analytics` | `alerts` (mysql_reader) |
| `ds-priceeye-data-collection` | `sales-poc-input-generator` (mysql_reader) |
| `ds-priceeye-data-collection` | `sales-poc-market-generator` (mysql_reader) |
| `ds-priceeye-data-collection` | `site-metrics/capacity-metrics-generator` (mysql_reader) |
| `ds-priceeye-data-collection` | `site-metrics/import-metrics-generator` (mysql_reader) |

#### Tables queried

| Table | Tier | Notes |
|---|---|---|
| `priceeye.customer_defaults` | **mysql** | Customer analytics flags |
| `priceeye.site_hierarchy` | **mysql** | Site/carrier hierarchy |
| `priceeye.transaction_rates` | **mysql** | |
| `analytics.alerts_schedule` | **mysql** | |
| `analytics.anomalies_direction_score` | **mysql** | |
| `analytics.anomalies_impact_score_weights` | **mysql** | |
| `analytics.cabin_group` | **mysql** | |
| `analytics.carrier_group` | **mysql** | |
| `analytics.date_range` | **mysql** | |
| `analytics.demo_carrier_substitutions` | **mysql** | |
| `analytics.geography_entry` | **mysql** | |
| `analytics.region` | **mysql** | |
| `analytics.segment` | **mysql** | |

---

### 3. `database-core-redshift-serverless-reader.properties`

**Type:** Redshift (Core serverless cluster)

#### Used in repos / modules

| Repo | Module |
|---|---|
| `ds-priceeye-data-collection` | `ingest-ttl` (redshift_reader) |
| `ds-priceeye-data-collection` | `site-metrics/site-metrics-monitor` (core_reader) |
| `ds-priceeye-data-collection` | `site-metrics/capacity-metrics-generator` (redshift_reader) |
| `ds-priceeye-data-collection` | `site-metrics/retry-metrics-generator` (core_reader) |
| `ds-priceeye-data-collection` | `site-metrics/cache-metrics-generator` (core_reader) |
| `ds-priceeye-data-collection` | `site-metrics/import-metrics-generator` (core_reader) |

#### Tables queried

| Table | Tier | Notes |
|---|---|---|
| `collection_optimizer.delta_swia_input_v1` | **analytics-env** | SWIA ingest data (ingest-ttl) |
| `local.site_metrics.provider_tps_by_intervals_v1` | **local-only** | TPS intervals (capacity-metrics-generator) |
| `prod.monitoring.provider_combined_audit` | **prod** | Via `retry_metrics_input_table` / `cache_metrics_input_table` config |
| `prod.monitoring.combined_audit` | **prod** | Via `import_metrics_input_table` config |
| `{smm_tableN}` | — | Dynamic tables monitored by site-metrics-monitor (from `smm-config.properties`) |

---

### 4. `database-core-local-redshift-serverless-reader.properties`

**Type:** Redshift (Core local/federated cluster)

#### Used in repos / modules

| Repo | Module |
|---|---|
| `ds-priceeye-data-collection` | `as-dashboard-generator` (core_reader) |

#### Tables queried

| Table | Tier | Notes |
|---|---|---|
| `local.federated_priceeye.site_hierarchy` | **local-only** | |
| `local.federated_scheduling.as_hourly_collection_plans` | **local-only** | AS hourly collection plans |
| `local.monitoring.combined_audit` | **local-only** | Dev data only; prefer `prod.monitoring.combined_audit` for production queries |
| `local.scheduling.auto_schedule_output` | **local-only** | Auto-schedule output |
| `local.site_metrics.capacity_final` | **local-only** | |
