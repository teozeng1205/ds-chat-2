# DS-* Reader Properties Reference

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

| Table | Notes |
|---|---|
| `analytics.pax_midt` | |
| `analytics.daily_itins_prices_v2` | |
| `analytics.market_level_analysis_v2` | |
| `analytics.market_level_anomalies_v3` | |
| `analytics.market_level_anomalies_v4` | |
| `analytics.oag_score_v2` | |
| `analytics.revenue_score_v1` | |
| `analytics.segment_level_analysis_v2` | |
| `analytics.anomalies_impact_score_weights` | |
| `metadata.carrier` | Airline metadata (pdf-generation) |
| `{schema}.channel_availability` | Dynamic; schema from config (pdf-generation) |
| `prod.common_output.common_output_format` | Via `daily_process.input_table` config (daily-process) |
| `federated_metadata.currencyexchangerates` | Via `daily_process.currency_exchange_table` config (daily-process) |

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

| Table | Notes |
|---|---|
| `priceeye.customer_defaults` | Customer analytics flags |
| `priceeye.site_hierarchy` | Site/carrier hierarchy |
| `priceeye.transaction_rates` | |
| `analytics.alerts_schedule` | |
| `analytics.anomalies_direction_score` | |
| `analytics.anomalies_impact_score_weights` | |
| `analytics.cabin_group` | |
| `analytics.carrier_group` | |
| `analytics.date_range` | |
| `analytics.demo_carrier_substitutions` | |
| `analytics.geography_entry` | |
| `analytics.region` | |
| `analytics.segment` | |

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

| Table | Notes |
|---|---|
| `collection_optimizer.delta_swia_input_v1` | SWIA ingest data (ingest-ttl) |
| `local.site_metrics.provider_tps_by_intervals_v1` | TPS intervals (capacity-metrics-generator) |
| `prod.monitoring.provider_combined_audit` | Via `retry_metrics_input_table` / `cache_metrics_input_table` config |
| `prod.monitoring.combined_audit` | Via `import_metrics_input_table` config |
| `{smm_tableN}` | Dynamic tables monitored by site-metrics-monitor (from `smm-config.properties`) |

---

### 4. `database-core-local-redshift-serverless-reader.properties`

**Type:** Redshift (Core local/federated cluster)

#### Used in repos / modules

| Repo | Module |
|---|---|
| `ds-priceeye-data-collection` | `as-dashboard-generator` (core_reader) |

#### Tables queried

| Table | Notes |
|---|---|
| `local.federated_priceeye.site_hierarchy` | |
| `local.federated_scheduling.as_hourly_collection_plans` | AS hourly collection plans |
| `local.monitoring.combined_audit` | |
| `local.scheduling.auto_schedule_output` | Auto-schedule output |
| `local.site_metrics.capacity_final` | |
