# Database Connections (PriceEye <-> Analytics)

## Core Identity Dimensions
- Provider dimension: `priceeye.provider.provider_code` -> monitoring `providercode` and site-level config tables.
- Site dimension: `priceeye.site.site_code` (+ `provider_code`) -> monitoring `sitecode` and scheduler site routing.
- Customer dimension: `priceeye.customer.name` -> monitoring/analytics `customer` or `customers` fields.

## Operational Scheduling Path (MySQL)
- `priceeye.customer_site_code` defines canonical customer-facing site codes.
- `priceeye.site_hierarchy` maps customer_site_code priority -> provider_code/site_code routes.
- `priceeye.site_metrics` and `priceeye.transaction_rates` provide capacity/rate controls.
- `priceeye.runtime_archive` and `priceeye.auto_schedule_trigger` capture scheduling/runtime behavior.

## Analytics Path (Redshift)
- Monitoring tables (`prod.monitoring.*`) capture request/provider-site audits.
- Analytics tables (`prod.analytics.market_level_anomalies*`, `segment_level_anomalies*`) capture anomaly facts.
- MySQL analytics config (`analytics.anomalies_direction_score`, `analytics.anomalies_impact_score_weights`) enriches impact scoring.
