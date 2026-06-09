# PriceEye — System Overview (end to end)

> **Tier A.** Single canonical "how PriceEye works" document, derived from the
> source repos under `~/git/`. For the exact per-stage tables, S3 prefixes, and
> triggers, see the workflow docs in `docs/workflows/`. If neither answers a
> question, read the repo source directly (Tier C).
>
> Environment note: bucket/Glue names carry an `${environment}` token that resolves
> to `-3vprod` in production (per `3v-build-deploy/databases/sql/prod-*-redshift.sql`).
> Examples below show the prod form `…-3vprod-…`.

## 1. What PriceEye is

PriceEye is ATPCO's real-time airline **price intelligence** platform. It crawls fares
from 20+ providers, audits every collection, normalizes the raw prices into a common
representation, and runs analytics that compute **competitive position** and price
**anomalies**, plus **monitoring**, **billing**, and **site-metrics** rollups. Outputs
land in S3 (Glue-cataloged parquet, queryable from Redshift Spectrum / Athena), Redshift,
and MySQL/Aurora, and drive alerts, dashboards, and the customer app.

## 2. Repos and their roles (`~/git/`)

| Repo | Role |
|---|---|
| `priceeye-v2` | Collection/crawl + audit persistence (`persist-audit-data-redshift`) → `prod.priceeye_audits.*`. Also owns the Glue `monitoring_db` table defs (`source/priceeye-deploy/yaml/glue-monitoring.yaml`). |
| `ds-internal-monitoring` | Dedup + join audit pipeline → `prod.monitoring.combined_audit`, `prod.monitoring.provider_combined_audit` (hourly). |
| `ds-customer-monitoring` | Customer-centric audit + **billing** → `customer_combined_audit_v2`, `prod.billing.customer_daily_requests_v1/v2/v3`. |
| `ds-priceeye-analytics` | DCO → competitive position → analysis → anomalies → alerts (the anomalies pipeline). |
| `ds-priceeye-data-collection` | Site/collection metrics → `prod.site_metrics.*`. |
| `ds-priceeye-enrichment` | Tax regression coefficients → `prod.tax_reg.*` (weekly). |
| `3v-build-deploy` | Redshift external-schema DDL (maps `prod.*` schemas to Glue DBs / federated MySQL). |

## 3. End-to-end data flow

```
providers ─▶ priceeye-v2 crawl
   │  raw prices ─▶ S3 s3-…-3vprod-…-pe-common-output/<customer>/YYYY/MM/DD/
   │               Redshift: prod.common_output.common_output_format
   │  audits ─persist-audit-data-redshift─▶ prod.priceeye_audits.*  (Core Redshift)
   │
   ├─▶ ds-internal-monitoring (hourly cron :10)
   │      deduped_* (×9) ─▶ combined_audit ─▶ provider_combined_audit
   │      (prod.monitoring.*)
   │         ├─▶ ds-customer-monitoring: customer_combined_audit_v2,
   │         │     prod.billing.customer_daily_requests_v1/v2/v3
   │         └─▶ ds-priceeye-data-collection: prod.site_metrics.* (capacity/cache/retry/import)
   │
   └─▶ ds-priceeye-analytics (anomalies pipeline)
          derived-common-output (DCO v2) ─▶ competitive_position v2
          ─▶ market/segment_level_analysis v2 ─▶ market/segment_level_anomalies v4
          ─▶ alerts Lambda ─▶ EventBridge "Price Anomaly" business event
```

## 4. Connectors, clusters, and external schemas

Tables are reached via three connectors, auto-routed by prefix:

- **redshift_analytics** — `prod.analytics.*`, `prod.common_output.*`, `prod.flight_summary.*`,
  `prod.priceeye_output.*`, `prod.tax_reg.*`, `prod.billing.*`, most `federated_*`.
- **redshift_core** — `prod.monitoring.*`, `prod.site_metrics.*`, `prod.scheduling.*`,
  `prod.priceeye_audits.*`, `billing`/`federated_scheduling.*`. `local.*` are DEV copies — never default to them.
- **mysql_priceeye** — `priceeye.*`, `sales_poc.*`, `taxregression.*`.

Redshift external schemas (declared in `3v-build-deploy/databases/sql/*-redshift.sql`) map a
Redshift schema to either a **Glue** database (S3-backed parquet via Spectrum) or a **federated
MySQL** database. Glue-backed: `priceeye_audits`, `monitoring`, `common_output`, `billing`,
`site_metrics`, `priceeye_output`, `flight_summary`, `midt_external`, `scheduling`. Federated MySQL:
`federated_priceeye` → MySQL `priceeye`, `federated_metadata` → Aurora `metadata`,
`federated_analytics` → `priceeye` reader. So one logical dataset can appear as a MySQL table,
a Redshift federated alias, and an S3/Glue external table.

## 5. S3 bucket convention

`s3-atp-3victors-<env>-use1-<purpose>`. Key purposes:
`pe-common-output` (raw), `derived-common-output` (DCO), `competitive-position`,
`anomaly-datasets` (analysis + anomalies + scores), `deduped-datasets` (monitoring dedup +
combined_audit), `provider-monitor`, `customer-monitor`, `billing`, `sitemetrics`,
`pe-analytics-audits` (alert audit), `ds-standard-brands` (brand equivalence).

## 6. Orchestration

- **Step Functions** sequence multi-stage work (e.g. `DS-Analytics-EventDriven-Jobs` runs
  market-level-generator → segment-level-generator; `unload-monitoring-step-function` runs the
  dedup → combined_audit → provider/customer rollups).
- **EventBridge `data-pipeline` bus** carries `Task Completed` and `Price Anomaly` business
  events; S3 `_SUCCESS` object-created rules also start state machines (e.g. segment-analysis
  `_SUCCESS` fires the anomaly generators).
- **Schedules**: internal monitoring hourly `cron(10 * * * ? *)`; billing daily `cron(45 10 …)`;
  site-metrics and scores in the 02:00–05:00 UTC window.

## 7. Key tables index (with producing pipeline)

| Table | Cluster | Produced by |
|---|---|---|
| `prod.common_output.common_output_format` | analytics | priceeye-v2 (raw), DCO input |
| `prod.analytics.derived_common_output` (`derived_common_output_v2`) | analytics | ds-priceeye-analytics dco-v2-spark |
| `prod.analytics.competitive_position` (`competitive_position_v2`) | analytics | competitive-position |
| `prod.analytics.market_level_analysis_v2` / `segment_level_analysis_v2` | analytics | market/segment-level-analysis |
| `prod.analytics.market_level_anomalies_v4` / `segment_level_anomalies_v4` | analytics | market/segment-level-generator |
| `prod.monitoring.combined_audit` | core | ds-internal-monitoring |
| `prod.monitoring.provider_combined_audit` | core | ds-internal-monitoring |
| `prod.billing.customer_daily_requests_v1/v2/v3` | core (Glue `billing_db`) | ds-customer-monitoring |
| `prod.site_metrics.capacity_final / cache_metrics_v1 / retry_metrics_v1 / import_metrics_v1` | core | ds-priceeye-data-collection |
| `prod.priceeye_audits.*` | core | priceeye-v2 |

Source: derived from `ds-priceeye-analytics`, `ds-internal-monitoring`, `ds-customer-monitoring`,
`ds-priceeye-data-collection`, `priceeye-v2`, and `3v-build-deploy` (Redshift DDL).
