# PriceEye — End-to-End System Overview

> **Tier A knowledge.** This is the single canonical "how PriceEye works end to end"
> document. For a specific process (its trigger, ordered steps, and the exact tables
> it reads/writes), see the per-process files in `docs/workflows/`. If neither this
> file nor a workflow file answers the question, explore the source repos directly
> under `~/git/` (Tier C).

## 1. What PriceEye is

PriceEye is ATPCO's real-time airline **price intelligence** platform. It continuously
collects fares from 20+ providers (airlines, GDSs, OTAs), normalizes them into a
**Common Output** representation, and feeds analytics pipelines that compute price
**anomalies**, **competitive position**, **billing/usage** metrics, and **monitoring**
audits. Outputs land in S3 (Glue-cataloged parquet), Redshift, and MySQL/Aurora, and
are consumed by alerts, dashboards (QuickSight), and the customer-facing app.

## 2. The end-to-end data flow

```
providers ─▶ priceeye-v2 (collection)
                │  raw search prices → S3 common-output/<customer>/YYYY/MM/DD/HH/
                ▼
        ┌───────────────────────────── fan-out ─────────────────────────────┐
        ▼                         ▼                      ▼                    ▼
 ds-priceeye-analytics    ds-internal-monitoring  ds-customer-monitoring  ds-priceeye-
 (DCO + anomalies)        (dedup + join audit)    (billing/usage)         data-collection
        │                         │                      │              (site/collection
        │                         │                      │               metrics)
        ▼                         ▼                      ▼                    ▼
 derived-common-output    prod.monitoring.            prod.billing.        prod.site_metrics.*
 → market/segment         combined_audit,             customer_daily_      prod.collection_
   anomaly tables         provider_combined_audit     requests_v1/v2/v3    optimizer.*
        │
        ▼
 EventBridge "data-pipeline" bus (Task Completed / Price Anomaly)
        │
        ▼
 Alert Lambda → SegmentLevel "Price Anomaly" events → S3 audit + customer alerts
```

ds-priceeye-enrichment runs separately (weekly, Tuesdays) producing `prod.tax_reg.*`
regression coefficients.

## 3. Stages, repos, and what each produces

| Stage | Repo (`~/git/`) | Produces |
|---|---|---|
| Collection | `priceeye-v2` | raw common output in S3 `common-output/…`; feeds everything downstream |
| DCO + anomalies | `ds-priceeye-analytics` | `derived-common-output` parquet; `prod.analytics.market_level_anomalies*`, `segment_level_anomalies*`, `competitive_position`, `pax_midt`, `revenue_score_v1`, `oag_score_v2` |
| Monitoring (dedup/join) | `ds-internal-monitoring` | `prod.monitoring.combined_audit`, `prod.monitoring.provider_combined_audit` (hourly) |
| Billing / usage | `ds-customer-monitoring` | `prod.billing.customer_daily_requests_v1/v2/v3` (primary billing source) |
| Site / collection metrics | `ds-priceeye-data-collection` | `prod.site_metrics.capacity_final / cache_metrics_v1 / retry_metrics_v1 / import_metrics_v1`, `collection_optimizer.*`, `yqyr_cache.*` |
| Enrichment | `ds-priceeye-enrichment` | `prod.tax_reg.tax_reg_output_v1` (weekly, Tuesdays) |
| Infra / DDL | `3v-build-deploy`, `priceeye-v2/docs/redshift` | Redshift external-schema definitions, partition config |

## 4. Connectors and where tables live

Tables are reached through three connectors (auto-routed by table prefix):

- **redshift_analytics** (Analytics serverless): `prod.analytics.*`, `prod.common_output.*`,
  `prod.flight_summary.*`, `prod.midt_external.*`, `prod.priceeye_output.*`, `prod.tax_reg.*`,
  `prod.billing.*`, and most `federated_*` schemas.
- **redshift_core** (Core serverless): `prod.monitoring.*`, `prod.site_metrics.*`,
  `prod.scheduling.*`, `billing_db.*`, `federated_scheduling.*`. `local.*` are DEV copies — never default to them.
- **mysql_priceeye** (MySQL/Aurora): `priceeye.*`, `sales_poc.*`, `taxregression.*`.

**External / federated tables (important for debugging):** Redshift exposes other stores
via external schemas, defined declaratively in `3v-build-deploy/databases/sql/*-redshift.sql`
and `priceeye-v2/docs/redshift/*.sql`:

- **Glue-backed (Spectrum):** `monitoring`, `priceeye_audits`, `priceeye_output`,
  `flight_summary`, `midt_external`, `site_metrics`, `scheduling`, `common_output`, `billing`
  → backed by Glue databases (`glue-atp-3victors-<env>-use1-<name>_db[_link]`), themselves
  backed by S3 parquet.
- **MySQL-federated (`CREATE EXTERNAL SCHEMA … FROM MYSQL`):** `federated_priceeye` → MySQL
  `priceeye`; `federated_metadata` → Aurora `metadata`; `federated_analytics` → `priceeye`
  reader. These are **live production data**, not dev copies — use them to JOIN MySQL config
  with Redshift facts without leaving Redshift.

So the same logical data can appear as a MySQL table (`priceeye.X`), a Redshift federated
alias (`federated_priceeye.X`), and an S3/Glue external table — same data, different connector.

## 5. Orchestration and ordering

Producer ordering is real and is encoded, not implicit:

- **Step Functions** run the analytics jobs in order — e.g. for every SegmentLevel Spark
  completion, both market-level and segment-level Python jobs run *after* it, so downstream
  parquet stays synchronized with the upstream Redshift facts.
- **`partition_details` (MySQL table)** drives Glue partition registration: columns
  `bucket`, `pattern`, `partition_order`, `destination_database`, `destination_table`,
  `emit_event`. `partition_order` encodes the partition-column ordering (e.g. `sales_date,customer`);
  `emit_event` controls whether a completion event fires on the EventBridge `data-pipeline` bus.
- **EventBridge** `data-pipeline` bus carries `Task Completed` and `Price Anomaly` events that
  trigger downstream consumers (e.g. the alert Lambda).

## 6. Debugging signals (empty / stale tables)

- A table with **no producer** is usually intentional/abandoned, not a bug: e.g.
  `travelport_carriers` and `valid_market_carriers` are empty in all envs because the EMR
  jobs that populated them were removed years ago. Downstream readers silently get empty results.
- A table **empty in DEV but not GOLD/PROD** is often a deliberate environment clean: e.g.
  `sales_poc.input_request` was cleared in DEV while other `sales_poc.*` tables kept data.
- A producer that fails **without emitting its event** (`emit_event=0`, or a silent failure)
  leaves downstream tables looking stale with no alert.
- A query that returns nothing may simply be missing the latest partition — check the newest
  available `sales_date` for that customer before concluding "no data."
- A cross-account external schema can be unreachable (e.g. the `swav` schema is marked
  `-- Not working` in `prod-core-redshift.sql`).

To diagnose: resolve the entity → find its producing process/job and the upstream tables it
depends on (workflow doc or repo) → check the job's last run + the table's live freshness/row
count → check whether the expected event fired.

## 7. Consumption

- **Alerts:** `ds-priceeye-analytics/source/alerts` Lambda runs hourly per customer, reads the
  segment anomaly table, and emits consolidated `Price Anomaly` (SegmentLevel) events, saving
  payloads to S3 for audit.
- **BI:** Glue tables backed by the analytics parquet are queried by Athena and QuickSight.
- **App:** the priceeye-* application repos serve customers from the MySQL/analytics tables.

## 8. Data stores at a glance

| Layer | Technology | Examples |
|---|---|---|
| Raw capture | S3 | `common-output/<customer>/YYYY/MM/DD/HH/` |
| Processed search | S3 | `derived-common-output/<version>/<customer>/…` |
| Competitive metrics | S3 + Redshift | `competitive-position/…`, `prod.analytics.market_level_anomalies`, `segment_level_anomalies` |
| Reference / config | MySQL / Aurora | `priceeye.site`, `priceeye.site_hierarchy`, `priceeye.customer`, `partition_details` |
| Analytics outputs | S3 (Glue) | `market-level/`, `segment-level/v3/`, `daily-itins/`, `oag-score/` |
| Monitoring / billing | Redshift | `prod.monitoring.*`, `prod.billing.customer_daily_requests_v*` |
| Notifications | EventBridge | `data-pipeline` bus: `Task Completed`, `Price Anomaly` |

Source: assembled from the ds-* repo READMEs, `3v-build-deploy`/`priceeye-v2` Redshift DDL,
and the PriceEye pipeline data flow.
