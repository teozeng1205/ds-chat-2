# Workflow: Monitoring Pipelines (audits → combined → provider / customer / billing / site-metrics)

**Cluster:** Core Redshift (Glue-backed external schema `monitoring`, `billing`, `site_metrics`).
Glue `monitoring_db` table defs live in `priceeye-v2/source/priceeye-deploy/yaml/glue-monitoring.yaml`.
Bucket/Glue names show prod form (`-3vprod-`).

## Upstream source — `prod.priceeye_audits.*` (raw collection audits)

The raw audit trail written by the **priceeye-v2** crawl (`persist-audit-data-redshift` Lambda).
External schema `priceeye_audits` is defined over Glue `priceeye_audits_db_link` on **both** the core and
analytics clusters (`3v-build-deploy/databases/sql/prod-{core,analytics}-redshift.sql`); the agent's default
routing reads them via **redshift_analytics** as `prod.priceeye_audits.<table>` (`local.priceeye_audits.*` are
dev copies — don't default to them). Canonical names: `priceeye-v2/source/database-util/.../AuditTables.java`.

Full set (17 tables; column counts from the live snapshot), ordered by pipeline phase:

| Table | cols | Captures |
|---|---|---|
| `scheduler_audit` | 7 | collection-run scheduling |
| `collection_run_audit` | 10 | collection-run lifecycle |
| `collection_completion_audit` | 4 | collection-completion markers |
| `provider_request_audit` | 25 | per provider request (core request fact) |
| `provider_request_audit_detail` | 12 | per-request detail rows (joined into combined_audit) |
| `provider_response_audit` | 8 | provider responses |
| `retry_audit` | 7 | request retries |
| `cache_audit` | 4 | cache usage |
| `cache_loader_audit` | 4 | cache loading |
| `tpfc_cache_audit` | 5 | TPFC cache |
| `global_filter_audit` | 11 | global-filter decisions |
| `global_filter_audit_summary` | 5 | global-filter summary |
| `enrichment_audit` | 39 | enrichment step (widest audit table) |
| `packager_audit` | 14 | result packaging |
| `delivery_scheduler_audit` | 7 | delivery scheduling |
| `delivery_audit` | 10 | delivery |
| `delivery_combiner_audit` | 15 | delivery combiner |

`AuditTables.AUDIT_TABLE_NAMES` enumerates 15 of these (the persisted set); `collection_run_audit` and
`collection_completion_audit` also exist live. The dedup stage (Pipeline 1 below) consumes
`provider_request_audit[_detail]`, `provider_response_audit`, `retry_audit`, `cache_loader_audit`,
`global_filter_audit_summary`, `enrichment_audit`, `packager_audit`, and `delivery_combiner_audit`.

## Pipeline 1 — Internal monitoring (dedup + join)  ·  repo `ds-internal-monitoring`

| Stage | Table (`prod.monitoring.*` / Glue `monitoring_db`) | Partition | S3 (bucket `s3-…-3vprod-…-deduped-datasets` unless noted) |
|---|---|---|---|
| 1a dedup (×9) | `deduped_provider_request_audit`, `…_detail`, `…_response_audit`, `…_retry_audit`, `…_cache_loader_audit`, `…_global_filter_audit_summary`, `…_enrichment_audit`, `…_packager_audit`, `…_delivery_audit` | sales_date | `/v1/<name>/` |
| 1b combined | `combined_audit` (`prod.monitoring.combined_audit`) | sales_date | `/v1/combined_audit/YYYY/MM/DD/` |
| 1c provider rollup | `provider_combined_audit` (`prod.monitoring.provider_combined_audit`) | sales_date | bucket `s3-…-provider-monitor` `/v1/provider-combined-audit/YYYY/MM/DD/` |
| 1d refined | `refined_collection_run_audit` (`prod.monitoring.refined_collection_run_audit`) | — (verify live) | `/v1/refined_collection_run_audit/` |

- **Dedup logic** (`source/combined-audits/deduped-audits/src/unload-deduped-*.py`): `SELECT <business cols>,
  COUNT(*) AS occurrences … GROUP BY <business cols>` over a 1–3 day `sales_date` window, hour-bounded by
  `actualscheduletimestamp` (so the hourly run only re-processes the current hour). The detail job joins
  detail→request on `providerrequestauditid=id`.
- **combined_audit** (`unload-combined-audit.py`): driven from provider-request-audit-detail; LEFT JOINs request,
  response, error-map (`local.federated_priceeye.error_mapping`), retry, global-filter, enrichment, cache (on
  `providerrequestauditid`), packager/delivery (on customer + collection). Derives **singular** `issue_source` /
  `issue_reason`; response collapsed to success/failed.
- **provider_combined_audit** (`provider-centric-dataset-unload.py`): reads `prod.monitoring.combined_audit`,
  GROUP BY provider-request `id`, `LISTAGG`s customer/collection/input-request dims. Computes **plural**
  `issue_sources` / `issue_reasons` and **`inputrequestid_count = regexp_count(input_request_ids,'|')+1`**
  (use `SUM(inputrequestid_count)`, never `COUNT(DISTINCT inputrequestid)`). Joins
  `local.federated_metadata.airportlocation_extra`/`citylocation_extra` for origin/destination geo.
- **`refined_collection_run_audit`** (`prod.monitoring.refined_collection_run_audit`; S3
  `…-deduped-datasets/v1/refined_collection_run_audit`): a refined collection-run audit emitted alongside the
  deduped tables. It is the **completed-collection signal the DCO trigger reads** to launch the analytics
  pipeline (see `anomalies-pipeline.md` → `dco-v2-spark-trigger`). The KB snapshot did not capture its
  columns/partition — run `inspect_table` for the live schema.

**Orchestration:** Step Function `unload-monitoring-step-function.asl.json` — Parallel dedup batches →
combined_audit (today + yesterday) → verify/refresh views → provider + customer rollups. **Trigger:**
EventBridge **`cron(10 * * * ? *)` — hourly at :10**. A standalone provider-centric SFN also runs `cron(0 2 …)` daily.

## Pipeline 2 — Customer monitoring & billing  ·  repo `ds-customer-monitoring`

- **`customer_combined_audit_v2`** (Glue `monitoring_db`) ← `prod.monitoring.combined_audit`; partition `sales_date`;
  S3 `s3-…-customer-monitor` `/v2/customer-combined-audit/YYYY/MM/DD/`. Applies customer-specific `sales_date`
  corrections (AA_UK/AA_B3/AA_B4/Advito); filters `sitecategory like '%main%'/'%substitute%'`. Schedules:
  `cron(30 17-23 …)` (intraday, DAYSOFFSET=-1) + `cron(0 2 …)` (daily, DAYSOFFSET=1).
- **Billing — `prod.billing.customer_daily_requests_v1/v2/v3`** (Glue `billing_db`, Redshift schema **`billing`**,
  *not* `billing_db.*`), all in bucket `s3-…-billing`, partition `sales_date`, **`cron(45 10 * * ? *)` daily 10:45 UTC**.
  All read `prod.monitoring.combined_audit`:
  - **v1** prefix `v1/customer_daily_requests` — per-customer daily; `billable_requests = unq_scheduled − true_site_issues`.
  - **v2** prefix `v2/customer_daily_requests` — adds `providercode`, `customer_site_code`, `customersitetype`
    (join `local.federated_priceeye.customer_site_code`).
  - **v3** prefix `v3/customer_daily_requests` — adds `customercollectionname`, `reference` to the grain.

## Pipeline 3 — Site / collection metrics  ·  repo `ds-priceeye-data-collection`

Consumes `prod.monitoring.provider_combined_audit` / `combined_audit`. Glue DB `site-metrics-db`, bucket
`s3-…-sitemetrics`, partition `sales_date`. Config `docs/site_metrics/site-metrics-config.properties`.

- **`capacity_final`** (prefix `capacity_metrics/v1/capacity_final`) — end of the capacity chain
  `provider_tps_validate_v1` → `provider_tps_by_intervals_v1` → capacity_*; `provider_tps_validate_v1` reads
  `provider_combined_audit` + `common_output_format`. Generator `cron(0 2 …)`.
- **`cache_metrics_v1`** (`cache_metrics/v2`) ← `provider_combined_audit`; `cron(0 2 …)`.
- **`retry_metrics_v1`** (`retry_metrics/v2`) ← `provider_combined_audit`; `cron(0 2 …)`.
- **`import_metrics_v1`** (`import_metrics/v1`) ← `combined_audit` + `priceeye.customer_imports`; `cron(0 2 …)`.
- Site-metrics input SFN runs `cron(0 5 …)`. These run 02:00–05:00 UTC, after the hourly internal-monitoring run.

## Health / debugging signals

- `prod.monitoring.*` should have rows for the current `sales_date` within ~1–2h (hourly :10 run). If stale:
  confirm the SFN execution succeeded and that upstream `prod.priceeye_audits.*` received data for that date.
- Billing is daily (10:45 UTC) — "missing today" before then is expected.
- `provider_combined_audit` request impact = `SUM(inputrequestid_count)`; filter `sales_date = YYYYMMDD` (int).
- Naming caveat: Redshift billing schema is `billing` (Glue DB `billing_db`); `site_metrics.import_metrics`
  config name vs Glue resource `import_metrics_v1` differ — verify against Glue when in doubt.
