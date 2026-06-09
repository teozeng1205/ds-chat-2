# Workflow: Anomalies Pipeline (common output → anomalies)

**Repo:** `ds-priceeye-analytics`. **Cluster:** analytics (jobs read/write S3 parquet directly;
the `prod.analytics.*` / `prod.common_output.*` Glue tables are the queryable catalog over the
same S3). Bucket names show the prod form (`-3vprod-`); `${environment}` resolves per env.

Stage order (each `_SUCCESS`/event chains to the next):
**common output → DCO v2 → competitive position → market/segment analysis → market/segment anomalies → alerts.**

| # | Stage | Table (Glue / Redshift) | Partition keys | S3 bucket + prefix | Producer (repo/module) |
|---|---|---|---|---|---|
| 0 | common output (raw) | `prod.common_output.common_output_format` (`common_output_db`) | customer, sales_date | `s3-…-3vprod-…-pe-common-output` `/<customer>/YYYY/MM/DD/` | priceeye-v2 (upstream) |
| 1 | derived common output (DCO) | `prod.analytics.derived_common_output` (`derived_common_output_v2`) | sales_date, customer | `s3-…-derived-common-output` `/v2/<customer>/YYYY/MM/DD/<collection>/<run>/` | `source/01-dco/dco-v2-spark` |
| 2 | competitive position | `competitive_position_v2` (`prod.analytics.competitive_position`) | sales_date, customer | `s3-…-competitive-position` `/v2/<customer>/YYYY/MM/DD/` | `source/02-competitive-position/competitive-position` |
| 3a | market-level analysis | `market_level_analysis_v2` | customer, sales_date | `s3-…-anomaly-datasets` `/market-analysis/v2/<customer>/YYYY/MM/DD/` | `source/03-anomalies/market-level-analysis` |
| 3b | segment-level analysis | `segment_level_analysis_v2` | customer, sales_date | `s3-…-anomaly-datasets` `/segment-analysis/v2/…/` (+`_SUCCESS`) | `source/03-anomalies/segment-level-analysis` |
| 4a | market-level anomalies | `market_level_anomalies_v4` (`prod.analytics.*`) | customer, sales_date | `s3-…-anomaly-datasets` `/market-level/v4/<customer>/YYYY/MM/DD/` | `source/03-anomalies/market-level-generator` |
| 4b | segment-level anomalies | `segment_level_anomalies_v4` | customer, sales_date | `s3-…-anomaly-datasets` `/segment-level/v4/…/` | `source/03-anomalies/segment-level-generator` |

## Stage detail

**0 — common output.** Raw collected prices. DCO also joins audit datasets from
`s3-…-deduped-datasets` (`v1/provider_request_audit_detail`, `v1/packager_audit`,
`v1/refined_collection_run_audit`) and brand equivalence from `s3-…-ds-standard-brands/brand_equivalence/v1`.
Config: `docs/dco_v2.properties`.

**1 — DCO v2.** ECS Fargate `dco-v2-spark` (`source/01-dco/dco-v2-spark/src/main.py`); paths in
`dao/parquet_reader.py` (`build_output_path`). Writes `/v2/…` parquet + `_SUCCESS`. Triggered by
`dco-v2-spark-trigger`, which scans `prod.priceeye_audits.provider_request_audit_detail` +
`local.monitoring.refined_collection_run_audit` for completed collections in a `(p_date, p_hour)`
window and `run_task`s the spark job per customer/sales_date.

**2 — competitive position.** ECS `competitive-position` (class `CompetitivePosition`, `src/main.py`).
Reads DCO (`competitive-position.properties` references `input.version=v1` + `prod.analytics.derived_common_output`);
writes `/v2/…`. Its own scheduler/event rule is **not in code** (the YAML is a task definition only).

**3 — analysis (market + segment).** ECS `market-level-analysis` / `segment-level-analysis`
(`src/main.py`). Read competitive-position v2; write `market-analysis/v2` and `segment-analysis/v2`
(+ `_SUCCESS`) under the `anomaly-datasets` bucket. Anomaly `threshold=0.07`. Supporting scores in
the same bucket feed the generators: `oag_score/v2` (`cron(0 2 …)`), `revenue_score/v1` (`cron(15 23 …)`),
`daily_itins_prices/v2` (`cron(30 12 …)`).

**4 — anomaly generators (final tables).** ECS `market-level-generator` reads `market-analysis/v2`
(~22-day window) + `oag_score/v2` + `revenue_score/v1` + direction scores, writes `market-level/v4`,
and registers the Glue partition in-code (`glue_util.add_partition(database, table, s3, [customer, run_date])`,
table `market_level_anomalies_v4`). ECS `segment-level-generator` reads `segment-analysis/v2` **and**
`market-level/v4`, writes `segment-level/v4`.

## Orchestration (Stage 4 hub)

`source/deploy/definitions/ds-analytics-eventdriven-jobs-step-function.asl.json` +
`commonfiles/ds-analytics-eventdriven-jobs.yaml`:
- EventBridge rule on S3 `Object Created`, key `segment-analysis/v2/*/*/*/*/_SUCCESS` →
  starts state machine `DS-Analytics-EventDriven-Jobs` with `taskName=SegmentLevel`.
- Also listens on bus `data-pipeline`, `detail-type="Task Completed"`, `source="threevictors.ecs.analytics"`,
  `detail.taskName ∈ {MarketLevel, SegmentLevel}`.
- Step Function order: **market-level-generator → (on success) segment-level-generator**; customer/sales_date
  parsed from the S3 key.

## Alerts (terminal)

Lambda `source/04-Alerts/alerts/src/lambda_main.py` reads `*_anomalies_v4` (analytics), emits an
EventBridge `data-pipeline` business event `{"SubType":"Price Anomaly", detail-type:"SegmentLevel"}`,
and writes an audit payload to `s3-…-pe-analytics-audits`. Per-customer UTC-hour gating from a MySQL schedule.

## Health / debugging signals

- Empty/late `market_level_anomalies_v4` for today: check the segment-analysis `_SUCCESS` fired, the
  `DS-Analytics-EventDriven-Jobs` execution, and whether the Glue partition for `[customer, sales_date]` was added.
- Stale upstream cascades: DCO → CP → analysis → generators; a gap at any stage starves the next.
- `segment_level_anomalies_v4` partition registration is done by the partition-creator registry (not the
  generator job), so a present table with no rows for a date can mean a missing partition.

Columns: `market_level_anomalies` uses `metro_market`, `competitive_position`, `segment_name`,
`itinerary_count`, `cp_score` (no `impact_score`); the v4 generators add impact-score columns.
