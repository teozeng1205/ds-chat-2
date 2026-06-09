# Workflow: Market-Level Anomalies

**Repo:** `ds-priceeye-analytics` (market-level-generator)
**Trigger:** Step Functions, fired after the SegmentLevel Spark job completes for a customer/sales_date.

## Ordered steps
1. SegmentLevel Spark produces `derived-common-output` parquet for the customer/sales_date.
2. Step Functions launches the market-level Python job (and the segment-level job) for that completion.
3. The job fetches source facts via the Redshift/Aurora DAO, runs the 22-day rolling Pandas model, and writes results.
4. Output parquet is written to S3 (`market-level/<customer>/YYYY/MM/DD/data.parquet`).
5. Glue partitions are registered per `partition_details` (`partition_order = sales_date,customer`).
6. If `emit_event = 1`, a `Task Completed` event fires on the EventBridge `data-pipeline` bus.

## Reads
- `derived-common-output` (S3, via Glue/Spectrum)
- competitive/source facts in `prod.analytics.*`

## Writes
- `prod.analytics.market_level_anomalies_v3` (current), older `_v2`; competitive view `prod.analytics.market_level_anomalies`
- S3 `market-level/<customer>/…` parquet

## Partition keys
`sales_date` (YYYYMMDD int) + `customer`. Always filter both; multi-day → `GROUP BY sales_date`.

## Health signals
- Latest `sales_date` present for the customer should track "yesterday".
- If empty for today: check whether the SegmentLevel Spark + Step Function execution ran and succeeded, and whether the partition was registered (`partition_details` / Glue).
- Columns: `metro_market`, `competitive_position`, `segment_name`, `itinerary_count`, `cp_score` (no `impact_score`).
