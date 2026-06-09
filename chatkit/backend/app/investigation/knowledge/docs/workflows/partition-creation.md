# Workflow: Glue Partition Creation

**Repo:** `partition-creator` (config in the `partition_details` MySQL table)
**Trigger:** After a producer writes parquet to an S3 prefix (often chained via EventBridge).

## How it works
A row in the `partition_details` MySQL table declares, per destination table:
- `bucket`, `preamble`, `pattern`, `partition_pattern` — the S3 prefix layout.
- `partition_order` — the partition-column order (e.g. `sales_date,customer`). **Must match the
  parquet schema ordering**, since partition columns are excluded from the parquet payload and
  supplied at partition-creation time. Misalignment causes Glue/Athena query failures.
- `destination_database`, `destination_table` — the Glue catalog target.
- `emit_event` — whether a completion event fires on the EventBridge `data-pipeline` bus.

## Reads
- New S3 objects under the configured prefix.
- `partition_details` (MySQL) for the registration rules.

## Writes
- Glue partitions on the destination Glue table (which Redshift external schemas / Athena then query).

## Health signals
- A table that exists but returns no rows for a date may simply be **missing its partition** —
  check whether the producer wrote the S3 object and whether the partition was registered.
- If a downstream consumer wasn't notified, check `emit_event` for that `destination_table`.
- Partition-column order in `partition_order` must match the parquet schema.
