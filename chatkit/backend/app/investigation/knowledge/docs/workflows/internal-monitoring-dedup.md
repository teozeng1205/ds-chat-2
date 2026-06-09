# Workflow: Internal Monitoring (Dedup + Join Audit)

**Repo:** `ds-internal-monitoring`
**Trigger:** Hourly schedule.

## Ordered steps
1. Read raw collection/audit signals originating from `priceeye-v2` collection.
2. Deduplicate input request records and join provider/site context.
3. Aggregate to provider/site granularity.
4. Write the combined audit tables in `prod.monitoring.*` (Core Redshift, Glue-backed external schema `monitoring`).

## Reads
- Upstream collection/audit data from `priceeye-v2`.

## Writes
- `prod.monitoring.combined_audit` (singular `issue_source`, `issue_reason`)
- `prod.monitoring.provider_combined_audit` (plural `issue_sources`, `issue_reasons`; aggregate `inputrequestid_count`, no raw `inputrequestid`, no `status`)

## Partition keys
`sales_date` (YYYYMMDD int). Filter `WHERE sales_date = YYYYMMDD`; for provider questions add `providercode`.

## Health signals
- Should have a row set for the current `sales_date` within ~1-2 hours.
- For request impact use `SUM(inputrequestid_count)`, never `COUNT(DISTINCT inputrequestid)`.
- If stale: confirm the hourly job ran; check whether upstream `priceeye-v2` collection produced data for that date.
