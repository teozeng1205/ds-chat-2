# S3 Bucket + Prefix Reference

All buckets follow the pattern `s3-atp-3victors-{env}-use1-{purpose}` where `{env}` is
`3vprod` for production or `3vdev` for development. The agent process runs on 3VDEV
AWS credentials. Some production S3 buckets are not listable from that account even when
the equivalent Redshift `prod.*` table is readable. For smoke tests and direct S3 reads,
prefer the verified accessible `3vdev` buckets unless the user explicitly asks for prod S3.
If a prod bucket returns `AccessDenied` or `NoSuchBucket`, report S3 as inaccessible/unknown
instead of concluding the data is absent.

## Buckets

### Collection anomalies
Verified accessible bucket: `s3-atp-3victors-3vdev-use1-collection-anomalies`
Production bucket may be inaccessible from 3VDEV credentials: `s3-atp-3victors-3vprod-use1-collection-anomalies`
- `collection-customer/v1/YYYY/MM/DD/` — Customer collection anomaly CSVs by date
- Fetch `collection-customer/v1/` or a date prefix, then filter customer rows in the
  returned data. Do not append `{customer}/` to this prefix; the verified layout is
  date-level files.

### Derived common output (DCO)
Bucket: `s3-atp-3victors-3vprod-use1-derived-common-output`
- `v1/{customer}/{YYYY}/{MM}/{DD}/{HH}/` — DCO Parquet (normalized price observations per customer)
- `v1/customer={code}/sales_date={YYYYMMDD}/` — Alternative partition path

### Anomaly datasets (v4 is current)
Bucket: `s3-atp-3victors-3vprod-use1-anomaly-datasets`
- `market-level/v4/{customer}/{YYYY}/{MM}/{DD}/` — Market-level anomaly Parquet (latest)
- `market-level/v3/customer={code}/sales_date={YYYYMMDD}/` — Legacy v3 path
- `segment-level/v4/{customer}/{YYYY}/{MM}/{DD}/` — Segment-level anomaly Parquet
- `daily_itins_prices/v2/{customer}/{YYYY}/{MM}/{DD}/` — Daily itinerary prices by AP band
- `oag_score/v2/{customer}/{YYYY}/{MM}/{DD}/` — OAG seat supply metrics
- `revenue_score/v1/{customer}/{YYYY}/{MM}/{DD}/revenue_estimates.csv` — Revenue estimates (CSV)
- `pax_midt/v1/{customer}/{YYYY}/{MM}/{DD}/` — PAX/MIDT booking data (CSV)

### Competitive position
Bucket: `s3-atp-3victors-3vprod-use1-competitive-position`
- `v2/{customer}/{YYYY}/{MM}/{DD}/data.parquet` — Competitive position Parquet

### PE common output (raw)
Verified accessible dev bucket: `s3-atp-3victors-3vdev-use1-pe-common-output`
Production bucket: `s3-atp-3victors-3vprod-use1-pe-common-output` if explicitly requested and accessible
- `{customer}/{YYYY}/{MM}/{DD}/{HH}/` — Raw common output before DCO normalization

## Redshift table → S3 mirror mapping

When a Redshift query returns 0 rows, the S3 mirror of the same pipeline's output is often
the fastest fallback. Key table-to-path mappings:

| Redshift table                       | S3 bucket                                                      | Key pattern |
|--------------------------------------|----------------------------------------------------------------|-------------|
| `market_level_anomalies_v4`          | `s3-atp-3victors-3vprod-use1-anomaly-datasets`                 | `market-level/v4/{customer}/{YYYY}/{MM}/{DD}/` |
| `market_level_anomalies_v3`          | `s3-atp-3victors-3vprod-use1-anomaly-datasets`                 | `market-level/v3/customer={code}/sales_date={YYYYMMDD}/` |
| `segment_level_anomalies_v*`         | `s3-atp-3victors-3vprod-use1-anomaly-datasets`                 | `segment-level/v4/{customer}/{YYYY}/{MM}/{DD}/` |
| `daily_itins_prices_v2`              | `s3-atp-3victors-3vprod-use1-anomaly-datasets`                 | `daily_itins_prices/v2/{customer}/{YYYY}/{MM}/{DD}/` |
| `oag_score_v2`                       | `s3-atp-3victors-3vprod-use1-anomaly-datasets`                 | `oag_score/v2/{customer}/{YYYY}/{MM}/{DD}/` |
| `revenue_score_v1`                   | `s3-atp-3victors-3vprod-use1-anomaly-datasets`                 | `revenue_score/v1/{customer}/{YYYY}/{MM}/{DD}/revenue_estimates.csv` |
| `pax_midt`                           | `s3-atp-3victors-3vprod-use1-anomaly-datasets`                 | `pax_midt/v1/{customer}/{YYYY}/{MM}/{DD}/` |
| `prod.common_output.*` (DCO)         | `s3-atp-3victors-3vprod-use1-derived-common-output`            | `v1/{customer}/{YYYY}/{MM}/{DD}/{HH}/` |
| collection anomalies                 | `s3-atp-3victors-3vdev-use1-collection-anomalies`              | `collection-customer/v1/YYYY/MM/DD/` |
| competitive-position (v2)            | `s3-atp-3victors-3vprod-use1-competitive-position`             | `v2/{customer}/{YYYY}/{MM}/{DD}/data.parquet` |

If a table doesn't appear above, search this document or call `search_kb("s3 ...")` with
the concept (e.g. `search_kb("oag score s3 path")`) to find the right bucket.

## Formats
`fetch_s3` reads CSV, Parquet, and JSONL automatically — no format flag needed.
