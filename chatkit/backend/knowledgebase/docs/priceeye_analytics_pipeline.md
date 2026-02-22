# PriceEye Analytics Pipeline (Brief)

Scope: map ds-priceeye-analytics/priceeye-analytics flow for anomaly investigations in `3VDEV`.

## Data Flow

1. Raw search capture: `s3://s3-atp-3victors-<env>/common-output/<customer>/YYYY/MM/DD/HH/`
2. Derived common output (Spark): `derived-common-output/<version>/<customer>/YYYY/MM/DD/HH/`
3. Competitive position (Spark): writes parquet + feeds Redshift anomaly tables.
4. Spark market/segment jobs publish `Task Completed` events.
5. DS Python generators (market-level then segment-level) compute anomaly flags + impact score and publish parquet.

## Primary Investigation Tables

- Redshift:
  - `analytics.market_level_anomalies`
  - `analytics.segment_level_anomalies`
  - `analytics.market_level_anomalies_v3`
  - `analytics.segment_level_anomalies_v3`
  - `analytics.oag_score_v2`
  - `analytics.revenue_score_v1`
- Aurora MySQL:
  - `analytics.anomalies_direction_score`
  - `analytics.anomalies_impact_score_weights`

## Key Partition Guidance

- Most anomaly facts are partitioned by `customer` and `sales_date`.
- Always enforce both predicates when required.
- For S3 anomaly datasets, prefer parquet and prune by customer/date prefix.

## High-Value Columns for Analysis

- Dimensions: `customer`, `sales_date`, `observation_date`, `mkt/seg_mkt`, `seg/segment_name`, `cp/competitive_position`
- Signals: `freq_pcnt`, `mag_nominal`, `mag_pcnt`, `impact_score`, `top_offenders`
- Supporting scores: `oag_score`, `revenue_score`, `direction_score`, customer `weights`
