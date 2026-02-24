# Analytics Serverless Partition Notes

- `billing.customer_daily_requests_v*`: partitioned by `sales_date`.
- `common_output.common_output_format`: partitioned by `sales_date, customer`.
- `webfares.infare_pricing_data_v1`: partitioned by `customer, sales_date`.
- `channel_comparison` tables: all include `sales_date`; some also partition by `airline` and/or `period`.
- Query policy: always include `sales_date`; include `customer` when present; include `airline/period` for channel-comparison pruning.

## billing

- `billing.customer_daily_requests_v1`: sales_date
- `billing.customer_daily_requests_v2`: sales_date
- `billing.customer_daily_requests_v3`: sales_date

## channel_comparison

- `channel_comparison.carrier_miles_stats`: sales_date
- `channel_comparison.channel_availability`: sales_date
- `channel_comparison.channel_daily_compare`: airline, sales_date
- `channel_comparison.channel_period_summary`: period, airline, sales_date

## common_output

- `common_output.common_output_format`: sales_date, customer

## webfares

- `webfares.infare_pricing_data_v1`: customer, sales_date
