# Analytics Serverless Reader Discovery

Reader profile: `database-analytics-redshift-serverless-reader.properties`.
Targeted schemas: `billing`, `channel_comparison`, `common_output`, `webfares`.

- Tables scanned: 9
- Resolved/queryable tables: 9
- New KB table specs added: 9 (`analytics_serverless_*`).

## billing (3 tables)

### billing.customer_daily_requests_v1
- Query name: `prod.billing.customer_daily_requests_v1`
- Partition columns: sales_date
- Partition count: 309
- Sample partition values: ["20250625"], ["20250808"], ["20250819"]
- Location: `s3://s3-atp-3victors-3vdev-use1-billing/v1/customer_daily_requests`

### billing.customer_daily_requests_v2
- Query name: `prod.billing.customer_daily_requests_v2`
- Partition columns: sales_date
- Partition count: 206
- Sample partition values: ["20260101"], ["20251005"], ["20260120"]
- Location: `s3://s3-atp-3victors-3vdev-use1-billing/v2/customer_daily_requests`

### billing.customer_daily_requests_v3
- Query name: `billing.customer_daily_requests_v3`
- Partition columns: sales_date
- Partition count: 84
- Sample partition values: ["20251222"], ["20251223"], ["20260221"]
- Location: `s3://s3-atp-3victors-3vdev-use1-billing/v3/customer_daily_requests`

## channel_comparison (4 tables)

### channel_comparison.carrier_miles_stats
- Query name: `channel_comparison.carrier_miles_stats`
- Partition columns: sales_date
- Partition count: 12
- Sample partition values: ["20260103"], ["20260203"], ["20251103"]
- Location: `s3://s3-atp-3victors-3vdev-use1-chnl-comp-datasets/v1/carrier_miles_stats/`

### channel_comparison.channel_availability
- Query name: `channel_comparison.channel_availability`
- Partition columns: sales_date
- Partition count: 13
- Sample partition values: ["20251012"], ["20260222"], ["20260215"]
- Location: `s3://s3-atp-3victors-3vdev-use1-chnl-comp-datasets/cc_exec/v1/channel_availability`

### channel_comparison.channel_daily_compare
- Query name: `channel_comparison.channel_daily_compare`
- Partition columns: airline, sales_date
- Partition count: 18139
- Sample partition values: ["LA","20250720"], ["LH","20250625"], ["LH","20260111"]
- Location: `s3://s3-atp-3victors-3vdev-use1-chnl-comp-datasets/cc_exec/v1/daily_compare/`

### channel_comparison.channel_period_summary
- Query name: `channel_comparison.channel_period_summary`
- Partition columns: period, airline, sales_date
- Partition count: 2003
- Sample partition values: ["monthly","HX","20250817"], ["weekly","QR","20260222"], ["weekly","AF","20250831"]
- Location: `s3://s3-atp-3victors-3vdev-use1-chnl-comp-datasets/cc_exec/v1/period_summary`

## common_output (1 tables)

### common_output.common_output_format
- Query name: `prod.common_output.common_output_format`
- Partition columns: sales_date, customer
- Partition count: 2630
- Sample partition values: ["20250613","Test0207"], ["20250301","Sanity1"], ["20250709","Test0207"]
- Location: `s3://s3-atp-3victors-3vdev-use1-pe-common-output/`

## webfares (1 tables)

### webfares.infare_pricing_data_v1
- Query name: `webfares.infare_pricing_data_v1`
- Partition columns: customer, sales_date
- Partition count: 16
- Sample partition values: ["LA","20260205"], ["F9","20260205"], ["AZ","20260205"]
- Location: `s3://s3-atp-3victors-3vdev-use1-webfares/infare/v1`
