# Analytics Reader Discovery

Reader profile: `database-priceeye-analytics-reader.properties`.
Catalog source: `svv_external_tables`, `svv_external_columns`, `svv_external_partitions`.

- Tables scanned: 44
- Resolved/queryable tables: 44
- KB table specs added: 25 (`analytics_reader_*`).

## analytics (19 tables)

### analytics.competitive_position
- Query name: `prod.analytics.competitive_position`
- Partition columns: sales_date, customer
- Partition count: 4161
- Sample partition values: ["20251019","AF"], ["20250326","GJ"], ["20250312","AS"]

### analytics.competitive_position_v2
- Query name: `analytics.competitive_position_v2`
- Partition columns: sales_date, customer
- Partition count: 11
- Sample partition values: ["20260223","B6"], ["20260223","CX"], ["20260222","CX"]

### analytics.daily_itins_prices_v2
- Query name: `prod.analytics.daily_itins_prices_v2`
- Partition columns: customer, sales_date
- Partition count: 1080
- Sample partition values: ["B6","20260213"], ["YY","20251121"], ["QATEST","20251128"]

### analytics.derived_common_output
- Query name: `prod.analytics.derived_common_output`
- Partition columns: sales_date, customer
- Partition count: 4164
- Sample partition values: ["20250325","AA"], ["20250513","WN"], ["20250329","Sabre"]

### analytics.derived_common_output_v2
- Query name: `analytics.derived_common_output_v2`
- Partition columns: sales_date, customer
- Partition count: 4
- Sample partition values: ["20260221","B6"], ["20260219","B6"], ["20260220","B6"]

### analytics.market_level_analysis_v2
- Query name: `analytics.market_level_analysis_v2`
- Partition columns: customer, sales_date
- Partition count: 7
- Sample partition values: ["YY","20260223"], ["B6","20260224"], ["B6","20260222"]

### analytics.market_level_anomalies
- Query name: `prod.analytics.market_level_anomalies`
- Partition columns: sales_date, customer
- Partition count: 3467
- Sample partition values: ["20250808","GJ"], ["20250314","Sanity"], ["20250905","INS"]

### analytics.market_level_anomalies_v2
- Query name: `prod.analytics.market_level_anomalies_v2`
- Partition columns: customer, sales_date
- Partition count: 222
- Sample partition values: ["SK","20250910"], ["SK","20250908"], ["INS","20250913"]

### analytics.market_level_anomalies_v3
- Query name: `prod.analytics.market_level_anomalies_v3`
- Partition columns: customer, sales_date
- Partition count: 333
- Sample partition values: ["YY","20251116"], ["INS","20250928"], ["INS","20250924"]

### analytics.market_level_anomalies_v4
- Query name: `analytics.market_level_anomalies_v4`
- Partition columns: customer, sales_date
- Partition count: 5
- Sample partition values: ["YY","20260222"], ["B6","20260223"], ["B6","20260224"]

### analytics.oag_score_v2
- Query name: `prod.analytics.oag_score_v2`
- Partition columns: customer, run_date
- Partition count: 1422
- Sample partition values: ["AS","20250813"], ["B6","20251207"], ["INS","20250914"]

### analytics.pax_midt
- Query name: `prod.analytics.pax_midt`
- Partition columns: customer, sales_date
- Partition count: 1114
- Sample partition values: ["WNVacations","20251003"], ["B6","20251212"], ["LA","20251125"]

### analytics.price_outlook
- Query name: `prod.analytics.price_outlook`
- Partition columns: sales_date, customer
- Partition count: 4133
- Sample partition values: ["20251122","WNVacations"], ["20250530","CHNL"], ["20250227","WNVacations"]

### analytics.revenue_score_v1
- Query name: `prod.analytics.revenue_score_v1`
- Partition columns: customer, sales_date
- Partition count: 941
- Sample partition values: ["MH","20251103"], ["CH","20251002"], ["MH","20251210"]

### analytics.segment_level_analysis_v2
- Query name: `analytics.segment_level_analysis_v2`
- Partition columns: customer, sales_date
- Partition count: 7
- Sample partition values: ["B6","20260223"], ["YY","20260221"], ["B6","20260221"]

### analytics.segment_level_anomalies
- Query name: `prod.analytics.segment_level_anomalies`
- Partition columns: sales_date, customer
- Partition count: 3467
- Sample partition values: ["20250303","WNVacations"], ["20250228","DL"], ["20250313","WNVacations"]

### analytics.segment_level_anomalies_v2
- Query name: `prod.analytics.segment_level_anomalies_v2`
- Partition columns: customer, sales_date
- Partition count: 212
- Sample partition values: ["AS","20250921"], ["AS","20250819"], ["B6","20250811"]

### analytics.segment_level_anomalies_v3
- Query name: `prod.analytics.segment_level_anomalies_v3`
- Partition columns: customer, sales_date
- Partition count: 317
- Sample partition values: ["B6","20260205"], ["INS","20251123"], ["B6","20251107"]

### analytics.segment_level_anomalies_v4
- Query name: `analytics.segment_level_anomalies_v4`
- Partition columns: customer, sales_date
- Partition count: 5
- Sample partition values: ["B6","20260222"], ["B6","20260223"], ["YY","20260223"]

## analytics_mysql (12 tables)

### analytics_mysql.alerts_schedule
- Query name: `analytics_mysql.alerts_schedule`
- Partition columns: none
- Partition count: 0

### analytics_mysql.anomalies_direction_score
- Query name: `analytics_mysql.anomalies_direction_score`
- Partition columns: none
- Partition count: 0

### analytics_mysql.anomalies_impact_score_weights
- Query name: `analytics_mysql.anomalies_impact_score_weights`
- Partition columns: none
- Partition count: 0

### analytics_mysql.cabin_group
- Query name: `analytics_mysql.cabin_group`
- Partition columns: none
- Partition count: 0

### analytics_mysql.carrier_group
- Query name: `analytics_mysql.carrier_group`
- Partition columns: none
- Partition count: 0

### analytics_mysql.city_code_override
- Query name: `analytics_mysql.city_code_override`
- Partition columns: none
- Partition count: 0

### analytics_mysql.date_range
- Query name: `analytics_mysql.date_range`
- Partition columns: none
- Partition count: 0

### analytics_mysql.demo_carrier_substitutions
- Query name: `analytics_mysql.demo_carrier_substitutions`
- Partition columns: none
- Partition count: 0

### analytics_mysql.geography
- Query name: `analytics_mysql.geography`
- Partition columns: none
- Partition count: 0

### analytics_mysql.geography_entry
- Query name: `analytics_mysql.geography_entry`
- Partition columns: none
- Partition count: 0

### analytics_mysql.region
- Query name: `analytics_mysql.region`
- Partition columns: none
- Partition count: 0

### analytics_mysql.segment
- Query name: `analytics_mysql.segment`
- Partition columns: none
- Partition count: 0

## federated_analytics (12 tables)

### federated_analytics.alerts_schedule
- Query name: `federated_analytics.alerts_schedule`
- Partition columns: none
- Partition count: 0

### federated_analytics.anomalies_direction_score
- Query name: `federated_analytics.anomalies_direction_score`
- Partition columns: none
- Partition count: 0

### federated_analytics.anomalies_impact_score_weights
- Query name: `federated_analytics.anomalies_impact_score_weights`
- Partition columns: none
- Partition count: 0

### federated_analytics.cabin_group
- Query name: `federated_analytics.cabin_group`
- Partition columns: none
- Partition count: 0

### federated_analytics.carrier_group
- Query name: `federated_analytics.carrier_group`
- Partition columns: none
- Partition count: 0

### federated_analytics.city_code_override
- Query name: `federated_analytics.city_code_override`
- Partition columns: none
- Partition count: 0

### federated_analytics.date_range
- Query name: `federated_analytics.date_range`
- Partition columns: none
- Partition count: 0

### federated_analytics.demo_carrier_substitutions
- Query name: `federated_analytics.demo_carrier_substitutions`
- Partition columns: none
- Partition count: 0

### federated_analytics.geography
- Query name: `federated_analytics.geography`
- Partition columns: none
- Partition count: 0

### federated_analytics.geography_entry
- Query name: `federated_analytics.geography_entry`
- Partition columns: none
- Partition count: 0

### federated_analytics.region
- Query name: `federated_analytics.region`
- Partition columns: none
- Partition count: 0

### federated_analytics.segment
- Query name: `federated_analytics.segment`
- Partition columns: none
- Partition count: 0

## pe_analytics_audits (1 tables)

### pe_analytics_audits.alerts_audit_v1
- Query name: `pe_analytics_audits.alerts_audit_v1`
- Partition columns: sales_date
- Partition count: 27
- Sample partition values: ["20251018"], ["20251012"], ["20251017"]
