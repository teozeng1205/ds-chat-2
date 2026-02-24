# Partition Audit (Existing KB Redshift Tables)

Audit source: live `svv_external_columns.part_key` + `svv_external_partitions` checks.

## analytics_market_level_anomalies
- Physical: `prod.analytics.market_level_anomalies`
- Resolved query name: `prod.analytics.market_level_anomalies`
- External table: True
- Partition columns: sales_date, customer
- Partition count: 3467
- Sample partition values: ["20250116","B6"], ["20250210","AA"], ["20250210","B6"]

## analytics_market_level_anomalies_v3
- Physical: `prod.analytics.market_level_anomalies_v3`
- Resolved query name: `prod.analytics.market_level_anomalies_v3`
- External table: True
- Partition columns: customer, sales_date
- Partition count: 333
- Sample partition values: ["AS","20250914"], ["AS","20250915"], ["AS","20250916"]

## analytics_oag_score_v2
- Physical: `prod.analytics.oag_score_v2`
- Resolved query name: `prod.analytics.oag_score_v2`
- External table: True
- Partition columns: customer, run_date
- Partition count: 1422
- Sample partition values: ["AF","20251014"], ["AF","20251015"], ["AF","20251016"]

## analytics_revenue_score_v1
- Physical: `prod.analytics.revenue_score_v1`
- Resolved query name: `prod.analytics.revenue_score_v1`
- External table: True
- Partition columns: customer, sales_date
- Partition count: 941
- Sample partition values: ["AF","20251015"], ["AF","20251016"], ["AF","20251017"]

## analytics_segment_level_anomalies
- Physical: `prod.analytics.segment_level_anomalies`
- Resolved query name: `prod.analytics.segment_level_anomalies`
- External table: True
- Partition columns: sales_date, customer
- Partition count: 3467
- Sample partition values: ["20250116","B6"], ["20250210","AA"], ["20250210","B6"]

## analytics_segment_level_anomalies_v3
- Physical: `prod.analytics.segment_level_anomalies_v3`
- Resolved query name: `prod.analytics.segment_level_anomalies_v3`
- External table: True
- Partition columns: customer, sales_date
- Partition count: 317
- Sample partition values: ["AS","20250921"], ["AS","20250922"], ["AS","20250923"]

## monitoring_combined_audit
- Physical: `prod.monitoring.combined_audit`
- Resolved query name: `prod.monitoring.combined_audit`
- External table: True
- Partition columns: sales_date
- Partition count: 389
- Sample partition values: ["20250201"], ["20250202"], ["20250203"]

## monitoring_provider_combined_audit
- Physical: `prod.monitoring.provider_combined_audit`
- Resolved query name: `prod.monitoring.provider_combined_audit`
- External table: True
- Partition columns: sales_date
- Partition count: 389
- Sample partition values: ["20250201"], ["20250202"], ["20250203"]
