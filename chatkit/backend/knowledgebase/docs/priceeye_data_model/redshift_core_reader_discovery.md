# Redshift Core Reader Discovery

Discovery source: `svv_external_tables`, `svv_external_columns`, `svv_external_partitions` via 3VDEV core reader.

- External tables discovered (analytics/monitoring/adf): 37
- New KB table specs added from this pass: 29 (`ext_*` table IDs).
- Temporary Athena scratch tables (`temp_table_*`) were excluded.

## adf (1 tables)

### adf.assembled_data_feed_v1
- Query name: `adf.assembled_data_feed_v1`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-adf/assembled_data_feed_emr/`
- Partition columns: customer, sales_date, sales_hour
- Partition count (catalog): 9
- Sample partition values: ["AS","20260129","10"], ["B6","20260210","10"], ["TS","20260210","20"]

## analytics (19 tables)

### analytics.competitive_position
- Query name: `prod.analytics.competitive_position`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-competitive-position/v1/`
- Partition columns: sales_date, customer
- Partition count (catalog): 4161
- Sample partition values: ["20250814","WNVacations"], ["20250306","WN"], ["20250923","SabreCERT"]

### analytics.competitive_position_v2
- Query name: `analytics.competitive_position_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-competitive-position/v2/`
- Partition columns: sales_date, customer
- Partition count (catalog): 11
- Sample partition values: ["20260223","CX"], ["20260223","B6"], ["20260223","YY"]

### analytics.daily_itins_prices_v2
- Query name: `prod.analytics.daily_itins_prices_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/daily_itins_prices/v2/`
- Partition columns: customer, sales_date
- Partition count (catalog): 1080
- Sample partition values: ["XY","20251121"], ["B6","20251030"], ["INS","20250922"]

### analytics.derived_common_output
- Query name: `prod.analytics.derived_common_output`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-derived-common-output/v1/`
- Partition columns: sales_date, customer
- Partition count (catalog): 4164
- Sample partition values: ["20250920","Sanity"], ["20250917","LH"], ["20251009","B6"]

### analytics.derived_common_output_v2
- Query name: `analytics.derived_common_output_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-derived-common-output/v2/`
- Partition columns: sales_date, customer
- Partition count (catalog): 4
- Sample partition values: ["20260219","B6"], ["20260220","B6"], ["20260222","B6"]

### analytics.market_level_analysis_v2
- Query name: `analytics.market_level_analysis_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/market-analysis/v2/`
- Partition columns: customer, sales_date
- Partition count (catalog): 7
- Sample partition values: ["YY","20260222"], ["B6","20260224"], ["B6","20260222"]

### analytics.market_level_anomalies
- Query name: `prod.analytics.market_level_anomalies`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/market-level/v1/`
- Partition columns: sales_date, customer
- Partition count (catalog): 3467
- Sample partition values: ["20250325","B6"], ["20250309","SK"], ["20250420","UA"]

### analytics.market_level_anomalies_v2
- Query name: `prod.analytics.market_level_anomalies_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/market-level/v2/`
- Partition columns: customer, sales_date
- Partition count (catalog): 222
- Sample partition values: ["AS","20250731"], ["AS","20250722"], ["B6","20250717"]

### analytics.market_level_anomalies_v3
- Query name: `prod.analytics.market_level_anomalies_v3`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/market-level/v3/`
- Partition columns: customer, sales_date
- Partition count (catalog): 333
- Sample partition values: ["SK","20251001"], ["B6","20260109"], ["INS","20251124"]

### analytics.market_level_anomalies_v4
- Query name: `analytics.market_level_anomalies_v4`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/market-level/v4/`
- Partition columns: customer, sales_date
- Partition count (catalog): 5
- Sample partition values: ["YY","20260223"], ["B6","20260222"], ["B6","20260223"]

### analytics.oag_score_v2
- Query name: `prod.analytics.oag_score_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/oag_score/v2/`
- Partition columns: customer, run_date
- Partition count (catalog): 1422
- Sample partition values: ["AS","20251114"], ["B6","20260204"], ["LA","20251031"]

### analytics.pax_midt
- Query name: `prod.analytics.pax_midt`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/pax_midt/v1/`
- Partition columns: customer, sales_date
- Partition count (catalog): 1114
- Sample partition values: ["YY","20251027"], ["CH","20251003"], ["B6","20260115"]

### analytics.price_outlook
- Query name: `prod.analytics.price_outlook`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-price-outlook/v1/`
- Partition columns: sales_date, customer
- Partition count (catalog): 4133
- Sample partition values: ["20250221","WNVacations"], ["20250729","Advito"], ["20250422","SK"]

### analytics.revenue_score_v1
- Query name: `prod.analytics.revenue_score_v1`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/revenue_score/v1/`
- Partition columns: customer, sales_date
- Partition count (catalog): 941
- Sample partition values: ["WN","20250924"], ["AS","20251005"], ["AF","20260211"]

### analytics.segment_level_analysis_v2
- Query name: `analytics.segment_level_analysis_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/segment-analysis/v2/`
- Partition columns: customer, sales_date
- Partition count (catalog): 7
- Sample partition values: ["B6","20260221"], ["YY","20260221"], ["YY","20260222"]

### analytics.segment_level_anomalies
- Query name: `prod.analytics.segment_level_anomalies`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/segment-level/v1/`
- Partition columns: sales_date, customer
- Partition count (catalog): 3467
- Sample partition values: ["20250228","DL"], ["20250227","GJ"], ["20250310","B6"]

### analytics.segment_level_anomalies_v2
- Query name: `prod.analytics.segment_level_anomalies_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/segment-level/v2/`
- Partition columns: customer, sales_date
- Partition count (catalog): 212
- Sample partition values: ["AS","20250925"], ["SK","20250810"], ["AS","20250815"]

### analytics.segment_level_anomalies_v3
- Query name: `prod.analytics.segment_level_anomalies_v3`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/segment-level/v3/`
- Partition columns: customer, sales_date
- Partition count (catalog): 317
- Sample partition values: ["YY","20251123"], ["B6","20251003"], ["SK","20251123"]

### analytics.segment_level_anomalies_v4
- Query name: `analytics.segment_level_anomalies_v4`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-anomaly-datasets/segment-level/v4/`
- Partition columns: customer, sales_date
- Partition count (catalog): 5
- Sample partition values: ["YY","20260223"], ["B6","20260223"], ["YY","20260222"]

## monitoring (17 tables)

### monitoring.auto_schedule_requests
- Query name: `prod.monitoring.auto_schedule_requests`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-as-converted-persistence/v1/`
- Partition columns: generation_id
- Partition count (catalog): 187
- Sample partition values: ["3189"], ["3226"], ["3251"]

### monitoring.combined_audit
- Query name: `prod.monitoring.combined_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/combined_audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20260107"], ["20251229"], ["20250719"]

### monitoring.customer_combined_audit_v1
- Query name: `prod.monitoring.customer_combined_audit_v1`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-customer-monitor/v1/customer-combined-audit/`
- Partition columns: sales_date
- Partition count (catalog): 244
- Sample partition values: ["20250413"], ["20250820"], ["20250802"]

### monitoring.customer_combined_audit_v2
- Query name: `prod.monitoring.customer_combined_audit_v2`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-customer-monitor/v2/customer-combined-audit/`
- Partition columns: sales_date
- Partition count (catalog): 184
- Sample partition values: ["20250920"], ["20250914"], ["20251027"]

### monitoring.deduped_cache_loader_audit
- Query name: `prod.monitoring.deduped_cache_loader_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/cache_loader_audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20250323"], ["20250320"], ["20250725"]

### monitoring.deduped_delivery_audit
- Query name: `prod.monitoring.deduped_delivery_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/delivery_audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20250514"], ["20251220"], ["20250526"]

### monitoring.deduped_enrichment_audit
- Query name: `prod.monitoring.deduped_enrichment_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/enrichment_audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20251106"], ["20250915"], ["20250517"]

### monitoring.deduped_global_filter_audit_summary
- Query name: `prod.monitoring.deduped_global_filter_audit_summary`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/global_filter_audit_summary/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20250605"], ["20260119"], ["20250720"]

### monitoring.deduped_packager_audit
- Query name: `prod.monitoring.deduped_packager_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/packager_audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20250725"], ["20250506"], ["20250708"]

### monitoring.deduped_provider_request_audit
- Query name: `prod.monitoring.deduped_provider_request_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/provider_request_audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20250629"], ["20250911"], ["20250503"]

### monitoring.deduped_provider_request_audit_detail
- Query name: `prod.monitoring.deduped_provider_request_audit_detail`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/provider_request_audit_detail/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20251002"], ["20250507"], ["20251215"]

### monitoring.deduped_provider_response_audit
- Query name: `prod.monitoring.deduped_provider_response_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/provider_response_audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20250902"], ["20251115"], ["20250723"]

### monitoring.deduped_retry_audit
- Query name: `prod.monitoring.deduped_retry_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-deduped-datasets/v1/retry_audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20250222"], ["20260111"], ["20250220"]

### monitoring.provider_combined_audit
- Query name: `prod.monitoring.provider_combined_audit`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-provider-monitor/v1/provider-combined-audit/`
- Partition columns: sales_date
- Partition count (catalog): 389
- Sample partition values: ["20250418"], ["20250207"], ["20250816"]

### monitoring.provider_health
- Query name: `monitoring.provider_health`
- S3 location: `s3://price-eye-provider-health-parquet/provider-health/v1/`
- Partition columns: eval_date, eval_hour
- Partition count (catalog): 5
- Sample partition values: ["20260221","15"], ["20260220","14"], ["20260220","4"]

### monitoring.provider_site_health
- Query name: `monitoring.provider_site_health`
- S3 location: `s3://price-eye-provider-health-parquet/provider-site-health/v1/`
- Partition columns: eval_date, eval_hour
- Partition count (catalog): 5
- Sample partition values: ["20260220","4"], ["20260220","14"], ["20260222","14"]

### monitoring.response_dupes
- Query name: `monitoring.response_dupes`
- S3 location: `s3://s3-atp-3victors-3vdev-use1-provider-monitor/v1/response-dupes`
- Partition columns: sales_date
- Partition count (catalog): 112
- Sample partition values: ["20251105"], ["20260124"], ["20251222"]
