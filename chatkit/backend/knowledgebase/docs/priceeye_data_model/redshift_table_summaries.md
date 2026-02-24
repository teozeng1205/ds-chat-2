# Redshift Table Summaries

Profile source: live schema inspection in 3VDEV.

## analytics_market_level_anomalies
- Physical: `prod.analytics.market_level_anomalies`
- Column count: 19
- Key columns (first 12): segment_name, competitive_position, metro_market, region_name, depart_period, carrier_group, cabin_group, top_offenders, carrier_contribution, itinerary_count, itinerary_percentage, impacted_dates
- Sample dimensions inspected: customer, sales_date, segment_name, region_name

## analytics_market_level_anomalies_v3
- Physical: `prod.analytics.market_level_anomalies_v3`
- Column count: 58
- Key columns (first 12): observation_date, mkt, seg, region_name, depart_period, carrier_group, cabin_group, top_offenders, impacted_dates, impact_dates_pcnt, itinerary_count, cp
- Sample dimensions inspected: customer, sales_date, observation_date, cp, region_name

## analytics_oag_score_v2
- Physical: `prod.analytics.oag_score_v2`
- Column count: 14
- Key columns (first 12): origin_metro, destination_metro, ct, carrier_code, number_of_seats, metro_od, normalized_market_score, normalized_customer_score, customer_market_share, total_seats_in_market, carrier_scores, oag_score_sum
- Sample dimensions inspected: customer

## analytics_revenue_score_v1
- Physical: `prod.analytics.revenue_score_v1`
- Column count: 10
- Key columns (first 12): ap_band, origin_metro, destination_metro, cabin_group, midt_pax, selected_fare, estimated_revenue, revenue_score, customer, sales_date
- Sample dimensions inspected: customer, sales_date

## analytics_segment_level_anomalies
- Physical: `prod.analytics.segment_level_anomalies`
- Column count: 20
- Key columns (first 12): segment_name, competitive_position, region_name, depart_period, carrier_group, cabin_group, top_offenders, carrier_contribution, itinerary_count, itinerary_percentage, impacted_markets, impacted_markets_percentage
- Sample dimensions inspected: customer, sales_date, segment_name, region_name

## analytics_segment_level_anomalies_v3
- Physical: `prod.analytics.segment_level_anomalies_v3`
- Column count: 53
- Key columns (first 12): observation_date, region_name, depart_period, carrier_group, cabin_group, top_offenders, impacted_markets, impacted_mkt_pcnt, itinerary_count, cp, dow, segment
- Sample dimensions inspected: customer, sales_date, observation_date, cp, region_name

## monitoring_combined_audit
- Physical: `prod.monitoring.combined_audit`
- Column count: 109
- Key columns (first 12): id, inputrequestid, customer, customercollectionid, customercollectionname, reference, sitecategory, customer_salesdate, customersitecode, customerpos, scheduledate, scheduletime
- Sample dimensions inspected: customer, providercode, sitecode, sales_date

## monitoring_provider_combined_audit
- Physical: `prod.monitoring.provider_combined_audit`
- Column count: 52
- Key columns (first 12): id, customers, crawl_start_date, scheduledate, scheduletime, actualscheduletimestamp, observationtimestamp, providercode, sitecode, pos, carriercodes, connectionairportcodes
- Sample dimensions inspected: customers, providercode, sitecode, sales_date
