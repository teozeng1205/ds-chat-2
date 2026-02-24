# MySQL Table Summaries (priceeye schema)

Profile source: live schema discovery in 3VDEV (`information_schema`).

## Analytics Support

### priceeye.airline_capacity_rank
- Approx rows: 705
- Columns (5): airline_code, capacity_seats, capacity_rank, source, updated_at
- Summary: airline_capacity_rank table for analytics support; ~705 rows; key columns: airline_code, capacity_seats, capacity_rank, source, updated_at.

### priceeye.tmp_rank_mysql
- Approx rows: 705
- Columns (3): carrier_code, capacity_seats, capacity_rank
- Summary: tmp_rank_mysql table for analytics support; ~705 rows; key columns: carrier_code, capacity_seats, capacity_rank.

## Customer Config

### priceeye.customer
- Approx rows: 44
- Columns (8): name, description, status, customer_type, timezone, industry_vertical, last_updated, last_updated_by
- Summary: customer table for customer config; ~44 rows; key columns: name, description, status, customer_type, timezone.

### priceeye.customer_billing
- Approx rows: 55
- Columns (7): name, billing_code, owned_carrier, iris, volume, pirisfes, last_updated
- Summary: customer_billing table for customer config; ~55 rows; key columns: name, billing_code, owned_carrier, iris, volume.

### priceeye.customer_collection
- Approx rows: 405
- Columns (14): id, customer, name, description, frequency, earliestStartTime, expectedDeliveryTime, run_hour, hints, input_file_format, status, customerPackagingId ...
- Summary: customer_collection table for customer config; ~405 rows; key columns: id, customer, name, description, frequency.

### priceeye.customer_collection_delivery
- Approx rows: 134
- Columns (4): id, deliveryId, customerCollectionId, last_updated
- Summary: customer_collection_delivery table for customer config; ~134 rows; key columns: id, deliveryId, customerCollectionId, last_updated.

### priceeye.customer_collection_input_files
- Approx rows: 208
- Columns (3): customerCollectionId, inputId, last_updated
- Summary: customer_collection_input_files table for customer config; ~208 rows; key columns: customerCollectionId, inputId, last_updated.

### priceeye.customer_defaults
- Approx rows: 64
- Columns (12): customer, packagingConfiguration, deliveryConfiguration, inputFormat, outputFormat, is_analytics, is_channel_comparison, is_monitoring, monitoring_deletion_date, analytics_deletion_date, channel_comparisons_deletion_date, last_updated_by
- Summary: customer_defaults table for customer config; ~64 rows; key columns: customer, packagingConfiguration, deliveryConfiguration, inputFormat, outputFormat.

### priceeye.customer_delivery
- Approx rows: 53
- Columns (18): id, customer, delivery_name, type, info_type, combine, size_is_post_compressed, max_result_file_size_mb, frequency, file_pattern, virus_check, encrypt ...
- Summary: customer_delivery table for customer config; ~53 rows; key columns: id, customer, delivery_name, type, info_type.

### priceeye.customer_imports
- Approx rows: 3
- Columns (8): customer, customerCollectionId, status, file_format, type, frequency, max_number_of_runs, last_updated
- Summary: customer_imports table for customer config; ~3 rows; key columns: customer, customerCollectionId, status, file_format, type.

### priceeye.customer_packaging
- Approx rows: 55
- Columns (14): id, customer, packaging_name, output_format, max_result_file_size_mb, filters, dedup, enriched_fields_used, hooks, currency, preferred_currency_code, time_to_live ...
- Summary: customer_packaging table for customer config; ~55 rows; key columns: id, customer, packaging_name, output_format, max_result_file_size_mb.

### priceeye.customer_reference_groups
- Approx rows: 0
- Columns (5): customer, reference_group, include_sql_pattern, exclude_sql_pattern, last_updated
- Summary: customer_reference_groups table for customer config; ~0 rows; key columns: customer, reference_group, include_sql_pattern, exclude_sql_pattern, last_updated.

### priceeye.dashboard_details
- Approx rows: 21
- Columns (4): customer, type, dashboard_id, last_updated
- Summary: dashboard_details table for customer config; ~21 rows; key columns: customer, type, dashboard_id, last_updated.

## Ingestion Requests

### priceeye.batches
- Approx rows: 293
- Columns (12): id, provider_code, site_code, crawl_timestamp, trip_type, customers, output_s3_uri, success_count, fail_count, no_result_count, status, last_updated
- Summary: batches table for ingestion requests; ~293 rows; key columns: id, provider_code, site_code, crawl_timestamp, trip_type.

### priceeye.import_rate
- Approx rows: 49
- Columns (9): provider_code, site_code, hour, rate, override_flag, override_ttl, override_timestamp, override_by, last_updated
- Summary: import_rate table for ingestion requests; ~49 rows; key columns: provider_code, site_code, hour, rate, override_flag.

### priceeye.input
- Approx rows: 212
- Columns (14): id, name, customer, status, start_date, start_time, end_date, end_time, row_count, run_count, max_run_count, reject_reason ...
- Summary: input table for ingestion requests; ~212 rows; key columns: id, name, customer, status, start_date.

### priceeye.input_archive
- Approx rows: 685
- Columns (14): id, name, customer, status, start_date, start_time, end_date, end_time, row_count, run_count, max_run_count, reject_reason ...
- Summary: input_archive table for ingestion requests; ~685 rows; key columns: id, name, customer, status, start_date.

### priceeye.input_criteria
- Approx rows: 6
- Columns (5): id, attribute, operator, value, transformation_id
- Summary: input_criteria table for ingestion requests; ~6 rows; key columns: id, attribute, operator, value, transformation_id.

### priceeye.input_details
- Approx rows: 272
- Columns (3): input_id, s3_file_location, last_updated
- Summary: input_details table for ingestion requests; ~272 rows; key columns: input_id, s3_file_location, last_updated.

### priceeye.input_request
- Approx rows: 209982
- Columns (24): id, input_id, customer, customer_site_code, customer_site_name, pos, carrier_codes, connection_airport_codes, max_stops, cabin, origin_airport_code, destination_airport_code ...
- Summary: input_request table for ingestion requests; ~209982 rows; key columns: id, input_id, customer, customer_site_code, customer_site_name.

### priceeye.input_request_old
- Approx rows: 211420
- Columns (24): id, input_id, customer, customer_site_code, customer_site_name, pos, carrier_codes, connection_airport_codes, max_stops, cabin, origin_airport_code, destination_airport_code ...
- Summary: input_request_old table for ingestion requests; ~211420 rows; key columns: id, input_id, customer, customer_site_code, customer_site_name.

### priceeye.poc_requests
- Approx rows: 27
- Columns (13): customer, status, activation_date, expiration_date, duration_days, extension_count, notes, submitted_by, contact_customer, cancelled_at, last_extended_at, created_on ...
- Summary: poc_requests table for ingestion requests; ~27 rows; key columns: customer, status, activation_date, expiration_date, duration_days.

### priceeye.request_priority
- Approx rows: 8
- Columns (9): customer, customerCollectionId, collectionFrequency, collectionHint, requestApStart, requestApEnd, priority, last_updated_by, last_updated
- Summary: request_priority table for ingestion requests; ~8 rows; key columns: customer, customerCollectionId, collectionFrequency, collectionHint, requestApStart.

## Operations Misc

### priceeye.airline_joint_business
- Approx rows: 63
- Columns (2): marketing_airline, joint_business_carrier
- Summary: airline_joint_business table for operations misc; ~63 rows; key columns: marketing_airline, joint_business_carrier.

### priceeye.audit_generator_config
- Approx rows: 3
- Columns (3): requestor, lastHour, lastOffset
- Summary: audit_generator_config table for operations misc; ~3 rows; key columns: requestor, lastHour, lastOffset.

### priceeye.blacklist_market_summary_old
- Approx rows: 24
- Columns (5): provider_code, site_code, origin_airport_code, destination_airport_code, last_updated
- Summary: blacklist_market_summary_old table for operations misc; ~24 rows; key columns: provider_code, site_code, origin_airport_code, destination_airport_code, last_updated.

### priceeye.cabin
- Approx rows: 9
- Columns (6): code, name, cabinType, validInput, hierarchy, lastUpdated
- Summary: cabin table for operations misc; ~9 rows; key columns: code, name, cabinType, validInput, hierarchy.

### priceeye.cache_rate
- Approx rows: 121
- Columns (12): provider_code, site_code, hour, cache_count, total, cachepct, cacheSource, override_flag, override_ttl, override_timestamp, override_by, last_updated
- Summary: cache_rate table for operations misc; ~121 rows; key columns: provider_code, site_code, hour, cache_count, total.

### priceeye.capability_definition
- Approx rows: 21
- Columns (6): capability_field, data_type, description, allowed_values, is_active, last_updated
- Summary: capability_definition table for operations misc; ~21 rows; key columns: capability_field, data_type, description, allowed_values, is_active.

### priceeye.confluence_page_map
- Approx rows: 4
- Columns (4): release_name, confluence_page_id, last_fetched, last_known_version
- Summary: confluence_page_map table for operations misc; ~4 rows; key columns: release_name, confluence_page_id, last_fetched, last_known_version.

### priceeye.contact_preferences
- Approx rows: 8
- Columns (5): user_id, notification_category, in_app_enabled, email_enabled, last_updated
- Summary: contact_preferences table for operations misc; ~8 rows; key columns: user_id, notification_category, in_app_enabled, email_enabled, last_updated.

### priceeye.contacts
- Approx rows: 97
- Columns (6): user_id, email, customer, is_admin, is_active, last_updated
- Summary: contacts table for operations misc; ~97 rows; key columns: user_id, email, customer, is_admin, is_active.

### priceeye.ecs_launcher_config
- Approx rows: 33
- Columns (5): queueName, taskDefinition, scope, arguments, queueDivisor
- Summary: ecs_launcher_config table for operations misc; ~33 rows; key columns: queueName, taskDefinition, scope, arguments, queueDivisor.

### priceeye.enrichment_sites
- Approx rows: 304
- Columns (7): provider_code, site_code, enrichment_type, enrichment_option, source_provider_code, source_site_code, last_updated
- Summary: enrichment_sites table for operations misc; ~304 rows; key columns: provider_code, site_code, enrichment_type, enrichment_option, source_provider_code.

### priceeye.error_mapping
- Approx rows: 98
- Columns (6): error_regex, error_sql, issue_source, issue_reason, last_updated_by, last_updated
- Summary: error_mapping table for operations misc; ~98 rows; key columns: error_regex, error_sql, issue_source, issue_reason, last_updated_by.

### priceeye.file_formats
- Approx rows: 14
- Columns (5): name, specification, industry_vertical, last_updated, last_updated_by
- Summary: file_formats table for operations misc; ~14 rows; key columns: name, specification, industry_vertical, last_updated, last_updated_by.

### priceeye.gds_surcharges
- Approx rows: 11
- Columns (6): carrier_code, geography, surcharge_type, currency_code, amount, applicability
- Summary: gds_surcharges table for operations misc; ~11 rows; key columns: carrier_code, geography, surcharge_type, currency_code, amount.

### priceeye.market_date_blacklist_old
- Approx rows: 284503
- Columns (16): sales_date, provider_code, site_code, pos, carrier_codes, connection_airport_codes, max_stops, cabin, origin_airport_code, destination_airport_code, depart_date, return_date ...
- Summary: market_date_blacklist_old table for operations misc; ~284503 rows; key columns: sales_date, provider_code, site_code, pos, carrier_codes.

### priceeye.output_field_validation
- Approx rows: 163
- Columns (5): customer, field_name, length, action, last_updated
- Summary: output_field_validation table for operations misc; ~163 rows; key columns: customer, field_name, length, action, last_updated.

### priceeye.output_file_formats
- Approx rows: 37
- Columns (16): name, specification, output_type, csv_field_separator, csv_field_quote_character, csv_date_format, csv_time_format, csv_date_time_format, csv_currency_format, compression_codec_name, field_value_separator, csv_write_header ...
- Summary: output_file_formats table for operations misc; ~37 rows; key columns: name, specification, output_type, csv_field_separator, csv_field_quote_character.

### priceeye.pending_input_actions
- Approx rows: 4
- Columns (3): pending_input_id, retiring_input_id, last_updated
- Summary: pending_input_actions table for operations misc; ~4 rows; key columns: pending_input_id, retiring_input_id, last_updated.

### priceeye.publish_raw_search_exclude
- Approx rows: 5
- Columns (2): provider_code, site_code
- Summary: publish_raw_search_exclude table for operations misc; ~5 rows; key columns: provider_code, site_code.

### priceeye.redis_cache_mapping
- Approx rows: 2
- Columns (4): provider_code, site_code, cache_source, last_updated
- Summary: redis_cache_mapping table for operations misc; ~2 rows; key columns: provider_code, site_code, cache_source, last_updated.

### priceeye.reference_swaps
- Approx rows: 11
- Columns (3): customer, incoming_reference, outgoing_reference
- Summary: reference_swaps table for operations misc; ~11 rows; key columns: customer, incoming_reference, outgoing_reference.

### priceeye.rental_agencies
- Approx rows: 226
- Columns (2): agency, shorthand
- Summary: rental_agencies table for operations misc; ~226 rows; key columns: agency, shorthand.

### priceeye.retry_rate
- Approx rows: 116
- Columns (12): provider_code, site_code, hour, retrycount, total, retrypct, retryprovidersite, override_flag, override_ttl, override_timestamp, override_by, last_updated
- Summary: retry_rate table for operations misc; ~116 rows; key columns: provider_code, site_code, hour, retrycount, total.

### priceeye.retry_substitution
- Approx rows: 59
- Columns (8): customer, original_provider_code, original_site_code, substitute_provider_code, substitute_site_code, start_time, end_time, last_updated
- Summary: retry_substitution table for operations misc; ~59 rows; key columns: customer, original_provider_code, original_site_code, substitute_provider_code, substitute_site_code.

### priceeye.system_configuration
- Approx rows: 0
- Columns (3): name, value, last_updated
- Summary: system_configuration table for operations misc; ~0 rows; key columns: name, value, last_updated.

### priceeye.transformation_rules
- Approx rows: 19
- Columns (8): id, customer, description, attribute_name, transform_action, transform_value, status, last_updated_by
- Summary: transformation_rules table for operations misc; ~19 rows; key columns: id, customer, description, attribute_name, transform_action.

### priceeye.valid_substitution_sites
- Approx rows: 84
- Columns (3): provider_code, site_code, last_updated
- Summary: valid_substitution_sites table for operations misc; ~84 rows; key columns: provider_code, site_code, last_updated.

## Packaging

### priceeye.last_packaging_run
- Approx rows: 62
- Columns (3): customer, reference_group, last_successful_cache_timestamp
- Summary: last_packaging_run table for packaging; ~62 rows; key columns: customer, reference_group, last_successful_cache_timestamp.

### priceeye.packager_scheduler_config
- Approx rows: 0
- Columns (4): lastCacheLoaderAuditId, lastAuditId, lastDropdeadDate, lastDropdeadTime
- Summary: packager_scheduler_config table for packaging; ~0 rows; key columns: lastCacheLoaderAuditId, lastAuditId, lastDropdeadDate, lastDropdeadTime.

### priceeye.packaging_dedup_keys
- Approx rows: 2
- Columns (3): name, specification, last_updated
- Summary: packaging_dedup_keys table for packaging; ~2 rows; key columns: name, specification, last_updated.

## Provider Integration

### priceeye.provider
- Approx rows: 32
- Columns (10): provider_code, provider_name, type, blacklist_days, max_connections, hostname, queue_name, schedule_adjustment, active, last_updated
- Summary: provider table for provider integration; ~32 rows; key columns: provider_code, provider_name, type, blacklist_days, max_connections.

### priceeye.provider_cabin_mapping
- Approx rows: 162
- Columns (7): provider_code, site_code, system_cabin, provider_site_cabin, forward_mapping, reverse_to_system, lastUpdated
- Summary: provider_cabin_mapping table for provider integration; ~162 rows; key columns: provider_code, site_code, system_cabin, provider_site_cabin, forward_mapping.

### priceeye.provider_dummy_config
- Approx rows: 21
- Columns (10): id, status, error_message, number_of_itineraries, delay_response, match_request_exactly, require_brand_enrichment, require_price_enrichment, require_flight_details, require_tax_enrichment
- Summary: provider_dummy_config table for provider integration; ~21 rows; key columns: id, status, error_message, number_of_itineraries, delay_response.

### priceeye.provider_error_messages
- Approx rows: 0
- Columns (2): providerCode, errorMessage
- Summary: provider_error_messages table for provider integration; ~0 rows; key columns: providerCode, errorMessage.

### priceeye.provider_pos_sitemap
- Approx rows: 99
- Columns (5): provider_code, site_code, pos, provider_site_code, is_default_site_code
- Summary: provider_pos_sitemap table for provider integration; ~99 rows; key columns: provider_code, site_code, pos, provider_site_code, is_default_site_code.

### priceeye.provider_ss_config
- Approx rows: 292
- Columns (2): site_code, agent_ids
- Summary: provider_ss_config table for provider integration; ~292 rows; key columns: site_code, agent_ids.

### priceeye.provider_ts_pccs
- Approx rows: 32
- Columns (3): pos, pcc, accessGroup
- Summary: provider_ts_pccs table for provider integration; ~32 rows; key columns: pos, pcc, accessGroup.

### priceeye.site
- Approx rows: 707
- Columns (9): provider_code, site_code, site_name, pos, type, provider_properties, retry_count, status, last_updated
- Summary: site table for provider integration; ~707 rows; key columns: provider_code, site_code, site_name, pos, type.

### priceeye.token
- Approx rows: 4
- Columns (4): provider, pcc, token, lastUpdated
- Summary: token table for provider integration; ~4 rows; key columns: provider, pcc, token, lastUpdated.

### priceeye.transaction_rates
- Approx rows: 55
- Columns (3): provider, hour, tps
- Summary: transaction_rates table for provider integration; ~55 rows; key columns: provider, hour, tps.

## Reference Data

### priceeye.atlas_carriers
- Approx rows: 127
- Columns (2): carrier, last_updated
- Summary: atlas_carriers table for reference data; ~127 rows; key columns: carrier, last_updated.

### priceeye.atlas_routes
- Approx rows: 189158
- Columns (7): origin, destination, carrier, isDirect, scheduleStart, scheduleEnd, last_updated
- Summary: atlas_routes table for reference data; ~189158 rows; key columns: origin, destination, carrier, isDirect, scheduleStart.

### priceeye.atp_brands
- Approx rows: 419235
- Columns (4): carrier_code, rule_tariff_code, fare_basis_code, brand_name
- Summary: atp_brands table for reference data; ~419235 rows; key columns: carrier_code, rule_tariff_code, fare_basis_code, brand_name.

### priceeye.excluded_brands
- Approx rows: 0
- Columns (4): provider_code, site_code, brand, last_updated
- Summary: excluded_brands table for reference data; ~0 rows; key columns: provider_code, site_code, brand, last_updated.

### priceeye.ojt_hotel_locations
- Approx rows: 722
- Columns (7): name, location_code, location_type, country_code, state_code, airport_code, last_updated
- Summary: ojt_hotel_locations table for reference data; ~722 rows; key columns: name, location_code, location_type, country_code, state_code.

### priceeye.sabre_pccs
- Approx rows: 5
- Columns (4): pos, site_code, pcc, last_updated
- Summary: sabre_pccs table for reference data; ~5 rows; key columns: pos, site_code, pcc, last_updated.

### priceeye.sk_brands
- Approx rows: 17
- Columns (3): pattern, brand, lastUpdated
- Summary: sk_brands table for reference data; ~17 rows; key columns: pattern, brand, lastUpdated.

### priceeye.travelport_pccs
- Approx rows: 160
- Columns (6): countrycode, pcc, branch, citycode, provider, validated
- Summary: travelport_pccs table for reference data; ~160 rows; key columns: countrycode, pcc, branch, citycode, provider.

## Scheduling Routing

### priceeye.auto_schedule_last_trigger
- Approx rows: 0
- Columns (3): app_name, trigger_id, last_updated
- Summary: auto_schedule_last_trigger table for scheduling routing; ~0 rows; key columns: app_name, trigger_id, last_updated.

### priceeye.auto_schedule_trigger
- Approx rows: 94
- Columns (14): id, provider_code, site_code, triggered_by, input_location, input_customer, input_format, input_collection_id, input_start_date, input_start_time, input_end_date, input_end_time ...
- Summary: auto_schedule_trigger table for scheduling routing; ~94 rows; key columns: id, provider_code, site_code, triggered_by, input_location.

### priceeye.customer_site_code
- Approx rows: 728
- Columns (8): customer_site_code, description, type, industry_vertical, is_points, visible_to_customer, last_updated_by, last_updated
- Summary: customer_site_code table for scheduling routing; ~728 rows; key columns: customer_site_code, description, type, industry_vertical, is_points.

### priceeye.runtime_archive
- Approx rows: 118
- Columns (17): customer, customer_site_code, provider_code, reference, pos, cabin, trip_type, ap_min, ap_max, day_of_week, schedule_hour, crawl_run_hour ...
- Summary: runtime_archive table for scheduling routing; ~118 rows; key columns: customer, customer_site_code, provider_code, reference, pos.

### priceeye.site_capabilities
- Approx rows: 593
- Columns (9): provider_code, site_code, capability, capability_value, detection_method, notes, status, last_updated_by, last_updated
- Summary: site_capabilities table for scheduling routing; ~593 rows; key columns: provider_code, site_code, capability, capability_value, detection_method.

### priceeye.site_carriers
- Approx rows: 497
- Columns (4): provider_code, site_code, carrier_code, last_updated
- Summary: site_carriers table for scheduling routing; ~497 rows; key columns: provider_code, site_code, carrier_code, last_updated.

### priceeye.site_dictionary_requests
- Approx rows: 495
- Columns (22): provider_code, site_code, capability_type, pos, carrier_codes, connection_airport_codes, max_stops, cabin, origin_airport_code, destination_airport_code, depart_date, trip_type ...
- Summary: site_dictionary_requests table for scheduling routing; ~495 rows; key columns: provider_code, site_code, capability_type, pos, carrier_codes.

### priceeye.site_hierarchy
- Approx rows: 1131
- Columns (13): customer, customer_site_code, priority, provider_code, site_code, tripType, overrideSiteCode, overrideCarrier, supplemental, siteCategory, qualityScore, relevancyScore ...
- Summary: site_hierarchy table for scheduling routing; ~1131 rows; key columns: customer, customer_site_code, priority, provider_code, site_code.

### priceeye.site_hierarchy_archive
- Approx rows: 2740
- Columns (13): customer, customer_site_code, priority, provider_code, site_code, tripType, overrideSiteCode, overrideCarrier, supplemental, siteCategory, qualityScore, relevancyScore ...
- Summary: site_hierarchy_archive table for scheduling routing; ~2740 rows; key columns: customer, customer_site_code, priority, provider_code, site_code.

### priceeye.site_mapping_archive
- Approx rows: 521
- Columns (13): customer, customer_site_code, original_pos, min_ap, max_ap, cabin, trip_type, provider_code, site_code, site_category, new_pos, new_carrier_codes ...
- Summary: site_mapping_archive table for scheduling routing; ~521 rows; key columns: customer, customer_site_code, original_pos, min_ap, max_ap.

### priceeye.site_metrics
- Approx rows: 12143
- Columns (11): provider_code, site_code, hour_of_day, rate, measure, delay, override_flag, override_ttl, override_timestamp, override_by, last_updated
- Summary: site_metrics table for scheduling routing; ~12143 rows; key columns: provider_code, site_code, hour_of_day, rate, measure.

### priceeye.site_metrics_archive
- Approx rows: 13327
- Columns (11): provider_code, site_code, hour_of_day, rate, measure, delay, override_flag, override_ttl, override_timestamp, override_by, last_updated
- Summary: site_metrics_archive table for scheduling routing; ~13327 rows; key columns: provider_code, site_code, hour_of_day, rate, measure.

## Vacation Compare

### priceeye.ql2_vacation_site
- Approx rows: 13
- Columns (2): site_code, ql2_site_code
- Summary: ql2_vacation_site table for vacation compare; ~13 rows; key columns: site_code, ql2_site_code.

### priceeye.vacation_compare
- Approx rows: 1
- Columns (3): compareId, status, last_modified
- Summary: vacation_compare table for vacation compare; ~1 rows; key columns: compareId, status, last_modified.

### priceeye.vacation_compare_delivery
- Approx rows: 2
- Columns (6): compareId, deliveryId, type, file_name_pattern, send_exception_file, last_modified
- Summary: vacation_compare_delivery table for vacation compare; ~2 rows; key columns: compareId, deliveryId, type, file_name_pattern, send_exception_file.

### priceeye.vacation_compare_input
- Approx rows: 2
- Columns (4): compareId, customer, customerCollectionId, last_modified
- Summary: vacation_compare_input table for vacation compare; ~2 rows; key columns: compareId, customer, customerCollectionId, last_modified.

### priceeye.vacation_compare_input_old
- Approx rows: 56
- Columns (4): deliveryId, customer, customerCollectionId, last_modified
- Summary: vacation_compare_input_old table for vacation compare; ~56 rows; key columns: deliveryId, customer, customerCollectionId, last_modified.

### priceeye.vacation_compare_old
- Approx rows: 49
- Columns (6): deliveryId, type, file_name_pattern, status, send_exception_file, last_modified
- Summary: vacation_compare_old table for vacation compare; ~49 rows; key columns: deliveryId, type, file_name_pattern, status, send_exception_file.
