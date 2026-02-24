# PriceEye Data Model KB

This KB section captures live-profiled table summaries from `3VDEV` for PriceEye and PriceEye Analytics.

## Included Documents
- `docs/priceeye_data_model/serverless_all_tables_inventory.md`
- `docs/priceeye_data_model/analytics_serverless_partition_notes.md`
- `docs/priceeye_data_model/analytics_serverless_reader_discovery.md`
- `docs/priceeye_data_model/analytics_reader_partition_notes.md`
- `docs/priceeye_data_model/analytics_reader_discovery.md`
- `docs/priceeye_data_model/redshift_partition_audit.md`
- `docs/priceeye_data_model/redshift_core_reader_discovery.md`
- `docs/priceeye_data_model/database_connections.md`
- `docs/priceeye_data_model/redshift_table_summaries.md`
- `docs/priceeye_data_model/mysql_table_summaries.md`
- `docs/priceeye_data_model/mysql_discovery_notes.md`

## Scope
- Source systems: Redshift (`prod.*`) and MySQL (`priceeye`, `analytics`).
- Summaries are built from live schema metadata and key-column inspection.
- Intended for entity resolution, table selection, and cross-source join planning.
