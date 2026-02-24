# PriceEye Data Model KB

This KB section captures live-profiled table summaries from `3VDEV` for PriceEye and PriceEye Analytics.

## Included Documents
- `docs/priceeye_data_model/database_connections.md`
- `docs/priceeye_data_model/redshift_table_summaries.md`
- `docs/priceeye_data_model/mysql_table_summaries.md`
- `docs/priceeye_data_model/mysql_discovery_notes.md`

## Scope
- Source systems: Redshift (`prod.*`) and MySQL (`priceeye`, `analytics`).
- Summaries are built from live schema metadata and key-column inspection.
- Intended for entity resolution, table selection, and cross-source join planning.
