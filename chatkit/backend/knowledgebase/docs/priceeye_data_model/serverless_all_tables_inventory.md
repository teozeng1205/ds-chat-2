# Serverless Reader Full Table Inventory

Reader profile: `database-analytics-redshift-serverless-reader.properties`.

## Coverage Status
- Scope: all discoverable non-system tables from this reader (`svv_external_tables` + non-system `pg_tables`).
- External tables discovered: 545
- Non-system pg tables discovered: 47
- Total discovered: 592
- New KB specs added in this pass: 521
- Coverage check result: 100% of discovered serverless-reader tables are represented in KB.

## External Schemas
- `adf`: 1
- `analytics`: 19
- `analytics_mysql`: 12
- `billing`: 3
- `brands_enrichment`: 1
- `channel_comparison`: 4
- `common_output`: 1
- `federated_analytics`: 12
- `federated_metadata`: 115
- `federated_priceeye`: 89
- `federated_replication`: 4
- `federated_sales_poc`: 15
- `metadata_mysql`: 115
- `monitoring`: 25
- `pe_analytics_audits`: 1
- `priceeye_audits`: 16
- `priceeye_mysql`: 89
- `replication_mysql`: 4
- `sales_poc`: 3
- `sales_poc_mysql`: 15
- `webfares`: 1

## Non-System PG Schemas
- `garcemont`: 1
- `grayson`: 2
- `pg_internal`: 1
- `pg_s3`: 1
- `quicksight_datasets`: 42

## KB Catalog Files
- `knowledgebase/tables/priceeye_serverless_all_tables.yaml` (comprehensive all-table catalog from this pass).
- Existing table catalogs remain in place for curated/query-priority use cases.
