# Analytics Reader Partition Notes

- `analytics_mysql` and `federated_analytics` tables are federated/MySQL-style and expose no partition keys in Redshift catalog.
- `pe_analytics_audits.analytics_audits` is S3-backed and partitioned by `sales_date`.
- Query policy: enforce customer predicates for federated lookup/config tables when relevant, and enforce `sales_date` on audit table access.

## analytics_mysql

- `analytics_mysql.alerts_schedule`: partitions = none
- `analytics_mysql.anomalies_direction_score`: partitions = none
- `analytics_mysql.anomalies_impact_score_weights`: partitions = none
- `analytics_mysql.cabin_group`: partitions = none
- `analytics_mysql.carrier_group`: partitions = none
- `analytics_mysql.city_code_override`: partitions = none
- `analytics_mysql.date_range`: partitions = none
- `analytics_mysql.demo_carrier_substitutions`: partitions = none
- `analytics_mysql.geography`: partitions = none
- `analytics_mysql.geography_entry`: partitions = none
- `analytics_mysql.region`: partitions = none
- `analytics_mysql.segment`: partitions = none

## federated_analytics

- `federated_analytics.alerts_schedule`: partitions = none
- `federated_analytics.anomalies_direction_score`: partitions = none
- `federated_analytics.anomalies_impact_score_weights`: partitions = none
- `federated_analytics.cabin_group`: partitions = none
- `federated_analytics.carrier_group`: partitions = none
- `federated_analytics.city_code_override`: partitions = none
- `federated_analytics.date_range`: partitions = none
- `federated_analytics.demo_carrier_substitutions`: partitions = none
- `federated_analytics.geography`: partitions = none
- `federated_analytics.geography_entry`: partitions = none
- `federated_analytics.region`: partitions = none
- `federated_analytics.segment`: partitions = none

## pe_analytics_audits

- `pe_analytics_audits.alerts_audit_v1`: partitions = sales_date
