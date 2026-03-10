# PriceEye AWS Infrastructure Reference

*Last refreshed: 2026-03-10*
*Environments: 3VDEV (590183652635) | 3VPROD (539247469204)*
*Regenerate: `python scripts/refresh_aws_infra.py` (requires 3VDEV creds)*

---

## Quick Reference

### Key Step Functions
| SFN Name | Pipeline | Schedule |
|---|---|---|
| `AggregationAnomaliesStepFunction` | ds-priceeye-analytics anomaly scoring | Hourly, triggered by EventBridge |
| `ProviderCentricStepFunction` | ds-internal-monitoring provider view | Hourly :10 UTC |
| `CustomerCentricStepFunction` | ds-internal-monitoring customer view | Hourly :30 UTC |
| `MidtDailyStepFunction` | PAX MIDT daily booking summary | Daily |
| `DS-Analytics-EventDriven-Jobs` | Analytics pipeline (competitive pos, market/seg) | Event-driven per customer |
| `DS-Sales-POC-Jobs` | Sales POC market data pipeline | Daily |
| `unload-monitoring-step-function` | Monitoring dedup + combined_audit pipeline | Hourly |
| `taxregression-step-function` | Tax regression coefficients (Tuesdays) | Weekly |
| `site-metrics-stepfunction` | Site capacity/cache/retry/import metrics | Daily |
| `dropdead-stepfunction` | Drop-dead deadline enforcement | Daily |

### Key Lambda Functions
| Function Name | Domain | Trigger |
|---|---|---|
| `anomalies_process_customer_v2` | Market/segment anomaly scoring | DS-Analytics-EventDriven-Jobs SFN |
| `alerts` | Anomaly alert publishing | EventBridge after anomaly SFN |
| `partitioncreator` | Glue partition creation for new S3 data | S3 PutObject events |
| `dropdead-detector` | Detects stalled pipelines past deadline | EventBridge scheduled |
| `persist-audit-data-redshift` | Writes audit data to Redshift Serverless | Kinesis stream |
| `persist-audit-data-mysql` | Writes audit data to Aurora MySQL | Kinesis stream |
| `collection-optimizer` | Delta SWIA → collection plan optimization | EventBridge |
| `site-metrics` | Site TPS/cache/retry metric aggregation | SFN task |
| `capacity-metrics` | Provider capacity (IQR-filtered TPH) | SFN task |

### Glue Databases
| Database | Domain |
|---|---|
| `analytics_db` | Market/segment anomalies, competitive position, scoring |
| `monitoring_db` | Deduped audits, combined_audit, provider/customer views |
| `billing_db` | Customer daily billing metrics |
| `collection_optimizer_db` | Delta SWIA input, ingest TTL |
| `site-metrics-db` | TPS, capacity, cache, retry, import metrics |
| `tax_reg_db` | Tax regression output |
| `priceeye_audits_db` | Raw provider audit tables (long-term S3-backed) |
| `common_output_db` | Normalized price observations (DCO) |
| `data_lakes_db` | Daily itinerary data, OAG, MIDT |
| `yqyr_cache_db` | YQYR tax prediction cache |

---

## 3VDEV — Development (590183652635)

### Redshift Serverless Workgroups

| Workgroup | Datasource Name | Endpoint |
|---|---|---|
| `analytics-3vdev` | `redshift_analytics` | `analytics-3vdev.590183652635.us-east-1.redshift-serverless.amazonaws.com:5439` |
| `redshift-3vdev` | `redshift_core` | `redshift-3vdev.590183652635.us-east-1.redshift-serverless.amazonaws.com:5439` |
| `monitoringgrp` | `monitoring` | `monitoringgrp.590183652635.us-east-1.redshift-serverless.amazonaws.com:5439` |

**IAM cross-account access:** `redshift-3vdev` carries IAM role `3VDEV-Access-3VPROD`, which grants read access to production Redshift Serverless in the 3VPROD account (539247469204). This is why `execute_sql` on `redshift_core` returns prod data.

**Namespaces:** `analytics-3vdev` / `redshift-3vdev` / `monitoring-3vdev`

CLI:
```bash
aws redshift-serverless list-workgroups --query 'workgroups[*].{Name:workgroupName,Status:status,Endpoint:endpoint.address}'
aws redshift-serverless get-workgroup --workgroup-name analytics-3vdev
```

### Aurora MySQL Clusters

| Cluster ID | Environment | Purpose |
|---|---|---|
| `rdsdbcluster-atp-3victors-3vdev-use1-price-eye` | 3VDEV | PriceEye operational DB (AutoSchedule, sales_poc, taxregression) |

CLI:
```bash
aws rds describe-db-clusters --query 'DBClusters[*].{ID:DBClusterIdentifier,Status:Status,Endpoint:Endpoint}'
aws rds describe-events --source-type db-cluster --duration 60
```

---

### Step Functions (3VDEV — 42 total)

| Name | Purpose | Schedule / Trigger |
|---|---|---|
| `AggregationAnomaliesStepFunction` | Market + segment anomaly scoring (Python/ECS) | EventBridge, per-customer post-DCO |
| `ProviderCentricStepFunction` | Provider-centric view from combined_audit | Hourly :10 UTC |
| `CustomerCentricStepFunction` | Customer-centric view + customer_combined_audit_v2 | Hourly :30 UTC |
| `MidtDailyStepFunction` | PAX MIDT daily booking summary → analytics | Daily |
| `DS-Analytics-EventDriven-Jobs` | Orchestrates anomalies_process_customer_v2 Lambda | Event-driven |
| `DS-Sales-POC-Jobs` | Market data → sales_poc.input_request MySQL | Daily |
| `unload-monitoring-step-function` | Dedup → combined_audit 9-stage pipeline | Hourly |
| `taxregression-step-function` | Tax regression coefficients (Tuesday) | Weekly (Tuesday) |
| `site-metrics-stepfunction` | TPS / capacity / cache / retry / import metrics | Daily |
| `dropdead-stepfunction` | Drop-dead deadline enforcement | Daily |
| `competitive-position-stepfunction` | Competitive position from DCO | Daily |
| `daily-itins-stepfunction` | Daily representative itinerary v4 | Daily |
| `oag-score-stepfunction` | OAG flight data scoring | Daily |
| `revenue-score-stepfunction` | Revenue score from PAX + itins | Daily |
| `collection-optimizer-stepfunction` | Collection plan optimization | Daily |
| `yqyr-cache-stepfunction` | YQYR tax prediction cache | Daily |
| `alerts-stepfunction` | Alert publishing to EventBridge | Hourly |
| `partition-creator-stepfunction` | Glue partition registration | Event-driven |
| `ingest-ttl-stepfunction` | 25th-pct hours-between-changes per carrier | Daily |

CLI:
```bash
# List all SFNs
aws stepfunctions list-state-machines --query 'stateMachines[*].{Name:name,ARN:stateMachineArn}'

# Get full definition (replace NAME with actual SFN name)
aws stepfunctions describe-state-machine \
  --state-machine-arn arn:aws:states:us-east-1:590183652635:stateMachine:AggregationAnomaliesStepFunction

# Recent executions (last 10)
aws stepfunctions list-executions \
  --state-machine-arn arn:aws:states:us-east-1:590183652635:stateMachine:CustomerCentricStepFunction \
  --max-results 10

# Failed executions
aws stepfunctions list-executions \
  --state-machine-arn arn:aws:states:us-east-1:590183652635:stateMachine:AggregationAnomaliesStepFunction \
  --status-filter FAILED --max-results 10

# Describe a specific execution
aws stepfunctions describe-execution --execution-arn ARN

# Step-by-step history
aws stepfunctions get-execution-history --execution-arn ARN
```

---

### Lambda Functions (3VDEV — 30+ total)

| Function | Runtime | Timeout | Domain |
|---|---|---|---|
| `anomalies_process_customer_v2` | python3.12 | 900s | Anomaly scoring (market + segment) |
| `alerts` | python3.12 | 300s | Alert publishing → EventBridge |
| `partitioncreator` | python3.12 | 300s | Glue partition registration |
| `dropdead-detector` | python3.12 | 60s | Stalled pipeline detection |
| `persist-audit-data-redshift` | python3.12 | 300s | Kinesis → Redshift Serverless |
| `persist-audit-data-mysql` | python3.12 | 300s | Kinesis → Aurora MySQL |
| `collection-optimizer` | python3.12 | 900s | Delta SWIA → scheduling plan |
| `site-metrics` | python3.12 | 600s | Site TPS/import metrics |
| `capacity-metrics` | python3.12 | 600s | Provider capacity (IQR-filtered) |
| `yqyr-cache-lambda` | python3.12 | 900s | YQYR cache unload |
| `tax-regression` | python3.12 | 900s | Tax regression weekly |
| `midt-daily` | python3.12 | 900s | PAX MIDT daily |
| `revenue-score` | python3.12 | 600s | Revenue score |
| `oag-score` | python3.12 | 600s | OAG score |
| `competitive-position` | python3.12 | 900s | Competitive position |
| `daily-itins` | python3.12 | 900s | Daily representative itineraries |

Key env vars on `anomalies_process_customer_v2`:
- `REDSHIFT_WORKGROUP`: `analytics-3vdev`
- `REDSHIFT_DATABASE`: `dev`
- `OUTPUT_S3_BUCKET`: `s3-atp-3victors-3vprod-use1-anomaly-datasets`
- `ENVIRONMENT`: `prod` (reads from 3VPROD S3, writes to 3VDEV Redshift)

CLI:
```bash
# List all
aws lambda list-functions --query 'Functions[*].{Name:FunctionName,Runtime:Runtime,Updated:LastModified}'

# Config (env vars, timeout, memory)
aws lambda get-function-configuration --function-name anomalies_process_customer_v2

# Policy (EventBridge/SFN triggers)
aws lambda get-policy --function-name anomalies_process_customer_v2

# Log group
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/anomalies
```

---

### Glue Databases & Tables (3VDEV — 58+ databases)

| Database | Key Tables |
|---|---|
| `analytics_db` | `market_level_anomalies_v3`, `market_level_anomalies_v4`, `segment_level_anomalies_v2`, `competitive_position_v2`, `market_level_analysis_v2`, `segment_level_analysis_v2`, `pax_midt`, `daily_itins_prices_v2`, `oag_score_v2`, `revenue_score_v1` |
| `monitoring_db` | `combined_audit`, `provider_combined_audit`, `customer_combined_audit_v2`, `response_dupes`, `deduped_provider_request_audit_detail`, `deduped_provider_response_audit`, `deduped_delivery_audit`, `deduped_packager_audit`, `deduped_retry_audit` |
| `billing_db` | `customer_daily_requests_v1`, `customer_daily_requests_v2`, `customer_daily_requests_v3` |
| `collection_optimizer_db` | `delta_swia_input_v1`, `ingest_ttl_v1` |
| `site-metrics-db` | `capacity_final`, `cache_metrics_v1`, `retry_metrics_v1`, `import_metrics_v1`, `provider_tps_validate_v1`, `provider_tps_by_intervals_v1` |
| `tax_reg_db` | `tax_reg_output_v1`, `tax_reg_output_com_v1` |
| `priceeye_audits_db` | `provider_request_audit`, `provider_response_audit`, `packager_audit`, `delivery_audit`, `enrichment_audit`, `retry_audit`, `cache_audit` |
| `common_output_db` | `common_output_format`, `derived_common_output` |
| `data_lakes_db` | `daily_representative_itinerary_v4`, `midt_daily_booking_summary` |
| `yqyr_cache_db` | `yqyr_cache_v1`, `yqyr_predictions` |

CLI:
```bash
# List all databases
aws glue get-databases --query 'DatabaseList[*].Name'

# List tables in a database
aws glue get-tables --database-name analytics_db --query 'TableList[*].Name'

# Inspect table schema + S3 location
aws glue get-table --database-name analytics_db --name market_level_anomalies_v4

# Latest partitions (check freshness)
aws glue get-partitions --database-name analytics_db --table-name market_level_anomalies_v4 \
  --max-results 3 --query 'Partitions[-3:].{Values:Values,Location:StorageDescriptor.Location}'

# Partitions for monitoring tables
aws glue get-partitions --database-name monitoring_db --table-name provider_combined_audit \
  --max-results 3

# Partitions for common output
aws glue get-partitions --database-name common_output_db --table-name common_output_format \
  --max-results 5
```

---

### EventBridge Rules (3VDEV — 30 total)

| Rule Name | Schedule | Target |
|---|---|---|
| `RunLambdaHourly1MinAfterUTC` | `cron(1 * * * ? *)` | `partitioncreator` Lambda |
| `AnomaliesEventDrivenTrigger` | Event pattern: DCO complete | `DS-Analytics-EventDriven-Jobs` SFN |
| `ProviderCentricSchedule` | `cron(10 * * * ? *)` | `ProviderCentricStepFunction` SFN |
| `CustomerCentricSchedule` | `cron(30 * * * ? *)` | `CustomerCentricStepFunction` SFN |
| `MonitoringUnloadSchedule` | `cron(0 * * * ? *)` | `unload-monitoring-step-function` SFN |
| `DailyAnalyticsSchedule` | `cron(0 8 * * ? *)` | `DS-Analytics-EventDriven-Jobs` SFN |
| `SiteMetricsSchedule` | `cron(0 10 * * ? *)` | `site-metrics-stepfunction` SFN |
| `TaxRegressionSchedule` | `cron(0 6 ? * 3 *)` | `taxregression-step-function` SFN (Tuesdays) |
| `MidtDailySchedule` | `cron(0 12 * * ? *)` | `MidtDailyStepFunction` SFN |
| `DropDeadSchedule` | `cron(0 14 * * ? *)` | `dropdead-stepfunction` SFN |

CLI:
```bash
# List all rules with schedule
aws events list-rules --query 'Rules[*].{Name:Name,State:State,Schedule:ScheduleExpression,EventPattern:EventPattern}'

# Get rule detail
aws events describe-rule --name CustomerCentricSchedule

# List targets for a rule
aws events list-targets-by-rule --rule CustomerCentricSchedule

# Rules targeting a specific Lambda
aws events list-rule-names-by-target \
  --target-arn arn:aws:lambda:us-east-1:590183652635:function:anomalies_process_customer_v2
```

---

### CloudWatch Alarms (3VDEV)

Common alarm patterns:
- `3Victors-AnomaliesLambda-Errors` — `anomalies_process_customer_v2` error rate
- `3Victors-CustomerCentricSFN-Failures` — CustomerCentricStepFunction execution failures
- `3Victors-ProviderCentricSFN-Failures` — ProviderCentricStepFunction execution failures
- `3Victors-UnloadMonitoring-Failures` — unload-monitoring-step-function failures
- `3Victors-Provider{CODE}-Errors` — Per-provider error rate alarms (AA, UA, DL, etc.)
- `3Victors-Redshift-CPUUtilization-High` — Redshift Serverless CPU
- `3Victors-RDS-CPUUtilization-High` — Aurora MySQL CPU
- `3Victors-DropDead-*` — Drop-dead deadline missed alarms

CLI:
```bash
# Alarms in ALARM state (the most useful health check)
aws cloudwatch describe-alarms --state-value ALARM \
  --query 'MetricAlarms[*].{Name:AlarmName,State:StateValue,Reason:StateReason}'

# All alarms regardless of state
aws cloudwatch describe-alarms \
  --query 'MetricAlarms[*].{Name:AlarmName,State:StateValue,Updated:StateUpdatedTimestamp}'

# Alarms for a specific metric namespace
aws cloudwatch describe-alarms-for-metric \
  --namespace AWS/States --metric-name ExecutionsFailed
```

---

### CloudWatch Logs (3VDEV)

Lambda log groups: `/aws/lambda/{function_name}`
Glue job log groups: `/aws-glue/jobs/{job_name}`

```bash
# List Lambda log groups
aws logs describe-log-groups --log-group-name-prefix /aws/lambda \
  --query 'logGroups[*].{Name:logGroupName,RetentionDays:retentionInDays}'

# List Glue log groups
aws logs describe-log-groups --log-group-name-prefix /aws-glue

# Recent log streams for a Lambda
aws logs describe-log-streams \
  --log-group-name /aws/lambda/anomalies_process_customer_v2 \
  --order-by LastEventTime --descending --limit 5

# Search logs for ERROR in last 1 hour
aws logs filter-log-events \
  --log-group-name /aws/lambda/anomalies_process_customer_v2 \
  --start-time $(($(date +%s)-3600))000 \
  --filter-pattern "ERROR"
```

**CloudWatch Logs Insights (async — use start-query + get-query-results):**
```bash
# Start a Logs Insights query
QUERY_ID=$(aws logs start-query \
  --log-group-name /aws/lambda/anomalies_process_customer_v2 \
  --start-time $(($(date +%s)-3600)) \
  --end-time $(date +%s) \
  --query-string 'fields @timestamp, @message | filter @message like /ERROR/ | stats count() as errorCount' \
  --query 'queryId' --output text)

# Poll for results (repeat until status=Complete)
aws logs get-query-results --query-id "$QUERY_ID"
```

---

### S3 Buckets (PriceEye-relevant, 3VDEV)

Naming pattern: `s3-atp-3victors-{env}-use1-{purpose}`

| Bucket Suffix | Purpose |
|---|---|
| `pe-common-output` | Hourly price observations per customer (3VPROD → shared) |
| `priceeye-raw` | Raw SWIA Avro data (delta + estream) |
| `anomaly-datasets` | Market/segment anomaly Spark output |
| `derived-common-output` | DCO Parquet after normalization |
| `competitive-position` | Competitive position analysis output |
| `priceeye-data` | General PriceEye data artifacts |
| `collection-optimizer` | AutoSchedule plans, AS comparison CSVs |
| `ds-sales-poc` | Sales POC market data |
| `as-scheduled-comparison` | AutoSchedule vs actual comparison |
| `logs` | CloudTrail, ELB, VPC flow logs |

```bash
aws s3 ls  # List all ~180 buckets
aws s3 ls s3://s3-atp-3victors-3vprod-use1-pe-common-output/ --recursive | head -20
```

---

### Secrets Manager (3VDEV — names only)

```bash
aws secretsmanager list-secrets --query 'SecretList[*].{Name:Name,Description:Description}'
```

Common secret name patterns:
- `/priceeye/db/*` — MySQL connection credentials
- `/redshift/*` — Redshift connection details
- `/api-keys/*` — External API credentials
- `3vdev/*` — Environment-specific secrets

---

### SSM Parameter Store (3VDEV)

```bash
aws ssm describe-parameters --query 'Parameters[*].{Name:Name,Type:Type}'
aws ssm get-parameters-by-path --path /priceeye/ --recursive --with-decryption
```

---

## 3VPROD — Production (539247469204)

Switch to prod: `assume 3VPROD`
Switch back to dev: `assume 3VDEV`

**Note:** `execute_sql` always uses 3VDEV credentials (which have cross-account prod Redshift read via IAM role `3VDEV-Access-3VPROD`). For AWS CLI on prod resources (SFN, Lambda, CW alarms), you must `assume 3VPROD` first.

### Redshift Serverless (3VPROD)

| Workgroup | Purpose |
|---|---|
| `analytics-3vprod` | Production analytics queries |
| `redshift-3vprod` | Production core/monitoring queries |

```bash
# After assume 3VPROD:
aws redshift-serverless list-workgroups --query 'workgroups[*].{Name:workgroupName,Status:status}'
```

### Aurora MySQL Clusters (3VPROD)

| Cluster ID | Purpose |
|---|---|
| `rdsdbcluster-atp-3victors-3vprod-use1-price-eye` | Production PriceEye operational DB |
| `aurora-master` | Production master Aurora cluster |

### Step Functions (3VPROD — same names as 3VDEV)

All Step Functions have the same names in 3VPROD. Check executions in prod:
```bash
# After assume 3VPROD:
aws stepfunctions list-state-machines --query 'stateMachines[*].name'

# Failed executions in prod CustomerCentricStepFunction
aws stepfunctions list-executions \
  --state-machine-arn arn:aws:states:us-east-1:539247469204:stateMachine:CustomerCentricStepFunction \
  --status-filter FAILED --max-results 10
```

### CloudWatch Alarms (3VPROD)

```bash
# After assume 3VPROD:
aws cloudwatch describe-alarms --state-value ALARM \
  --query 'MetricAlarms[*].{Name:AlarmName,Reason:StateReason}'
```

### S3 Buckets (3VPROD)

Pattern: `s3-atp-3victors-3vprod-use1-{purpose}`
Key prod bucket: `s3-atp-3victors-3vprod-use1-pe-common-output`

---

## Pipeline → Table → Resource Cross-Reference

| Output Table | SFN | Lambda/Glue Job | Datasource | Glue DB |
|---|---|---|---|---|
| `prod.analytics.market_level_anomalies_v4` | `AggregationAnomaliesStepFunction` | `anomalies_process_customer_v2` | `redshift_analytics` | `analytics_db` |
| `prod.analytics.segment_level_anomalies_v2` | `AggregationAnomaliesStepFunction` | `anomalies_process_customer_v2` | `redshift_analytics` | `analytics_db` |
| `prod.analytics.competitive_position` | `DS-Analytics-EventDriven-Jobs` | `competitive-position` Lambda | `redshift_analytics` | `analytics_db` |
| `prod.analytics.pax_midt` | `MidtDailyStepFunction` | `midt-daily` Lambda | `redshift_analytics` | `data_lakes_db` |
| `prod.monitoring.combined_audit` | `unload-monitoring-step-function` | Glue dedup jobs | `redshift_core` | `monitoring_db` |
| `prod.monitoring.provider_combined_audit` | `ProviderCentricStepFunction` | Glue provider job | `redshift_core` | `monitoring_db` |
| `prod.monitoring.customer_combined_audit_v2` | `CustomerCentricStepFunction` | Glue customer job | `redshift_core` | `monitoring_db` |
| `prod.common_output.common_output_format` | priceeye-v2 (Java) | ECS task | `redshift_analytics` | `common_output_db` |
| `billing_db.customer_daily_requests_v3` | `CustomerCentricStepFunction` | Glue billing job | `redshift_core` | `billing_db` |
| `tax_reg_db.tax_reg_output_v1` | `taxregression-step-function` | `tax-regression` Lambda | `redshift_core` | `tax_reg_db` |
| `site_metrics.capacity_final` | `site-metrics-stepfunction` | `capacity-metrics` Lambda | `redshift_core` | `site-metrics-db` |
| `collection_optimizer.delta_swia_input_v1` | `collection-optimizer-stepfunction` | `collection-optimizer` Lambda | `redshift_core` | `collection_optimizer_db` |
| `yqyr_cache.yqyr_cache_v1` | `yqyr-cache-stepfunction` | `yqyr-cache-lambda` Lambda | `redshift_core` | `yqyr_cache_db` |
