---
name: aws_readonly
description: Read-only AWS CLI patterns, 3VDEV account context, cross-account S3 access, and why 3VPROD infra is not reachable.
keywords: [aws, s3, glue, lambda, cloudwatch, sfn, step, function, ecs, eventbridge, redshift, rds, ssm, secrets, athena, cloudformation, iam, ecr, kinesis, sqs, sns, 3vdev, 3vprod, account, prod, dev, logs, insights]
tier: high
---

## AWS CLI (read-only investigation)

`aws` CLI is available in every `bash()` call. Avoid mutating ops (s3 rm, delete-*, put-*).
Region: `us-east-1`. Check identity: `aws sts get-caller-identity`.

Resource names (Lambda functions, SFN state machines, ECS clusters, alarm names):
use `search_kb("aws infrastructure")` or `search_kb("lambda functions priceeye")` —
the KB indexes `~/git/documentations/`. Alternatively:
`aws lambda list-functions --query 'Functions[].FunctionName' --output text`.

| Service | Key commands |
|---|---|
| S3 | `aws s3 ls` / `aws s3 ls s3://BUCKET/PREFIX/` / `aws s3 cp s3://... /tmp/` |
| CloudWatch Logs | `aws logs filter-log-events --log-group-name NAME --start-time $(($(date +%s)-3600))000 --filter-pattern "ERROR"` |
| Logs Insights (async) | `aws logs start-query … --query-string '…'` → poll `aws logs get-query-results --query-id ID` |
| Glue | `aws glue get-databases` / `get-tables --database-name DB` / `get-partitions --database-name DB --table-name T` |
| Step Functions | `aws stepfunctions list-state-machines` / `list-executions --status-filter FAILED` / `describe-execution` / `get-execution-history` |
| Lambda | `aws lambda list-functions` / `get-function-configuration --function-name NAME` |
| EventBridge | `aws events list-rules` / `list-targets-by-rule --rule NAME` |
| CloudWatch alarms | `aws cloudwatch describe-alarms --state-value ALARM` |
| Athena | `aws athena list-work-groups` / `get-query-execution --query-execution-id ID` |
| Redshift Serverless | `aws redshift-serverless list-workgroups` (use `execute_sql` for actual SQL) |
| RDS / Aurora | `aws rds describe-db-clusters` / `describe-events --source-type db-cluster --duration 60` |
| SSM | `aws ssm get-parameters-by-path --path /priceeye/ --recursive` |
| Secrets Manager | `aws secretsmanager list-secrets` |
| CloudFormation | `aws cloudformation list-stacks` / `describe-stacks --stack-name NAME` |
| ECS | `aws ecs list-clusters` / `list-services --cluster CLUSTER` / `list-tasks --cluster CLUSTER --service-name SVC` / `describe-tasks` |
| SQS | `aws sqs list-queues` / `get-queue-attributes --queue-url URL --attribute-names All` |
| SNS | `aws sns list-topics` / `list-subscriptions-by-topic --topic-arn ARN` |
| Kinesis | `aws kinesis list-streams` / `describe-stream-summary --stream-name NAME` (prefer `kinesis_tail` tool for records) |
| IAM | aws iam list-roles --query 'Roles[?contains(RoleName,`priceeye`)].RoleName' |
| ECR | `aws ecr describe-repositories` / `describe-images --repository-name NAME` |
| CloudTrail | `aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventName,AttributeValue=StartExecution` |
| EC2 metadata | `curl -s http://169.254.169.254/latest/meta-data/instance-type` |

S3 bucket pattern: `s3-atp-3victors-{3vdev|3vprod}-use1-{purpose}` — use `aws s3 ls` to discover.
`fetch_s3` is preferred over raw `aws s3 cp` for structured data investigation.

## Account context — CRITICAL

`aws sts get-caller-identity` → always **3VDEV (590183652635)**. There is no `assume 3VPROD`.

| Goal | Use this | Note |
|---|---|---|
| Production SQL data | `execute_sql` | Cross-account IAM to 3VPROD Redshift |
| Production S3 data | `fetch_s3` or `aws s3` on `s3-atp-3victors-3vprod-use1-*` | Cross-account bucket policy |
| 3VPROD alarms / SFN / Lambda / Glue / Logs | ❌ Not reachable | Those are 3VPROD-only |

**3VPROD S3 buckets readable with 3VDEV creds (cross-account bucket policy):**
- `s3-atp-3victors-3vprod-use1-pe-common-output` — hourly price observations
- `s3-atp-3victors-3vprod-use1-anomaly-datasets` — market/segment anomaly output
- `s3-atp-3victors-3vprod-use1-collection-anomalies` — collection anomaly files
- `s3-atp-3victors-3vprod-use1-pe-packager-archive` — customer delivery files

**When asked about production pipeline health (alarms, SFN failures, Lambda errors):**
Do NOT use `aws cloudwatch` / `stepfunctions` / `lambda` / `logs` — those see only 3VDEV. Use instead:
- Data freshness → `aws s3 ls s3://s3-atp-3victors-3vprod-use1-anomaly-datasets/market-level/v4/…` or `execute_sql` checking latest `sales_date` in `prod.analytics.*`.
- Collection issues → `execute_sql` on `prod.monitoring.provider_combined_audit`.
- 3VDEV infra (dev Athena / SFN / Lambda) → regular AWS CLI, but be explicit in your answer that those are **dev resources, not production**.

**When a cascading data investigation hits the prod-ops wall** (see
`cascading_investigation.md` Step 5d), stop immediately and hand off. The canonical
phrasing is:

> I've traced the break to `<stage>` in PROD. I can't reach 3VPROD Lambda / SFN
> logs from our 3VDEV session. Please run `assume 3VPROD` locally and then
> `aws stepfunctions list-executions --state-machine-arn <ARN> --status-filter FAILED`
> / `aws logs tail /aws/lambda/<NAME> --since 2h` and paste the output back —
> I'll analyse it from there.

**Never** call `aws sts assume-role` to try to cross into 3VPROD. The user has
explicitly forbidden role assumption in this agent.
