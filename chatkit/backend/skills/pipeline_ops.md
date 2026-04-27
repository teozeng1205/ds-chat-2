---
name: pipeline_ops
description: Read-only AWS ops patterns through guarded AWS CLI commands.
keywords: [sfn, step, function, lambda, ecs, cloudwatch, alarm, eventbridge, rule, logs, insights, cluster, task, pipeline, ops, failure, error, timeout, schedule]
---

# Pipeline Ops

## Live AWS ops access

Use `bash` with AWS CLI for read-only AWS ops checks. The process runs with
3VDEV credentials; be explicit when an answer is about dev resources.

AWS mutating or execution commands (`run-task`, `invoke`, `start-execution`,
`put-*`, `update-*`, `delete-*`, `send-message`, `publish`, etc.) are guarded
and require explicit approval support. Do not use them for routine investigation.

## Read-only command patterns

- Step Functions state machines:
  `aws stepfunctions list-state-machines`
- Step Functions executions:
  `aws stepfunctions list-executions --state-machine-arn <ARN> --status-filter FAILED --max-results 20`
- Step Functions execution detail:
  `aws stepfunctions describe-execution --execution-arn <ARN>`
- Lambda recent errors:
  `aws logs filter-log-events --log-group-name /aws/lambda/<FUNCTION> --filter-pattern '?ERROR ?Exception ?Traceback ?Task timed out' --limit 50`
- Logs Insights:
  `aws logs start-query --log-group-name <GROUP> --start-time <EPOCH> --end-time <EPOCH> --query-string 'fields @timestamp, @message | sort @timestamp desc | limit 50'`
  then `aws logs get-query-results --query-id <ID>`.
- ECS running tasks:
  `aws ecs list-tasks --cluster <CLUSTER>` then
  `aws ecs describe-tasks --cluster <CLUSTER> --tasks <TASK_ARNS...>`.
- ECS stopped task reasons:
  `aws ecs list-tasks --cluster <CLUSTER> --desired-status STOPPED`
  then `aws ecs describe-tasks`.
- CloudWatch alarms:
  `aws cloudwatch describe-alarms --state-value ALARM --max-records 100`.
- EventBridge rule and targets:
  `aws events describe-rule --name <RULE>` and
  `aws events list-targets-by-rule --rule <RULE>`.

Prefer small `--query` filters or pipe to `python -m json.tool` / `jq` when
available so the output stays compact.

