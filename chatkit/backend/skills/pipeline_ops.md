---
name: pipeline_ops
description: Live AWS pipeline inspection — Step Functions, Lambda logs, ECS, alarms, EventBridge schedules.
keywords: [sfn, step, function, lambda, ecs, cloudwatch, alarm, eventbridge, rule, logs, insights, cluster, task, pipeline, ops, failure, error, timeout, schedule]
tier: high
---

## Live AWS ops tools (read-only)

All tools run with 3VDEV credentials and do NOT mutate the cloud.

### Step Functions
- `sfn_list_executions(state_machine_arn, status_filter?)` — e.g. `statusFilter="FAILED"` for last failures.
- `sfn_describe_execution(execution_arn)` — status, input, output, error, cause.
- `sfn_get_execution_history(execution_arn, max_results=200)` — flattened events; `failure_count` is pre-computed in the response.

### Lambda errors
- `lambda_get_last_errors(function_name, lookback_hours=6)` — filters `/aws/lambda/{fn}` for
  ERROR / Exception / Traceback / "Task timed out".

### Logs Insights
- `logs_insights_query(log_group, query, since_seconds=3600)` — polls to completion up to 60s.
  Use for ad-hoc scans: `fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc`.

### ECS
- `ecs_describe_tasks(cluster)` — running tasks + health.
- `ecs_list_stopped_reasons(cluster, service?)` — stopped tasks + `stoppedReason` + container exit codes
  (`137` = OOMKilled, `139` = segfault).

### CloudWatch / EventBridge
- `cloudwatch_alarms(state_value="ALARM")` — currently-firing alarms.
- `eventbridge_describe_rule(name)` — the schedule, the event pattern, and the rule's targets.

## Typical investigations

- **Overnight pipeline broke?** → `sfn_list_executions(arn, status_filter="FAILED", since=last night)`
  → `sfn_describe_execution` on the first failure → `sfn_get_execution_history` to find the failed
  state → `lambda_get_last_errors` on the responsible Lambda.
- **Why did my ECS service crash?** → `ecs_list_stopped_reasons(cluster, service)`
  → `logs_insights_query` on the ECS log group.
- **What triggers this job?** → `eventbridge_describe_rule(name)` — returns scheduleExpression + targets.

All tools return plain dicts with `ok` + the payload, never raise for expected
AWS errors (missing resources return `ok=False` with `error_type="NotFound"`).
