---
name: cascading_investigation
description: Autonomous root-cause walk when a prod table / S3 output is empty or wrong — cascade through the S3 mirror, upstream pipeline stages, federated MySQL config, and the producer code.
keywords: [empty, no data, 0 rows, missing, why is, missing data, investigate empty, missing rows, customer not found, stale, staleness, data lag, root cause, upstream, cascade, fallback, pipeline break, break point, break-point, anomalies missing, no anomalies, empty table]
tier: high
---

## When to use this skill

Trigger this playbook whenever the user asks something that looks like:

- "Why is `prod.analytics.market_level_anomalies_v4` empty for customer DE today?"
- "Are there anomalies for B6 today?" → you query and get 0 rows
- "Why is customer X missing from the anomalies pipeline?"
- "Why does the dashboard show no data for DE?"
- Any `execute_sql` or `fetch_s3` you ran came back empty and the user is asking why.

**Do not** jump straight to speculating about producers. Walk the cascade — the tool
chain is fast (~30-60s) and produces a grounded answer naming the break-point stage.

## The cascade (stop at the first step that explains the emptiness)

### 0. Ground the concept
```
search_kb("<what the user asked>")           # e.g. "market anomalies customer empty"
search_kb("s3 bucket <concept>")             # for the S3 mirror
```
This resolves the canonical table + S3 prefix. Don't name tables from memory — you
will hallucinate.

### 1. Query the prod Redshift table
```python
execute_sql(
  "SELECT COUNT(*) AS n FROM prod.analytics.market_level_anomalies_v4 "
  "WHERE sales_date = 20260419 AND customer = 'DE'",
  datasource="redshift_analytics",
)
```
- `n > 0` → answer the original question; skip the rest.
- `n == 0` → continue.

### 2. Check what partitions ARE loaded
```
inspect_table("prod.analytics.market_level_anomalies_v4")
```
- Latest partition < today → report a **data-lag** diagnosis; re-run on the latest
  available partition and offer that answer instead.
- Today's partition is loaded but empty for this `customer` → continue.

### 3. Check the S3 mirror
`search_kb("s3 bucket <concept>")` gives you the Redshift → S3 mirror lookup in
`s3_buckets.md`. Call `fetch_s3` on the date + customer prefix.

- S3 has the file → the pipeline wrote the output but the Redshift load lagged or
  filtered this customer. Say so, cite both the S3 key and the Redshift row count.
- S3 missing → continue.

### 4. Walk upstream in the lineage graph
```
trace_pipeline("market_level_anomalies_v4", direction="upstream", depth=6)
```
You'll get the stages in order. For the anomalies chain that's usually:
`common-output → derived-common-output → competitive-position → market-level-analysis
 → market-level-generator → market_level_anomalies_v4`.

For **each** upstream stage starting with the one nearest the empty output:
```python
# (a) Redshift output (the stage writes here per lineage graph)
execute_sql("SELECT COUNT(*) FROM <stage's redshift output> WHERE sales_date=... AND customer='DE'")

# (b) S3 output (the stage's prefix per s3_buckets.md)
fetch_s3(bucket="s3-atp-3victors-3vprod-use1-<purpose>",
         key_or_prefix="<pattern>/2026/04/19/DE/")
```
- Both > 0 → this stage is fine; walk one step further upstream.
- Both empty → **this is the break-point**. Stop walking; move to Step 5.

### 5. At the break-point, find the cause

#### 5a. Config — is the customer even onboarded / enabled?
Most "customer missing today" cases are a config-drop. Check the federated MySQL
mirrors that Redshift exposes directly:
```python
execute_sql(
  "SELECT * FROM federated_priceeye.customer_defaults WHERE customer = 'DE'",
  datasource="redshift_analytics",
)
execute_sql(
  "SELECT * FROM federated_priceeye.site_hierarchy WHERE customer = 'DE'",
  datasource="redshift_analytics",
)
```
- Row missing or `is_enabled = 0` → customer not scheduled today. **Answer.**
- Row present and enabled → continue to 5b.

(See `federated_schemas.md` and `investigation_patterns.md` Step 3.5 for the full
federated catalog.)

#### 5b. Code — read the producer to find silent filters
From `trace_pipeline`'s response, every stage node carries `metadata.repo` and
`metadata.config_file`. Locate the code:
```
bash("ls ~/git/<repo>/")
bash("cd ~/git/<repo> && grep -rn 'def main\\|def run\\|if __name__' src/ | head")
read_file("~/git/<repo>/<entry>.py")          # or .java / .scala
```
Look for:
- Early returns on empty input (`if df.empty: return`)
- `customer in EXCLUDED_CUSTOMERS` lists
- `if not config.enabled: skip`
- Hard-coded filters that could drop `DE` (e.g., min row thresholds)
- Recent commits with `git("log --oneline -10 -- <file>", working_dir="~/git/<repo>")`

#### 5c. 3VDEV CloudWatch (safe)
Only if the break-point stage has a DEV mirror Lambda / SFN. You CAN look at 3VDEV
logs for crashes since the process runs on 3VDEV creds:
```
sfn_list_executions(state_machine_arn="arn:aws:states:...:<DEV_SFN>", status="FAILED")
lambda_get_last_errors(function_name="<DEV_LAMBDA>")
logs_insights_query(log_group="/aws/lambda/<DEV_LAMBDA>", ...)
```
Evidence found → pair it with the code reading from 5b. **Be explicit in your
answer that this is a DEV observation**, not a prod observation — the two
environments are not guaranteed identical.

#### 5d. 3VPROD CloudWatch (HARD BOUNDARY — do not attempt)
The process runs on 3VDEV with cross-account access for **S3 + Redshift only**.
3VPROD CloudWatch logs, 3VPROD Lambda, 3VPROD Step Functions are **not reachable**.
**Do not call `lambda_get_last_errors` / `sfn_list_executions` / `logs_insights_query`
with a prod ARN — the call will fail with a `AccessDenied` or return nothing
meaningful, and you MUST NOT try to `aws sts assume-role` into 3VPROD.**

When you hit this boundary, end the investigation with a concrete handoff:

> I've traced the break to `<stage>` in PROD. I can't reach 3VPROD Lambda / SFN
> logs from our 3VDEV session. To close this out, please run:
>
> ```
> assume 3VPROD
> aws stepfunctions list-executions \
>   --state-machine-arn <ARN> --status-filter FAILED --max-results 5
> aws logs tail /aws/lambda/<NAME> --since 2h --format short
> ```
>
> and paste the output (or the specific error) back here — I'll analyse it and
> name the root cause.

### 6. Write up the chain
Always include, in this order:

1. The original empty query you ran (with the partition filter).
2. Every upstream stage you checked with its row count (Redshift + S3).
3. The break-point stage name and the specific cause found (config miss / code
   filter / upstream empty / data lag).
4. If you hit 5d, the exact `assume 3VPROD` commands the user should run.

## Hard rules

- **Never assume a role other than 3VDEV.** No `aws sts assume-role`, no profile
  switching, no `assume 3VPROD` from inside the agent.
- **Default to PROD data** (per the session banner) unless the user asked for dev.
  All `execute_sql` / `fetch_s3` in this skill assume prod tables + buckets.
- **Stop walking upstream as soon as you find the break-point.** Don't exhaust the
  full chain if stage N-1 already explains the emptiness — the chain below N-1 is
  presumably still fine.
- **Quote the graph, not your memory.** Every stage name / table / bucket in your
  answer must come from a `search_kb`, `trace_pipeline`, or tool-call result you
  just ran. If the lineage graph is silent on some stage, say so.

## See also

- `pipeline_lineage.md` — how `trace_pipeline` works and what it returns
- `s3_buckets.md` — Redshift → S3 mirror lookup
- `investigation_patterns.md` — the 5-step fallback chain embedded in this cascade
- `aws_readonly.md` — why 3VPROD CloudWatch / Lambda / SFN is out of reach
- `federated_schemas.md` — the federated MySQL schemas exposed through Redshift
