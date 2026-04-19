---
name: pipeline_lineage
description: How to use the cross-repo pipeline lineage graph — who writes what, who reads what, how data flows end to end between apps.
keywords: [pipeline, lineage, dag, upstream, downstream, chain, flow, produces, consumes, writes, reads, competitive-position, derived-common-output, dco, market-level, segment-level, mla, mlg, sla, slg, common-output, anomalies, trace, trace_pipeline]
tier: high
---

## The pipeline graph

A lot of PriceEye data flows through **chained stages across repos**. For
example the anomalies chain:

`common-output` → `derived-common-output` (Java) → `competitive-position`
(v2, Python) → `market-level-analysis` / `segment-level-analysis` →
`market-level-generator` / `segment-level-generator`.

Each stage has its own Redshift table and its own S3 bucket prefix, and is
deployed as a Lambda / ECS service / Step Function. A bad number at the
end of the chain usually comes from an earlier stage.

The agent has a **lineage graph** of every app, stage, Redshift table, S3
prefix, Glue table, Step Function, and EventBridge rule the crawler has
discovered across the DS repos. Use `trace_pipeline` to walk it.

## When to call `trace_pipeline`

**Rule 1 — investigating a bad number.** When the user asks about a
value in a Redshift table, an S3 file, or a dashboard that feels wrong,
**call `trace_pipeline(table_or_bucket, direction="upstream")` first**.
Then walk each upstream stage in order, spot-checking row counts /
partition freshness / last-run status via the Step Functions / Lambda /
Logs Insights tools.

**Rule 2 — "how does X work?"** When the user asks how an app, table,
or pipeline works, **call `trace_pipeline(X, direction="both", depth=3)`**
and summarize the chain (upstream → origin → downstream) *before* diving
into any single stage. Name the stages in order so the user sees the big
picture first.

## Inputs `trace_pipeline` accepts

Any of:
- a Redshift table name — `"market_level_anomalies_v4"`, `"prod.analytics.foo"`
- an S3 bucket prefix — `"s3-atp-3victors-3vprod-use1-anomaly-datasets"`
- a Glue table — `"glue-atp-…-analytics_db.market_level_anomalies_v4"`
- an app / stage canonical name — `"competitive-position"`, `"market-level-generator"`
- an alias — `"DCO"`, `"MLG"`, `"MLA"`, `"comp-pos"` (see `aliases.yaml`)

## Reading the result

The tool returns:
- `origin` — the node it resolved your string to
- `upstream` — ordered list of nodes reached walking upstream
- `downstream` — ordered list walking downstream
- `stages` — every stage/app in the neighborhood
- `tables` — every Redshift/Glue table in the neighborhood
- `s3_prefixes` — every S3 prefix in the neighborhood
- `edges` — every edge traversed with `{source, target, rel, provenance}`
- `summary` — a one-line human description you can quote to the user

If the graph is empty, the tool returns a clear error suggesting
`python scripts/build_pipeline_graph.py`. If the entity doesn't resolve,
the tool suggests trying an alternative name. Don't speculate about
lineage the graph doesn't confirm — say so and move on.

## Example — the anomalies question

User: "Why is market_level_anomalies_v4 wrong for B6 today?"

1. `trace_pipeline("market_level_anomalies_v4", direction="upstream", depth=4)` — reveals `market-level-generator` (writes it), `market-level-analysis` (its input), `competitive-position` (its input), `derived-common-output` (its input), `common-output` (its input).
2. For each upstream stage, check pipeline health: `sfn_list_executions` + `lambda_get_last_errors` + a quick `execute_sql` to confirm recent row counts in each stage's output table.
3. Report the bottleneck to the user in terms of the graph — "market-level-generator ran fine, but competitive-position had no output today for B6, which is why market-level-generator produced stale results." Then link the Redshift tables / S3 prefixes where they can verify.
