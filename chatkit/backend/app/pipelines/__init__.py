"""Pipeline-lineage discovery + graph.

Cross-repo lineage extractor that produces a canonical graph of apps,
stages, Redshift/Glue tables, S3 prefixes, and the AWS resources that
move data between them.

The crawler runs multiple discovery passes over every repo (configs,
live AWS, code patterns, LLM summaries, doc DAGs) and merges results
by canonical node ID. See `docs/plan.md` and the plan file at
`~/.claude/plans/research-on-the-frontier-fluffy-bubble.md` for the
full design.
"""
