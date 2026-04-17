# Skills

Markdown files loaded dynamically by `app.skills.SkillRegistry` and
appended to the main agent's system prompt per turn based on the
user's request.

Format (YAML-ish frontmatter + body):

```
---
name: sql_investigation
description: When to run SQL vs. S3 fetches; partition rules.
keywords: [sql, redshift, partition, mysql, analytics, monitoring]
tier: high
---
Body goes here as regular markdown. The body is injected verbatim
into the main prompt inside `<skill name="...">...</skill>` tags.
```

Rules:
- `name`: short snake_case ID used in tags.
- `description`: one-line human summary.
- `keywords`: small set of tokens (case-insensitive). The default
  classifier scores by fraction of keywords that appear in the user's
  prompt.
- `tier`: `high` | `normal` | `low`. High-tier skills get a small
  score boost; low-tier ones get a small penalty.
- Filenames starting with `_` (like this one) are NOT loaded as skills.

The classifier will pick the top ~3 skills per turn. Keep skill
bodies focused and brief; long bodies bloat the prompt.
