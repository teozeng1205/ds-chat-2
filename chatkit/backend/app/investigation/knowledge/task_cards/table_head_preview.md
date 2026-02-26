---
id: table_head_preview
title: Table head preview
signals:
  - query the head
  - head of table
  - preview table
  - show sample rows
required_entities:
  - table
candidate_tables:
  - prod.monitoring.combined_audit
  - prod.monitoring.provider_combined_audit
actions:
  - action: inspect_table_metadata
    inputs:
      table_name: "{{resolved_table}}"
      datasource: "{{resolved_datasource}}"
  - action: extract_sql
    inputs:
      datasource: "{{resolved_datasource}}"
      query_template: "SELECT * FROM {{resolved_table}} LIMIT 200"
      dataset_name: "table_head"
  - action: run_analysis
    inputs:
      analysis_spec:
        mode: profile_dataset
        focus: preview
analysis_mode: profile_dataset
analysis_instructions: "Profile the sampled rows and summarize schema, missingness, and quick numeric stats."
python_template: ""
---
Use this card for quick table previews when user asks for head/preview/sample rows.
