---
id: table_eda
title: Deep table EDA
signals:
  - eda
  - profile table
  - explore table
  - analyze table
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
      dataset_name: "table_preview"
  - action: extract_sql
    inputs:
      datasource: "{{resolved_datasource}}"
      query_template: "SELECT * FROM {{resolved_table}} LIMIT {{sample_rows}}"
      dataset_name: "table_profile_sample"
  - action: run_analysis
    inputs:
      analysis_spec:
        mode: profile_dataset
        focus: deep_eda
analysis_mode: profile_dataset
analysis_instructions: "Build a deep EDA report with missingness, cardinality, numeric quantiles, temporal coverage, and correlations."
python_template: |
  # Optional deeper custom stats.
  rows = list_datasets()
  if rows:
      df = load_dataset(rows[-1]["dataset_id"])
      numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
      payload = {
          "analysis_mode": "python_eda_extras",
          "results": {
              "numeric_column_count": len(numeric_cols),
              "row_count": int(len(df)),
          },
          "summary_stats": {
              "numeric_column_count": len(numeric_cols),
              "rows": int(len(df)),
          },
          "report_markdown": "## Python EDA Extras\n- Numeric columns: {}\n- Rows: {}".format(len(numeric_cols), len(df)),
          "caveats": [],
      }
      save_analysis(payload)
---
Use this card when the user asks for EDA/profile/exploration for a specific table.
