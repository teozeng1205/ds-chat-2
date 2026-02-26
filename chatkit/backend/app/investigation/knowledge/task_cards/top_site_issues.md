---
id: top_site_issues
title: Top site issues and impact
signals:
  - top site issues
  - site issue
  - what is the impact
  - issue impact
required_entities:
  - provider
candidate_tables:
  - prod.monitoring.provider_combined_audit
actions:
  - action: inspect_table_metadata
    inputs:
      table_name: "prod.monitoring.provider_combined_audit"
      datasource: "redshift_core"
  - action: extract_sql
    inputs:
      datasource: "redshift_core"
      query_template: "SELECT issue_sources, issue_reasons, providercode, sitecode, COUNT(*) AS issue_count FROM prod.monitoring.provider_combined_audit WHERE sales_date = {{sales_date}} {{provider_filter}} {{site_filter}} AND issue_sources <> 'request' AND issue_sources <> '' AND issue_reasons <> '' GROUP BY issue_sources, issue_reasons, providercode, sitecode ORDER BY issue_count DESC"
      dataset_name: "site_issue_groups"
  - action: extract_sql
    inputs:
      datasource: "redshift_core"
      query_template: "SELECT providercode, sitecode, COUNT(*) AS total_requests, SUM(CASE WHEN (issue_sources <> '' OR filterreason <> '') THEN 1 ELSE 0 END) AS issue_requests, ROUND(100.0 * SUM(CASE WHEN (issue_sources <> '' OR filterreason <> '') THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS issue_rate_pct FROM prod.monitoring.provider_combined_audit WHERE sales_date = {{sales_date}} {{provider_filter}} {{site_filter}} GROUP BY providercode, sitecode ORDER BY issue_rate_pct DESC"
      dataset_name: "issue_impact"
  - action: run_analysis
    inputs:
      analysis_spec:
        mode: profile_dataset
        focus: issue_impact
analysis_mode: profile_dataset
analysis_instructions: "Summarize top issue groups and impact rates by provider/site; call out scope and caveats."
python_template: |
  rows = list_datasets()
  if len(rows) >= 2:
      issues = load_dataset(rows[-2]["dataset_id"])
      impact = load_dataset(rows[-1]["dataset_id"])
      top = issues.head(10).to_dict(orient="records")
      max_rate = None
      if "issue_rate_pct" in impact.columns:
          series = pd.to_numeric(impact["issue_rate_pct"], errors="coerce").dropna()
          if not series.empty:
              max_rate = float(series.max())
      payload = {
          "analysis_mode": "python_site_issue_summary",
          "results": {"top_issues": top},
          "summary_stats": {"issue_groups": int(len(issues)), "max_issue_rate_pct": max_rate},
          "report_markdown": "## Top Site Issues\n- Issue groups: {}\n- Max issue rate pct: {}".format(len(issues), max_rate),
          "caveats": [],
      }
      save_analysis(payload)
---
Use for provider/site issue and impact investigations.
