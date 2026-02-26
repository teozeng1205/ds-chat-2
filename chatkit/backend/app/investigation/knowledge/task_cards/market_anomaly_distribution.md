---
id: market_anomaly_distribution
title: Market anomalies with impact-score distribution
signals:
  - market anomalies
  - distribution of impact score
  - impact score
required_entities:
  - date
candidate_tables:
  - prod.analytics.market_level_anomalies_v3
actions:
  - action: inspect_table_metadata
    inputs:
      table_name: "prod.analytics.market_level_anomalies_v3"
      datasource: "redshift_analytics"
  - action: extract_sql
    inputs:
      datasource: "redshift_analytics"
      query_template: "SELECT observation_date, mkt, seg, top_offenders, cp, dow, impact_score, customer, sales_date FROM prod.analytics.market_level_anomalies_v3 WHERE sales_date = {{sales_date}} {{customer_filter}} AND any_anomaly = 1 ORDER BY impact_score DESC"
      dataset_name: "market_anomalies"
  - action: run_analysis
    inputs:
      analysis_spec:
        mode: profile_dataset
        focus: market_distribution
analysis_mode: profile_dataset
analysis_instructions: "Compute and report distribution stats for impact_score and top markets/offenders."
python_template: |
  rows = list_datasets()
  if rows:
      df = load_dataset(rows[-1]["dataset_id"])
      if "impact_score" in df.columns:
          series = pd.to_numeric(df["impact_score"], errors="coerce").dropna()
          stats = {
              "count": int(series.count()),
              "mean": float(series.mean()) if not series.empty else None,
              "p50": float(series.quantile(0.5)) if not series.empty else None,
              "p90": float(series.quantile(0.9)) if not series.empty else None,
              "max": float(series.max()) if not series.empty else None,
          }
          payload = {
              "analysis_mode": "python_market_distribution",
              "results": {"stats": stats},
              "summary_stats": stats,
              "report_markdown": "## Market Anomaly Distribution\n- Count: {count}\n- Mean: {mean}\n- P50: {p50}\n- P90: {p90}\n- Max: {max}".format(**stats),
              "caveats": [],
          }
          save_analysis(payload)
---
Use for market anomaly and impact-score distribution requests.
