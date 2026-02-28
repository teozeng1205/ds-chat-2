---
id: competitive_position_detailed
title: Detailed competitive position analysis
signals:
  - competitive position
  - detailed analysis of competitive position
  - competitive analysis
required_entities:
  - customer
  - date
candidate_tables:
  - analytics.market_level_analysis_v2
  - analytics.segment_level_analysis_v2
actions:
  - action: inspect_table_metadata
    inputs:
      table_name: "analytics.market_level_analysis_v2"
      datasource: "redshift_analytics"
  - action: extract_sql
    inputs:
      datasource: "redshift_analytics"
      query_template: "SELECT customer, sales_date, segment_name, competitive_position, comparison_type, customer_brand, competitor_brand, metro_market, region_name, depart_period, carrier_group, cabin_group, top_offenders, carrier_contribution, itinerary_count, itinerary_percentage, impacted_dates, impacted_dates_percentage, avg_diff_min_ow, avg_pcnt_diff_min_ow, cp_weight, cp_score FROM analytics.market_level_analysis_v2 WHERE sales_date = {{sales_date}} {{customer_filter}} ORDER BY cp_score DESC"
      dataset_name: "competitive_position_market_level"
  - action: run_analysis
    inputs:
      analysis_spec:
        mode: profile_dataset
        focus: competitive_position
python_template: |
  rows = list_datasets()
  if rows:
      df = load_dataset(rows[-1]["dataset_id"])
      cp = df["competitive_position"].astype(str).str.upper() if "competitive_position" in df.columns else pd.Series(dtype=str)
      cp_counts = cp.value_counts().head(20).to_dict() if not cp.empty else {}
      top_offenders = {}
      if "top_offenders" in df.columns:
          top_offenders = df["top_offenders"].astype(str).value_counts().head(15).to_dict()
      score = pd.to_numeric(df["cp_score"], errors="coerce").dropna() if "cp_score" in df.columns else pd.Series(dtype=float)
      payload = {
          "analysis_mode": "python_competitive_position",
          "results": {
              "competitive_position_counts": cp_counts,
              "top_offenders": top_offenders,
              "cp_score_mean": float(score.mean()) if not score.empty else None,
              "cp_score_p50": float(score.quantile(0.5)) if not score.empty else None,
              "cp_score_p90": float(score.quantile(0.9)) if not score.empty else None,
          },
          "summary_stats": {
              "rows": int(len(df)),
              "distinct_competitive_positions": int(len(cp_counts)),
          },
          "report_markdown": "## Competitive Position Detailed Analysis\n- Rows: {}\n- Distinct competitive positions: {}\n- Top offenders captured: {}".format(len(df), len(cp_counts), len(top_offenders)),
          "caveats": [],
      }
      save_analysis(payload)
---
Use this card for detailed competitive-position analysis requests (customer/date scoped).
