---
id: derived_common_output_eda
title: Derived common output EDA with price outlook plot
signals:
  - derived common output
  - common output
  - price outlook
  - eda of derived
required_entities:
  - customer
  - date
candidate_tables:
  - prod.common_output.common_output_format
actions:
  - action: inspect_table_metadata
    inputs:
      table_name: "prod.common_output.common_output_format"
      datasource: "redshift_analytics"
  - action: extract_sql
    inputs:
      datasource: "redshift_analytics"
      query_template: "SELECT customer, sales_date, observation_date, observation_time, origin, destination, carrier, channel, price_inc, price_exc, tax, yqyr, cabin, trip_type FROM prod.common_output.common_output_format WHERE sales_date = {{sales_date}} {{customer_filter}} ORDER BY observation_time DESC"
      dataset_name: "derived_common_output"
  - action: run_analysis
    inputs:
      analysis_spec:
        mode: profile_dataset
        focus: derived_common_output_eda
python_template: |
  import uuid
  rows = list_datasets()
  if rows:
      df = load_dataset(rows[-1]["dataset_id"])
      if "price_inc" in df.columns:
          series = pd.to_numeric(df["price_inc"], errors="coerce").dropna()
      else:
          series = pd.Series(dtype=float)
      plot_path = None
      if not series.empty:
          fig, ax = plt.subplots(figsize=(8, 4))
          series.hist(ax=ax, bins=40)
          ax.set_title("Price Outlook Distribution (price_inc)")
          ax.set_xlabel("price_inc")
          ax.set_ylabel("count")
          plot_path = f"/tmp/price_outlook_{uuid.uuid4().hex[:8]}.png"
          fig.tight_layout()
          fig.savefig(plot_path, dpi=120)
          plt.close(fig)
      payload = {
          "analysis_mode": "python_price_outlook",
          "results": {
              "row_count": int(len(df)),
              "price_inc_count": int(series.count()),
              "price_inc_mean": float(series.mean()) if not series.empty else None,
              "price_inc_p50": float(series.quantile(0.5)) if not series.empty else None,
              "price_inc_p90": float(series.quantile(0.9)) if not series.empty else None,
              "plot_path": plot_path,
          },
          "summary_stats": {
              "rows": int(len(df)),
              "price_inc_count": int(series.count()),
          },
          "report_markdown": "## Derived Common Output EDA\n- Rows: {}\n- price_inc count: {}\n- Plot: {}".format(len(df), int(series.count()), plot_path or "not generated"),
          "caveats": [],
      }
      save_analysis(payload)
---
Use this card for EDA-style prompts on derived/common output with customer + date constraints and price outlook plotting.
