---
id: customer_collection_anomalies
title: Customer collection anomalies from S3
signals:
  - customer collection anomalies
  - collection anomalies
  - anomalies yesterday
required_entities:
  - date
candidate_tables: []
actions:
  - action: extract_s3
    inputs:
      bucket: "s3-atp-3victors-3vdev-use1-collection-anomalies"
      key_template: "collection-customer/v1/{{yyyy}}/{{mm}}/{{dd}}/"
      dataset_name: "customer_collection_anomalies"
  - action: run_analysis
    inputs:
      analysis_spec:
        mode: profile_dataset
        focus: anomaly_summary
analysis_mode: profile_dataset
analysis_instructions: "Summarize anomaly counts and key dimensions from collection anomaly files."
python_template: |
  rows = list_datasets()
  if rows:
      df = load_dataset(rows[-1]["dataset_id"])
      confirmed = df
      for col in ["confirmed", "is_confirmed", "confirmed_anomaly", "status"]:
          if col in df.columns:
              if str(df[col].dtype).lower() == "bool":
                  confirmed = df[df[col]]
              else:
                  normalized = df[col].astype(str).str.lower()
                  confirmed = df[normalized.isin({"1", "true", "yes", "confirmed", "y"})]
              break
      payload = {
          "analysis_mode": "python_collection_anomaly_summary",
          "results": {"preview": confirmed.head(20).to_dict(orient="records")},
          "summary_stats": {"rows": int(len(df)), "confirmed_anomalies": int(len(confirmed))},
          "report_markdown": "## Customer Collection Anomalies\n- Rows: {}\n- Confirmed anomalies: {}".format(len(df), len(confirmed)),
          "caveats": [],
      }
      save_analysis(payload)
---
Use for customer collection anomaly requests where source data is in S3 prefixes.
