---
id: generic_compare
title: Generic comparative analysis recipe
signals:
  - compare
  - versus
  - vs
  - difference between
required_entities: []
candidate_tables: []
actions:
  - action: run_analysis
    inputs:
      analysis_spec:
        mode: profile_dataset
        focus: comparison
analysis_mode: profile_dataset
analysis_instructions: "Profile available datasets and highlight differences in row counts, null rates, and numeric distributions."
python_template: ""
---
Use as a generic fallback when user asks to compare outputs without a specific predefined task.
