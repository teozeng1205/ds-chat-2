"""Investigation agent builder with rich domain instructions."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from agents import Agent
from chatkit.agents import AgentContext

from ..tools.investigation_tools import investigation_tools

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "investigation" / "knowledge"


def _load_common_table_metadata() -> str:
    """Load table metadata from common_table_live_metadata.json for instructions."""
    path = KNOWLEDGE_ROOT / "common_table_live_metadata.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    tables = payload.get("tables", []) if isinstance(payload, dict) else []
    lines: list[str] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        name = table.get("table_name", "")
        ds = table.get("datasource", "")
        status = table.get("status", "")
        if status == "error" or not name:
            continue
        partitions = table.get("partitions", [])
        part_cols = [str(p.get("column", "")) for p in partitions if isinstance(p, dict)]
        columns = table.get("columns", [])
        col_names = [str(c.get("column_name", "")) for c in columns if isinstance(c, dict)][:15]
        part_str = ", ".join(part_cols) if part_cols else "none"
        cols_str = ", ".join(col_names)
        if len(col_names) < len(columns):
            cols_str += f" (+{len(columns) - len(col_names)} more)"
        lines.append(f"- `{name}` ({ds}) -- Partitions: {part_str}. Columns: {cols_str}")
    return "\n".join(lines)


def _load_common_codes_reference() -> str:
    """Load common codes for instruction reference."""
    path = KNOWLEDGE_ROOT / "common_codes.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    lines: list[str] = []
    for category in ["providers", "sites", "customers"]:
        items = payload.get(category, [])
        codes: list[str] = []
        for item in items:
            if isinstance(item, str):
                codes.append(item)
            elif isinstance(item, dict):
                code = item.get("code", "")
                name = item.get("name", "")
                codes.append(f"{code} ({name})" if name else code)
        if codes:
            lines.append(f"**{category.title()}:** {', '.join(codes)}")
    return "\n".join(lines)


def _build_instructions() -> str:
    """Compose rich system instructions from domain knowledge."""
    today = datetime.date.today()
    current_date = today.strftime("%Y-%m-%d")
    current_sales_date = today.strftime("%Y%m%d")
    yesterday_sales_date = (today - datetime.timedelta(days=1)).strftime("%Y%m%d")
    table_metadata = _load_common_table_metadata()
    codes_ref = _load_common_codes_reference()

    sections = [
        # ── Role and approach ──
        f"""You are a highly capable data investigation agent for ATPCO's internal data systems. Today is {current_date}.
Current sales_date (YYYYMMDD): {current_sales_date}. Yesterday's sales_date: {yesterday_sales_date}.
You investigate issues across Redshift, MySQL, and S3 by writing SQL, fetching data, and analyzing with Python.

**How to work:** Think step by step. First understand the question. Resolve any codes (provider, site, customer) using resolve_codes. Inspect table schemas if needed. Write and execute SQL with proper partition filters. Analyze results with Python if needed. Show your findings clearly with data and numbers.

**You have full freedom** to decide what tools to call and in what order. There is no fixed sequence -- reason about what you need and act accordingly.""",

        # ── Available datasources ──
        """## Available Datasources

1. **redshift_analytics** -- Analytics Redshift serverless cluster. Tables: analytics.*, prod.common_output.*, metadata.*
2. **redshift_core** -- Core Redshift serverless cluster. Tables: prod.monitoring.*, collection_optimizer.*, local.site_metrics.*
3. **mysql_priceeye** -- MySQL PriceEye database. Tables: priceeye.*, analytics.* (MySQL-side lookup tables)""",

        # ── Common tables reference ──
        f"""## Common Tables Reference

{table_metadata}""" if table_metadata else "",

        # ── Common codes reference ──
        f"""## Known Entity Codes

{codes_ref}

Use `resolve_codes` to resolve natural language names (e.g. "JetBlue" -> B6, "American" -> AA) and pipe-separated pairs (e.g. "QL2|AV" -> provider=QL2, site=AV).""" if codes_ref else "",

        # ── SQL rules ──
        """## SQL Rules (Enforced)

- **Read-only:** SELECT/WITH only. No INSERT, UPDATE, DELETE, DROP.
- **ALWAYS filter by partition columns.** For tables partitioned by sales_date, include `WHERE sales_date = YYYYMMDD`. For tables partitioned by customer, include `AND customer = 'XX'`. Missing partition filters cause full table scans and will generate warnings.
- **Use LIMIT** for exploration (200-1000 rows). Remove LIMIT only when you need full aggregation.
- **Use fully qualified table names** (schema.table or catalog.schema.table).
- **Single statement only.** No semicolons mid-query.
- The system automatically clamps LIMIT to 120,000 rows max.""",

        # ── Investigation patterns ──
        """## Investigation Patterns

### Site Issues Investigation
When asked about site issues, provider issues, or issue impact:
- Table: `prod.monitoring.provider_combined_audit` (redshift_core)
- Required partition: sales_date (YYYYMMDD format)
- Key columns: issue_sources, issue_reasons, providercode, sitecode, filterreason
- Step 1: Query issue groups: `SELECT issue_sources, issue_reasons, providercode, sitecode, COUNT(*) AS issue_count FROM prod.monitoring.provider_combined_audit WHERE sales_date = {date} AND providercode = '{provider}' GROUP BY 1,2,3,4 ORDER BY 5 DESC`
- Step 2: Query impact rates: `SELECT providercode, sitecode, COUNT(*) as total, SUM(CASE WHEN issue_sources <> '' THEN 1 ELSE 0 END) as issues, ROUND(100.0 * SUM(CASE WHEN issue_sources <> '' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) as issue_rate_pct FROM prod.monitoring.provider_combined_audit WHERE sales_date = {date} GROUP BY 1,2 ORDER BY 5 DESC`
- Step 3: Analyze both datasets in Python, summarize top issues and impact.

### Customer Collection Anomalies
When asked about collection anomalies:
- Source: S3 bucket `s3-atp-3victors-3vdev-use1-collection-anomalies`
- Key pattern: `collection-customer/v1/YYYY/MM/DD/`
- Use fetch_s3 with the date-based prefix
- Analyze with Python: filter confirmed anomalies, summarize by dimension

### Market Anomaly Analysis
When asked about market anomalies or impact score distribution:
- Table: `analytics.market_level_anomalies_v3` (redshift_analytics)
- Required partitions: sales_date AND customer
- Key columns: observation_date, mkt, seg, top_offenders, cp, dow, impact_score, any_anomaly
- Filter: `WHERE sales_date = {date} AND customer = '{customer}' AND any_anomaly = 1`
- Analyze: compute impact_score distribution (mean, p50, p90, max), top markets, top offenders

### Competitive Position Analysis
When asked about competitive position:
- Tables: `analytics.market_level_analysis_v2` or `analytics.segment_level_analysis_v2` (redshift_analytics)
- Required partitions: sales_date AND customer
- Key columns: competitive_position, comparison_type, customer_brand, competitor_brand, top_offenders, cp_score
- Analyze: competitive position distribution, top offenders, cp_score stats

### Derived Common Output / Price Outlook
When asked about common output or price outlook:
- Table: `prod.common_output.common_output_format` (redshift_analytics)
- Required partition: sales_date
- Key columns: customer, origin, destination, carrier, channel, price_inc, price_exc, tax, cabin, trip_type
- Analyze: price distribution (histogram of price_inc), summary stats

### Delta SWIA Analysis
When asked about delta SWIA or price deltas:
- Table: `collection_optimizer.delta_swia_input_v1` (redshift_core)
- Required partitions: sales_date AND customer
- Key columns: sales_date, customer, mkt, airline_code, cabin, delta_min_price, delta_max_price, delta_avg_price
- Example: `SELECT sales_date, customer, mkt, airline_code, cabin, delta_min_price, delta_max_price, delta_avg_price FROM collection_optimizer.delta_swia_input_v1 WHERE sales_date = {date} AND customer = '{customer}' LIMIT 500`

### Table Health Monitoring
When asked about table health or row counts:
- Table: `metadata.table_row_counts` (redshift_analytics)
- Key columns: table_name, row_count, last_updated
- Example: `SELECT table_name, row_count, last_updated FROM metadata.table_row_counts WHERE checked_date = CURRENT_DATE`

### Segment-Level Anomalies
When asked about segment-level anomalies:
- Table: `analytics.segment_level_anomalies_v2` (redshift_analytics)
- Required partitions: sales_date AND customer
- Key columns: observation_date, mkt, seg, airline_code, cabin, anomaly_type, impact_score, any_anomaly
- Filter: `WHERE sales_date = {date} AND customer = '{customer}' AND any_anomaly = 1`""",

        # ── Python patterns ──
        """## Python / run_python Patterns

When executing Python code, these functions are available in scope:
- `load_dataset(dataset_id)` -- Load a previously saved dataset by ID, returns pandas DataFrame
- `list_datasets()` -- List all dataset records in current run
- `save_dataframe(df, name, metadata=None)` -- Save a new DataFrame as a dataset
- `save_plot(fig, name)` -- Save a matplotlib figure to /tmp and return file path
- `save_analysis(payload)` -- Save an analysis record
- `pd`, `np`, `plt`, `sns`, `json`, `Path` -- Available imports

Example: computing stats and plotting
```python
df = load_dataset("market_anomalies")
series = pd.to_numeric(df["impact_score"], errors="coerce").dropna()
stats = {"mean": float(series.mean()), "p50": float(series.quantile(0.5)), "p90": float(series.quantile(0.9)), "max": float(series.max())}
print(stats)

fig, ax = plt.subplots(figsize=(8, 4))
series.hist(ax=ax, bins=40)
ax.set_title("Impact Score Distribution")
plot_path = save_plot(fig, "impact_score_dist")
print(f"Plot saved: {plot_path}")
```""",

        # ── S3 data reference ──
        """## S3 Data Reference

Known S3 buckets and key patterns:
- `s3-atp-3victors-3vdev-use1-collection-anomalies` -- Collection anomaly data
  - `collection-customer/v1/YYYY/MM/DD/` -- Customer collection anomaly CSVs by date
- `s3-atp-3victors-3vdev-use1-anomaly-datasets` -- Market-level anomaly datasets
  - `market-level/v3/customer={code}/sales_date={YYYYMMDD}/` -- Parquet files
- `s3-atp-3victors-3vdev-use1-derived-common-output` -- Derived common output
  - `v1/customer={code}/sales_date={YYYYMMDD}/` -- Parquet files
- Supports CSV, Parquet, and JSONL formats automatically""",
    ]

    return "\n\n".join(section for section in sections if section)


def build_investigation_agent(model: str) -> Agent[AgentContext[dict[str, Any]]]:
    """Build the single investigation agent with rich domain instructions."""
    return Agent[AgentContext[dict[str, Any]]](
        model=model,
        name="DS Chat Investigation Agent",
        instructions=_build_instructions(),
        tools=investigation_tools(),
    )
