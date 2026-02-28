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


def _load_system_overview() -> str:
    """Load the PriceEye system overview doc for instructions."""
    path = KNOWLEDGE_ROOT / "docs" / "priceeye_system.md"
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    # Extract just the "Common Investigation Scenarios" section and the process → table map
    # to keep instructions focused and not bloated
    lines = text.splitlines()
    # Find the scenarios section
    scenarios_start = next((i for i, l in enumerate(lines) if "## Common Investigation Scenarios" in l), None)
    if scenarios_start is not None:
        return "\n".join(lines[scenarios_start:])
    return ""


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
    system_scenarios = _load_system_overview()

    sections = [
        # ── Role and approach ──
        f"""You are a highly capable data investigation agent for ATPCO's PriceEye platform. Today is {current_date}.
Current sales_date (YYYYMMDD): {current_sales_date}. Yesterday's sales_date: {yesterday_sales_date}.
You investigate issues across Redshift, MySQL, and S3 by writing SQL, fetching data, and analyzing with Python.

**PriceEye** is a real-time airline price intelligence platform. It collects prices from 20+ providers (airlines, GDS, OTAs), produces a Common Output table, and feeds analytics pipelines that compute anomalies, competitive positions, and billing metrics. The key data flow is:
  priceeye-v2 → combined_audit + common_output → ds-priceeye-analytics (anomalies) → market/segment anomaly tables
  priceeye-v2 → ds-internal-monitoring (dedup + join) → prod.monitoring.combined_audit + provider_combined_audit
  priceeye-v2 → ds-customer-monitoring → billing_db.customer_daily_requests_v1/v2/v3
  priceeye-v2 → ds-priceeye-data-collection → collection_optimizer.*, site_metrics.*, yqyr_cache.*
  priceeye-v2 → ds-priceeye-enrichment → tax_reg.* (regression coefficients, runs weekly on Tuesdays)

**How to work:** Think step by step. First understand the question. Resolve any codes (provider, site, customer) using resolve_codes. Inspect table schemas if needed. Write and execute SQL with proper partition filters. Analyze results with Python if needed. Show your findings clearly with data and numbers.

**You have full freedom** to decide what tools to call and in what order. There is no fixed sequence -- reason about what you need and act accordingly.""",

        # ── Available datasources ──
        """## Available Datasources

1. **redshift_analytics** -- Analytics Redshift serverless cluster.
   Tables: analytics.* (anomalies, scoring, analysis, pax_midt, daily_itins_prices_v2, oag_score_v2, revenue_score_v1), prod.common_output.*, billing_db.*, metadata.*, tax_reg.*, yqyr_cache.*
2. **redshift_core** -- Core Redshift serverless cluster.
   Tables: prod.monitoring.* (combined_audit, provider_combined_audit), collection_optimizer.* (delta_swia_input_v1, ingest_ttl_v1), local.site_metrics.* (capacity_final, cache_metrics_v1, retry_metrics_v1, import_metrics_v1), local.monitoring.*, local.federated_*
3. **mysql_priceeye** -- MySQL PriceEye database.
   Tables: priceeye.* (customer_defaults, site_hierarchy, transaction_rates), analytics.* (MySQL-side lookup tables: cabin_group, carrier_group, region, segment, anomalies_direction_score, anomalies_impact_score_weights, demo_carrier_substitutions), sales_poc.*, taxregression.*""",

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
- Filter: `WHERE sales_date = {date} AND customer = '{customer}' AND any_anomaly = 1`

### Audit Lifecycle Tracing
When asked to trace a request, diagnose what happened to a specific collection, or investigate why a request failed/wasn't delivered:
- Table: `prod.monitoring.combined_audit` (redshift_core)
- Required partition: sales_date
- Key columns: id, inputrequestid, customer, customercollectionid, providercode, sitecode, response_status, response_itinerarycount, issue_source, issue_reason, filterreason, retry_reason, enrichment_success_count, enrichment_failure_count, cache_itinerarycount, delivery_status, delivery_failurereason
- Step 1: Query by customer/provider/date: `SELECT id, providercode, sitecode, response_status, issue_source, issue_reason, filterreason, retry_reason, delivery_status, delivery_failurereason, response_itinerarycount FROM prod.monitoring.combined_audit WHERE sales_date = {date} AND customer = '{customer}' AND providercode = '{provider}' LIMIT 500`
- Step 2: Summarize issue counts by status: `SELECT issue_source, issue_reason, delivery_status, COUNT(*) AS cnt FROM prod.monitoring.combined_audit WHERE sales_date = {date} AND customer = '{customer}' GROUP BY 1,2,3 ORDER BY 4 DESC LIMIT 50`
- Step 3: Analyze in Python to classify root cause (request error vs. site error vs. filter vs. delivery failure)

### Provider Performance Analysis
When asked about retry rates, cache hit rates, TPS capacity, or provider health metrics:
- **Retry rates**: `local.site_metrics.retry_metrics_v1` (redshift_core) -- `retry_rate_pct` per providercode
- **Cache hit rates**: `local.site_metrics.cache_metrics_v1` (redshift_core) -- `cache_hit_rate`, `cache_miss_rate` per providercode/sitecode
- **Capacity (TPS)**: `local.site_metrics.capacity_final` (redshift_core) -- `capacity_tph` (transactions/hour, IQR-filtered; includes floor patches for QL2 ≥180 TPH, SS ≥3600 TPH)
- Required partition: sales_date on all site_metrics tables
- Example: `SELECT providercode, retry_rate_pct, total_requests FROM local.site_metrics.retry_metrics_v1 WHERE sales_date = {date} ORDER BY retry_rate_pct DESC LIMIT 50`

### Billing Metrics Analysis
When asked about billing, request counts, billable requests, or GDS/OTA/MSE breakdown per customer:
- Table: `billing_db.customer_daily_requests_v3` (redshift_analytics) -- most granular; broken down by site category
- Also available: `billing_db.customer_daily_requests_v1` (basic), `billing_db.customer_daily_requests_v2` (+ site code)
- Required partition: sales_date
- Key columns: customer, sales_date, site_category, total_reqs, requested_by_customers, GDS_scheduled, OTA_scheduled, MSE_scheduled, polled, cached, filtered, enrichment, success, failed, site_failed, bad_requests, true_site_issues, billable_requests
- Example: `SELECT customer, SUM(total_reqs), SUM(billable_requests), SUM(site_failed) FROM billing_db.customer_daily_requests_v3 WHERE sales_date = {date} GROUP BY customer ORDER BY 2 DESC`

### Collection Optimization & Ingest TTL
When asked about collection frequency, how often prices change, or TTL (time-to-live) for a carrier:
- **Delta SWIA data**: `collection_optimizer.delta_swia_input_v1` (redshift_core) -- minimum prices per carrier/OD/cabin/AP
  - Key columns: sales_date, customer, origin, destination, cabin, airline, brand, ap, min_price, pos
- **Ingest TTL**: `collection_optimizer.ingest_ttl_v1` (redshift_core) -- 25th-percentile hours between price changes per carrier
  - Key columns: sales_date, airline, carrier, pos, origin, destination, travel_period, cabin, ttl_hours
  - Both require sales_date partition; ingest_ttl_v1 also partitioned by airline
- Example: `SELECT carrier, pos, cabin, AVG(ttl_hours) AS avg_ttl FROM collection_optimizer.ingest_ttl_v1 WHERE sales_date = {date} AND airline = '{airline}' GROUP BY 1,2,3 ORDER BY avg_ttl DESC`

### PAX/MIDT Booking Analysis
When asked about passenger counts, booking volumes, or MIDT data for a customer:
- Table: `analytics.pax_midt` (redshift_analytics)
- Required partitions: sales_date AND customer
- Key columns: customer, origin, destination, carrier, cabin, ap_band, pax_count, booking_date
- Example: `SELECT origin, destination, carrier, cabin, SUM(pax_count) AS total_pax FROM analytics.pax_midt WHERE sales_date = {date} AND customer = '{customer}' GROUP BY 1,2,3,4 ORDER BY 5 DESC LIMIT 100`

### OAG Score (Seat Supply) Analysis
When asked about seat availability, market share, or OAG data for a customer:
- Table: `analytics.oag_score_v2` (redshift_analytics)
- Required partitions: sales_date AND customer
- Key columns: customer, origin, destination, carrier, cabin, flight_count, seat_count, market_share_pct
- Example: `SELECT origin, destination, carrier, SUM(seat_count), AVG(market_share_pct) FROM analytics.oag_score_v2 WHERE sales_date = {date} AND customer = '{customer}' GROUP BY 1,2,3 ORDER BY 4 DESC LIMIT 100`

### Revenue Score Analysis
When asked about revenue estimates, estimated impact, or revenue scoring for anomalies:
- Table: `analytics.revenue_score_v1` (redshift_analytics)
- Required partitions: sales_date AND customer
- Key columns: customer, origin, destination, carrier, cabin, ap_band, avg_price, pax_count, estimated_revenue
- Example: `SELECT origin, destination, carrier, SUM(estimated_revenue) AS est_revenue FROM analytics.revenue_score_v1 WHERE sales_date = {date} AND customer = '{customer}' GROUP BY 1,2,3 ORDER BY 4 DESC LIMIT 100`

### Customer-Level Collection Monitoring
When asked about per-customer collection health, success rates, substitute usage, or delivery rates:
- Table: `billing_db.customer_daily_requests_v2` or `prod.monitoring.combined_audit` (for granular)
- For daily totals: `SELECT customer, SUM(total_reqs), SUM(success), ROUND(100.0*SUM(success)/NULLIF(SUM(total_reqs),0),2) AS success_pct, SUM(site_failed) FROM billing_db.customer_daily_requests_v2 WHERE sales_date = {date} GROUP BY customer ORDER BY success_pct ASC`
- For combined_audit breakdown: filter by `customer = '{customer}'` and group by `providercode`, `sitecode`, `response_status`

### Tax Regression Analysis
When asked about tax regression, YQ/YR tax coefficients, or the tax regression pipeline:
- Coefficients table: `tax_reg.tax_reg_output_v1` (redshift_analytics) -- slope m, intercept b, R² per market
  - Partitioned by sales_date; key columns: pos, od, is_one_way, search_class, carrier, currency, m, b, r2, correlation
- Current MySQL coefficients: `taxregression.tax_regression_v1` (mysql_priceeye) -- overwritten every Tuesday
- YQYR predictions: `yqyr_cache.yqyr_predictions` -- predicted YQ/YR taxes per itinerary
- Example: `SELECT pos, od, carrier, m, b, r2 FROM tax_reg.tax_reg_output_v1 WHERE sales_date = {date} AND pos = 'US' AND carrier = '{carrier}' LIMIT 100`""",

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

        # ── PriceEye investigation scenarios ──
        f"""## PriceEye Process Quick Reference

Use `search_kb` to retrieve full process details. Key table → process mappings:
- **combined_audit / provider_combined_audit** → produced by ds-internal-monitoring (dedup pipeline, runs hourly)
- **billing_db.customer_daily_requests_*** → produced by ds-customer-monitoring (primary billing source)
- **analytics.market_level_anomalies_v4** → produced by ds-priceeye-analytics market-level-generator (22-day rolling model)
- **analytics.segment_level_anomalies_v2** → produced by ds-priceeye-analytics segment-level-generator
- **analytics.market_level_analysis_v2 / segment_level_analysis_v2** → intermediate competitive analysis tables
- **collection_optimizer.delta_swia_input_v1** → produced by ds-priceeye-data-collection delta SWIA unload (Glue job)
- **collection_optimizer.ingest_ttl_v1** → produced by ds-priceeye-data-collection ingest-ttl (25th-pct hours between price changes)
- **site_metrics.capacity_final / cache_metrics_v1 / retry_metrics_v1** → produced by ds-priceeye-data-collection site-metrics lambdas
- **prod.common_output.common_output_format** → produced by priceeye-v2 (raw) then normalized by priceeye-analytics DCO Spark job
- **tax_reg.tax_reg_output_v1** → produced by ds-priceeye-enrichment (runs every Tuesday)
- **analytics.pax_midt** → produced by ds-priceeye-analytics pax-midt process
- **analytics.revenue_score_v1** → produced by ds-priceeye-analytics revenue-score process
- **analytics.oag_score_v2** → produced by ds-priceeye-analytics oag-score process

{system_scenarios}""" if system_scenarios else "",

        # ── S3 data reference ──
        """## S3 Data Reference

All buckets follow the pattern `s3-atp-3victors-{env}-use1-{purpose}`. Production env = `3vdev` (or `3vprod` for some older paths).

Known S3 buckets and key patterns (use fetch_s3 with these):
- `s3-atp-3victors-3vdev-use1-collection-anomalies`
  - `collection-customer/v1/YYYY/MM/DD/` -- Customer collection anomaly CSVs by date
- `s3-atp-3victors-3vdev-use1-derived-common-output`
  - `v1/{customer}/{YYYY}/{MM}/{DD}/{HH}/` -- DCO Parquet (normalized price observations per customer)
  - `v1/customer={code}/sales_date={YYYYMMDD}/` -- Alternative partition path
- `s3-atp-3victors-3vdev-use1-anomaly-datasets`
  - `market-level/v4/{customer}/{YYYY}/{MM}/{DD}/` -- Market-level anomaly Parquet (v4 is latest)
  - `market-level/v3/customer={code}/sales_date={YYYYMMDD}/` -- Legacy v3 path
  - `segment-level/v4/{customer}/{YYYY}/{MM}/{DD}/` -- Segment-level anomaly Parquet
  - `daily_itins_prices/v2/{customer}/{YYYY}/{MM}/{DD}/` -- Daily itinerary prices by AP band
  - `oag_score/v2/{customer}/{YYYY}/{MM}/{DD}/` -- OAG seat supply metrics
  - `revenue_score/v1/{customer}/{YYYY}/{MM}/{DD}/revenue_estimates.csv` -- Revenue estimates (CSV)
  - `pax_midt/v1/{customer}/{YYYY}/{MM}/{DD}/` -- PAX/MIDT booking data (CSV)
- `s3-atp-3victors-3vdev-use1-competitive-position`
  - `v2/{customer}/{YYYY}/{MM}/{DD}/data.parquet` -- Competitive position Parquet
- `s3-atp-3victors-3vprod-use1-pe-common-output`
  - `{customer}/{YYYY}/{MM}/{DD}/{HH}/` -- Raw common output before DCO normalization
- `s3-atp-3victors-3vdev-use1-ds-collection-optimizer`
  - `delta_input/v1/sales_date={YYYYMMDD}/` -- Delta SWIA input (Parquet)
  - `ingest_ttl/v1/sales_date={YYYYMMDD}/airline={code}/` -- Ingest TTL (Parquet)
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
