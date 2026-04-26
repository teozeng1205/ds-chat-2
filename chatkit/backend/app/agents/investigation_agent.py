"""Investigation agent builder with rich domain instructions."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "investigation" / "knowledge"


def _load_common_table_metadata() -> str:
    """Load table metadata and emit a compact schema summary (grouped by datasource + schema)."""
    path = KNOWLEDGE_ROOT / "common_table_live_metadata.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    tables = payload.get("tables", []) if isinstance(payload, dict) else []

    # Group by datasource → schema → count
    from collections import defaultdict
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for table in tables:
        if not isinstance(table, dict):
            continue
        if table.get("status") == "error" or not table.get("table_name"):
            continue
        ds = table.get("datasource", "unknown")
        name = table.get("table_name", "")
        parts = name.split(".")
        if len(parts) >= 3:
            schema = parts[1]
        elif len(parts) == 2:
            schema = parts[0]
        else:
            schema = "other"
        groups[ds][schema] += 1

    lines: list[str] = []
    for ds, schemas in sorted(groups.items()):
        schema_parts = ", ".join(f"{s} ({c})" for s, c in sorted(schemas.items()))
        lines.append(f"- **{ds}**: {schema_parts}")
    if lines:
        lines.append("Use `inspect_table_metadata(table_name)` for column details, partitions, and freshness.")
        lines.append("Use `search_kb` for table-level docs and query examples.")
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
    scenarios_start = next(
        (i for i, line in enumerate(lines) if "## Common Investigation Scenarios" in line),
        None,
    )
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
   Tables: prod.analytics.* (anomalies, scoring, pax_midt, oag_score_v2, revenue_score_v1), prod.common_output.*, prod.data_lakes.*, prod.flight_summary.*, prod.midt_external.*, prod.federated_metadata.*, prod.federated_priceeye.*, prod.billing.*, prod.tax_reg.*, prod.priceeye_output.*
2. **redshift_core** -- Core Redshift serverless cluster.
   Tables: prod.monitoring.* (combined_audit, provider_combined_audit), prod.site_metrics.* (capacity_final, cache_metrics_v1, retry_metrics_v1, import_metrics_v1), prod.scheduling.*
   Note: `local.*` schemas exist as DEV copies of the above but should never be the default — only use them if the user explicitly asks for dev/local/staging data.
3. **mysql_priceeye** -- MySQL PriceEye database.
   Tables: priceeye.* (customer_defaults, site_hierarchy, transaction_rates, site, provider, customer), sales_poc.*, taxregression.*""",

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
- **DEFAULT TO `prod.*` TABLES.** Always prefer the `prod.*` version of any table unless the user explicitly asks for dev, local, or staging data. Never silently fall back to `local.*` without telling the user.
- **Table namespace tiers — know which to use:**
  - **Tier 1 — `prod.*` (DEFAULT — always production data):** Use these unless the user says otherwise.
    Examples: `prod.monitoring.*`, `prod.common_output.*`, `prod.analytics.*`, `prod.data_lakes.*`,
    `prod.flight_summary.*`, `prod.midt_external.*`, `prod.federated_metadata.*`, `prod.federated_priceeye.*`,
    `prod.billing.*`, `prod.priceeye_output.*`, `prod.site_metrics.*`, `prod.tax_reg.*`.
    `local.*` equivalents exist but return dev data — only use them when the user explicitly asks.
- **Single statement only.** No semicolons mid-query.
- The system automatically clamps LIMIT to 120,000 rows max.""",

        # ── Domain knowledge lookup ──
        """## Domain Knowledge Lookup

**Route knowledge questions as follows:**

- **"How does X work?" / "What does Y pipeline do?" / "Which table has Z?"** →
  Use **both** the KB and the real codebase. Do not answer these from memory and do not stop at `search_kb`.
  1. `search_kb("{topic}")` — get the doc snippet, `document_hints`, and related tables.
  2. **Verify against the actual repo/doc checkout.** Use the `document_hints` source to identify the likely repo, run `bash("ls ~/git/{repo}/")` to confirm it exists, then `read_file` the key entry-point files and/or `read_file("~/git/documentations/{source}")` for the full doc.
  3. In the final answer, clearly separate **KB/documentation guidance** from **code-verified facts**. Include at least one concrete implementation detail from the repo when a matching repo is available.
  4. If no repo can be identified or the repo is missing locally, say so explicitly and fall back to the KB/doc answer rather than pretending the code was checked.

- **When the user asks about a specific named component** — a specific pipeline, service, job, scheduler, or process (e.g. "auto-scheduler", "dedup pipeline", "anomaly detection job", "schedule-cutover", "preemptive polling") — a KB snippet alone is NOT enough. Do all three steps:
  1. `search_kb("{component name}")` — get the doc snippet and `document_hints`.
  2. **Go to the actual codebase.** From the `document_hints` source field (e.g. `priceeye-scheduling.md` → repo `priceeye-scheduling`), run `bash("ls ~/git/priceeye-scheduling/")` to confirm it's cloned, then `read_file` the key entry-point files to show real class names, method names, SQS queue names, Lambda handlers, Step Function names. Do NOT just paraphrase the wiki doc — show actual code.
  3. **Surface related tables.** From `candidate_tables` / `table_hints`, name the relevant Redshift tables and offer to run a live query (latest partition, row counts) so the user sees real data tied to the component.

- **"Show me the code for X" / "Where is Y implemented?"** → same three steps above, with deeper `read_file` into source files.

**`search_kb` response fields:**
- `candidate_tables` — matching table names
- `table_hints` — table metadata with partition info and query examples
- `document_hints` — list of `{source: "priceeye-scheduling.md", snippet: "...relevant excerpt..."}`.
  Read the full doc with `read_file("~/git/documentations/{source}")` if you need more context.

**Escalation order for specific-component questions:**
1. `search_kb` — doc snippets + table hints
2. `read_file("~/git/documentations/{source}")` — full wiki doc
3. `bash("ls ~/git/{repo}/")` → `read_file` or `bash grep` — actual source code, class/method names
4. `execute_sql` — live table data tied to the component""",

        # ── Investigation patterns (indexed in KB, not inlined here) ──
        """## Investigation Patterns

The catalog of "which table answers which question" (every concept → table
/ partition / key-columns mapping, plus the full 5-step fallback strategy)
is indexed in the KB as `investigation_patterns.md`. **Call `search_kb`
BEFORE naming a table from memory** — the KB is the source of truth; your
training data is not. Examples:

    search_kb("site issues table")
    search_kb("market anomalies partition columns")
    search_kb("billing request counts table")

If `search_kb` returns no relevant doc and you truly need a table you
don't know, run a `SELECT DISTINCT table_schema, table_name FROM
svv_columns WHERE table_name LIKE '%...%'` discovery query rather than
guessing.

Fallback chain when a query returns 0 rows:
  0. Discover tables with `svv_columns` / `information_schema.tables`.
  1. Verify partitions with `inspect_table(name)`.
  2. Try adjacent versions (_v4 → _v3 → _v2).
  3. Check the S3 mirror (call `search_kb("s3 <concept>")`).
  4. Only fall back to `local.*` if the user explicitly asked for dev.
  5. If nothing works, report clearly what was tried and why.""",

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
- **prod.monitoring.combined_audit / provider_combined_audit** → produced by ds-internal-monitoring (dedup pipeline, runs hourly)
- **prod.billing.customer_daily_requests_v1/v2/v3** → produced by ds-customer-monitoring (primary billing source)
- **prod.analytics.market_level_anomalies_v3** → produced by ds-priceeye-analytics market-level-generator (22-day rolling model)
- **prod.analytics.segment_level_anomalies_v3** → produced by ds-priceeye-analytics segment-level-generator
- **prod.analytics.market_level_anomalies / segment_level_anomalies** → competitive analysis tables
- **prod.site_metrics.capacity_final / cache_metrics_v1 / retry_metrics_v1** → produced by ds-priceeye-data-collection site-metrics lambdas
- **prod.common_output.common_output_format** → produced by priceeye-v2 (raw) then normalized by priceeye-analytics DCO Spark job
- **prod.tax_reg.tax_reg_output_v1** → produced by ds-priceeye-enrichment (runs every Tuesday)
- **prod.analytics.pax_midt** → produced by ds-priceeye-analytics pax-midt process
- **prod.analytics.revenue_score_v1** → produced by ds-priceeye-analytics revenue-score process
- **prod.analytics.oag_score_v2** → produced by ds-priceeye-analytics oag-score process

{system_scenarios}""" if system_scenarios else "",

        # ── S3 data reference (indexed in KB, not inlined here) ──
        """## S3 Data Reference

The full S3 bucket + prefix catalog (one entry per known purpose:
collection anomalies, DCO, anomaly datasets v4/v3, competitive
position, pe-common-output, etc.) plus the Redshift → S3-mirror lookup
are indexed in the KB as `s3_buckets.md`. **Call `search_kb` for bucket
/ prefix questions** — do not list buckets from memory, you will
hallucinate names that don't exist. Examples:

    search_kb("what s3 buckets")
    search_kb("market anomaly s3 bucket prefix")
    search_kb("dco derived common output s3 path")
    search_kb("competitive position s3")

Bucket naming convention: `s3-atp-3victors-{env}-use1-{purpose}`.
Default `{env}` = `3vprod` (process runs on 3VDEV creds with
cross-account read into 3VPROD). Swap to `3vdev` only when the user
explicitly asks for dev.

`fetch_s3` reads CSV, Parquet, and JSONL automatically.""",
    ]

    return "\n\n".join(section for section in sections if section)
