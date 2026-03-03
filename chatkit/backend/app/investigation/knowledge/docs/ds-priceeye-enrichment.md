# ds-priceeye-enrichment

> Weekly pipeline that computes airline fare tax regression coefficients from raw shopping data and delivers results to American Airlines and a shared MySQL table.

> **Branch note**: The documented state reflects the `develop` branch. The `master`/`main` branch represents what is currently running in production — verify against master before treating this as the production reference.

---

## Architecture Overview

```
[EventBridge cron: every Tuesday 11:00 AM UTC]
       │
       ▼
[Step Function: taxregression-step-function]
       │
       ├─ 1 ──► [taxregression-calc-market-list]
       │              Reads: Redshift prod.common_output.common_output_format
       │              Writes: S3 .../tax_reg_market_list/v1/ → Glue: tax_reg_market_list_v1
       │
       ├─ 2 ──► [taxregression-calc-original]
       │              Reads: Redshift daily_representative_itinerary_v4 ∩ tax_reg_market_list_v1
       │              Writes: S3 .../tax_reg_raw/v1/ → Glue: tax_reg_raw_v1 (USD fares)
       │
       ├─ 3 ──► [taxregression-calc-non-usd]
       │              Reads: Glue tax_reg_raw_v1 (pos_exchange_rate ≠ 0,1)
       │              Writes: S3 .../tax_reg_raw/v1/ → Glue: tax_reg_raw_v1 (non-USD converted)
       │
       ├─ 4 ──► [taxregression-calc-coefficients]
       │              Reads: Glue tax_reg_raw_v1
       │              Writes: S3 .../tax_reg_output/v1/ → Glue: tax_reg_output_v1 (per-carrier)
       │
       ├─ 5 ──► [taxregression-calc-default-coefficients]
       │              Reads: Glue tax_reg_raw_v1
       │              Writes: S3 .../tax_reg_output/v1/ → Glue: tax_reg_output_v1 (fallback)
       │
       ├─ 6 ──► [taxregression-calc-mcla-com-raw]
       │              Reads: Redshift common_output (Volaris/Vivaaerobus carriers)
       │              Writes: S3 .../tax_reg_raw_com/v1/ → Glue: tax_reg_raw_com_v1
       │
       ├─ 7 ──► [taxregression-calc-mcla-com-output]
       │              Reads: Glue tax_reg_raw_com_v1
       │              Writes: S3 .../tax_reg_output_com/v1/ → Glue: tax_reg_output_com_v1
       │
       ├─ 8 ──► [taxregression-calc-generate-csv]
       │              Reads: Glue tax_reg_market_list_v1, tax_reg_output_v1, tax_reg_output_com_v1
       │              Writes: S3 .../client-aa/tax_reg/ (CSV for AA)
       │                      S3 .../ds-tax-regression/tax_reg_archive/ (dated archive)
       │
       └─ 9 ──► [taxregression-calc-mysql-refresh]
                      Reads: S3 tax_reg_output/v1/ + tax_reg_output_com/v1/ (parquet)
                      Writes: Aurora MySQL taxregression.tax_regression_v1 (overwrite + backup)
```

---

## Orchestration

### Step Function: taxregression-step-function

- **Trigger**: EventBridge rule `taxregression-stepfunction-task` — cron schedule every Tuesday at 11:00 AM UTC (`cron(0 11 ? * 3 *)`)
- **Pipeline** (sequential, each step must succeed before the next):
  1. `taxregression-calc-market-list`
  2. `taxregression-calc-original`
  3. `taxregression-calc-non-usd`
  4. `taxregression-calc-coefficients`
  5. `taxregression-calc-default-coefficients`
  6. `taxregression-calc-mcla-com-raw`
  7. `taxregression-calc-mcla-com-output`
  8. `taxregression-calc-generate-csv`
  9. `taxregression-calc-mysql-refresh`
- **On failure**: Any step failure routes to a `FailState` with error `GlueJobFailed`; no automatic retries (`MaxRetries: 0`)
- **Definition**: `source/deploy/definitions/taxregression-step-function.asl.json`

All jobs use Glue ETL (`glueetl`, Python 3, Glue 4.0) and load runtime configuration from `s3://config-server-{PROFILENAME}/default/tax-reg-config.properties`. The observation date used throughout is **T-2 days** (today minus 2 days).

---

## Components

_(Ordered by pipeline execution sequence.)_

---

### taxregression-calc-market-list

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 1
**Compute**: G.1X worker × 10, timeout 60 min
**Connection**: Redshift connection

**What it does**: Queries the last 10 days of Redshift `prod.common_output.common_output_format` to build a deduplicated market list — the set of active (customer, profile, POS, origin, destination, OD, currency) combinations with booking counts. Excludes internal/test customers (CH, Sanity, WN, QA, GJ, Advito, etc.) and non-standard sources (GDS, Points). Writes a single consolidated parquet file named `tax_reg_market_list.parquet` to S3 and registers a Glue partition.

**Input**:
- Redshift table: `prod.common_output.common_output_format` (last 10 days of `sales_date`)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_market_list/v1/{YYYY}/{MM}/{DD}/tax_reg_market_list.parquet`
- Glue table: `tax_reg_market_list_v1` in `glue-atp-3victors-{env}-use1-tax_reg_db`

**Table Schema** (`tax_reg_market_list_v1`):

| Column | Type |
|--------|------|
| customer | varchar(256) |
| profile | varchar(256) |
| pos | varchar(256) |
| origin | varchar(256) |
| destination | varchar(256) |
| od | varchar(256) |
| currency | varchar(256) |
| ct | bigint |

_Partition key: `sales_date` (bigint)_

---

### taxregression-calc-original

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 2
**Compute**: G.1X worker × 10, timeout 90 min
**Connection**: Redshift connection

**What it does**: Extracts raw fare observations from the Redshift `daily_representative_itinerary_v4` table, filtered to markets present in `tax_reg_market_list_v1` for the current sales date and to positive fares (`source IN ('ES', 'SS')`). Iterates over five cabin buckets (d→E, y→E, p→P, j→B, f→F) and writes one parquet partition per cabin/currency combination. This produces the primary USD-priced raw dataset used as input for regression.

**Input**:
- Redshift table: `prod.data_lakes.daily_representative_itinerary_v4` (observation_date = sales_date)
- Redshift table: `tax_reg.tax_reg_market_list_v1` (for market filtering)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_raw/v1/{YYYY}/{MM}/{DD}/currency=original/search_class={cls}/`
- Glue table: `tax_reg_raw_v1` in `glue-atp-3victors-{env}-use1-tax_reg_db`

**Table Schema** (`tax_reg_raw_v1`):

| Column | Type |
|--------|------|
| pos | varchar(10) |
| od | varchar(10) |
| is_one_way | int |
| search_class | varchar(10) |
| carrier | varchar(10) |
| pos_exchange_rate | double |
| pos_currency_code | varchar(3) |
| atp_exchange_rate | double |
| atp_currency_code | varchar(3) |
| current_currency | varchar(3) |
| nbr_outbound_stop | int |
| nbr_inbound_stop | int |
| price | double |
| tax | double |
| price_exc | double |
| q_surcharge | double |
| yqyr_surcharge | double |
| src | varchar(10) |
| zn_nbr | int |
| cbn | varchar(10) |
| currency | string |

_Partition key: `sales_date` (int)_

---

### taxregression-calc-non-usd

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 3
**Compute**: G.1X worker × 10, timeout 60 min
**Connection**: Redshift connection (reads via Redshift external schema)

**What it does**: Reads the USD-priced records from `tax_reg_raw_v1` (step 2 output) and converts prices/taxes into each market's local POS currency by dividing by `pos_exchange_rate`. Only processes records where `pos_exchange_rate NOT IN (0, 1)` — i.e., non-USD markets. Appends a `currency=usd` (local-converted) partition back into `tax_reg_raw_v1`, giving the regression two currency perspectives per market.

**Input**:
- Redshift external table: `tax_reg.tax_reg_raw_v1` (sales_date = current)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_raw/v1/{YYYY}/{MM}/{DD}/currency=usd/`
- Glue table: `tax_reg_raw_v1` (appended partition)

---

### taxregression-calc-coefficients

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 4
**Compute**: G.1X worker × 10, timeout 60 min
**Connection**: Redshift connection (reads via external schema)

**What it does**: Computes per-market linear regression coefficients relating total fare price (x) to pre-tax price (y = price_exc) from `tax_reg_raw_v1`. Calculates slope `m`, intercept `b`, R², and Pearson correlation for each `(pos, od, is_one_way, search_class, carrier, currency, nbr_outbound_stop, nbr_inbound_stop)` segment. Writes a single parquet file `tax_reg_coeff_output.parquet`. These are the specific-carrier, specific-class coefficients.

**Input**:
- Redshift external table: `tax_reg.tax_reg_raw_v1` (sales_date = current)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_output/v1/{YYYY}/{MM}/{DD}/tax_reg_coeff_output.parquet`
- Glue table: `tax_reg_output_v1` in `glue-atp-3victors-{env}-use1-tax_reg_db`

**Table Schema** (`tax_reg_output_v1`):

| Column | Type |
|--------|------|
| pos | string |
| od | string |
| is_one_way | int |
| search_class | string |
| carrier | string |
| currency | string |
| nbr_outbound_stop | int |
| nbr_inbound_stop | int |
| ct | bigint |
| minx | double |
| x_bar | double |
| maxx | double |
| miny | double |
| y_bar | double |
| maxy | double |
| m | double |
| b | double |
| r2 | double |
| correlation | double |
| added_at | timestamp |

_Partition key: `sales_date` (bigint)_

---

### taxregression-calc-default-coefficients

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 5
**Compute**: G.1X worker × 10, timeout 60 min
**Connection**: Redshift connection (reads via external schema)

**What it does**: Computes carrier-agnostic and search-class-agnostic ("wildcard") fallback regression coefficients from the same `tax_reg_raw_v1` data. Groups by `(pos, od, is_one_way, currency, nbr_outbound_stop, nbr_inbound_stop)` only — setting `search_class='*'` and `carrier='*'` — to produce default coefficients that apply when no carrier/class-specific coefficient is available. Appends (`mode=append`) to the same `tax_reg_output_v1` Glue table alongside the specific coefficients from step 4.

**Input**:
- Redshift external table: `tax_reg.tax_reg_raw_v1` (sales_date = current, carrier NOT IN ('O2'))

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_output/v1/{YYYY}/{MM}/{DD}/tax_reg_defa_coeff_output.parquet`
- Glue table: `tax_reg_output_v1` (appended)

---

### taxregression-calc-mcla-com-raw

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 6
**Compute**: G.1X worker × 10, timeout 60 min
**Connection**: Redshift connection

**What it does**: Extracts raw fare data for MCLA (multi-carrier low-cost airline) carriers — specifically Volaris (Y4, customer='AA') and Vivaaerobus (VB, customer='GJ') — from `common_output.common_output_format`. Maps fields to the standard raw regression schema (price_inc→price, preferred_currency_rate as exchange rate, etc.) and writes parquet to the `.com` raw table. Uses the last 7 days of data (`sales_date >= last_7`).

**Input**:
- Redshift table: `common_output.common_output_format` (filtered to VB/Y4 carriers, last 7 days)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_raw_com/v1/{YYYY}/{MM}/{DD}/`
- Glue table: `tax_reg_raw_com_v1` in `glue-atp-3victors-{env}-use1-tax_reg_db`

**Table Schema** (`tax_reg_raw_com_v1`):

| Column | Type |
|--------|------|
| pos | varchar(10) |
| od | varchar(10) |
| is_one_way | int |
| search_class | varchar(10) |
| carrier | varchar(10) |
| pos_exchange_rate | double |
| pos_currency_code | varchar(3) |
| atp_exchange_rate | double |
| atp_currency_code | varchar(3) |
| current_currency | varchar(3) |
| nbr_outbound_stop | int |
| nbr_inbound_stop | int |
| price | double |
| tax | double |
| price_exc | double |
| q_surcharge | double |
| yqyr_surcharge | double |
| src | varchar(10) |
| zn_nbr | int |
| cbn | varchar(10) |
| currency | varchar(10) |

_Partition key: `sales_date` (bigint)_

---

### taxregression-calc-mcla-com-output

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 7
**Compute**: G.1X worker × 10, timeout 60 min
**Connection**: Redshift connection (reads via external schema)

**What it does**: Computes regression coefficients for the MCLA carriers using the same linear regression logic as step 4, but operating on `tax_reg_raw_com_v1` instead of the main raw table. Groups by `(pos, od, is_one_way, search_class, carrier, currency, nbr_outbound_stop, nbr_inbound_stop)`. Outputs a single parquet file `tax_reg_mcla_coeff_output.parquet`.

**Input**:
- Redshift external table: `tax_reg.tax_reg_raw_com_v1` (sales_date = current)

**Output**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_output_com/v1/{YYYY}/{MM}/{DD}/tax_reg_mcla_coeff_output.parquet`
- Glue table: `tax_reg_output_com_v1` in `glue-atp-3victors-{env}-use1-tax_reg_db`

**Table Schema** (`tax_reg_output_com_v1`):

| Column | Type |
|--------|------|
| pos | varchar(256) |
| od | varchar(256) |
| is_one_way | int |
| search_class | varchar(256) |
| carrier | varchar(256) |
| currency | varchar(256) |
| nbr_outbound_stop | int |
| nbr_inbound_stop | int |
| ct | bigint |
| minx / x_bar / maxx | double |
| miny / y_bar / maxy | double |
| m | double |
| b | double |
| r2 | double |
| correlation | double |
| added_at | timestamp |

_Partition key: `sales_date` (bigint)_

---

### taxregression-calc-generate-csv

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 8
**Compute**: G.2X worker × 10, timeout 60 min
**Connection**: Redshift connection

**What it does**: Generates three CSV deliverables from the Redshift external tax_reg schema for the current sales_date:
1. `tax_reg_market_list_{sales_date}.csv` — the active market list (archive only)
2. `tax_regression_output.csv` / `tax_regression_output_{sales_date}.csv` — AA-filtered regression coefficients from `tax_reg_output_v1` (markets where customer='AA' in market list)
3. `tax_regression_output_com.csv` / `tax_regression_output_com{sales_date}.csv` — MCLA regression coefficients from `tax_reg_output_com_v1`

Writes the current-date CSVs (no suffix) to the live AA delivery bucket and all files (with date suffix) to an archive bucket.

**Input**:
- Redshift external tables: `tax_reg.tax_reg_market_list_v1`, `tax_reg.tax_reg_output_v1`, `tax_reg.tax_reg_output_com_v1`

**Output**:
- S3 (AA delivery): `s3://s3-atp-3victors-{env}-use1-client-aa/tax_reg/tax_regression_output.csv`
- S3 (AA delivery): `s3://s3-atp-3victors-{env}-use1-client-aa/tax_reg/tax_regression_output_com.csv`
- S3 (archive): `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_archive/tax_reg_market_list_{YYYYMMDD}.csv`
- S3 (archive): `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_archive/tax_regression_output_{YYYYMMDD}.csv`
- S3 (archive): `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_archive/tax_regression_output_com{YYYYMMDD}.csv`

---

### taxregression-calc-mysql-refresh

**Type**: AWS Glue Job (ETL)
**Trigger**: Step Function step 9 (final)
**Compute**: G.2X worker × 10, timeout 60 min
**Connection**: Aurora MySQL (`aurora-master-user_code` Secrets Manager secret)

**What it does**: Reads both parquet coefficient datasets from S3 (the standard regression output and the MCLA com output) for the current date, unions them, deduplicates on `(pos, od, is_one_way, search_class, carrier, currency, nbr_inbound_stop, nbr_outbound_stop)`, and writes the result to Aurora MySQL. First appends to the backup table (`tax_regression_v1_old`), then overwrites the live table (`tax_regression_v1`). Excludes Volaris (Y4) and Vivaaerobus (VB) carriers from the standard output before the union (those come from the MCLA output instead). Uses JDBC with `mysql-connector-j:8.4.0`.

**Input**:
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_output/v1/{YYYY}/{MM}/{DD}/`
- S3: `s3://s3-atp-3victors-{env}-use1-ds-tax-regression/tax_reg_output_com/v1/{YYYY}/{MM}/{DD}/`
- Secrets Manager: `aurora-master-user_code` (MySQL host, port, username, password)

**Output**:
- Aurora MySQL database `taxregression`, table `tax_regression_v1` (overwrite — live table)
- Aurora MySQL database `taxregression`, table `tax_regression_v1_old` (append — rolling backup)

---

## Scaffolded Components (Not Yet Implemented)

The `source/yqyr/` directory contains four placeholder component directories with only `.gitkeep` files — no source code or CloudFormation templates exist yet:

| Directory | Description |
|-----------|-------------|
| `yqyr-cache` | Future YQYR caching component |
| `yqyr-classification` | Future YQYR classification component |
| `yqyr-input-data` | Future YQYR input data component |
| `yqyr-regression` | Future YQYR regression component |

---

## Glue Databases

| Database | Tables |
|----------|--------|
| `glue-atp-3victors-{env}-use1-tax_reg_db` | `tax_reg_market_list_v1`, `tax_reg_raw_v1`, `tax_reg_raw_com_v1`, `tax_reg_output_v1`, `tax_reg_output_com_v1`, `tax_reg_aa_output_v1` |

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| Glue ETL Jobs | 9 |
| Step Functions | 1 |
| Glue Databases | 1 |
| Glue Tables | 6 |
| EventBridge Rules | 1 |
| Aurora MySQL Tables (destination) | 2 |

---

## Configuration

All runtime paths, table names, and bucket names are externalized in:

```
s3://config-server-{PROFILENAME}/default/tax-reg-config.properties
```

A reference copy lives at `docs/properties/tax-reg-config.properties`. Key config groups:

| Prefix | Controls |
|--------|----------|
| `tax_market_list_*` | Step 1 — market list S3 paths, Redshift input table, Glue destination |
| `tax_reg_original_*` | Step 2 — raw USD output paths and Redshift input tables |
| `tax_reg_non_usd_*` | Step 3 — non-USD raw output path |
| `tax_reg_coef_*` | Step 4 — coefficient output path |
| `tax_reg_defa_coef_*` | Step 5 — default coefficient output path |
| `tax_reg_calc_mcla_*` | Step 6 — MCLA raw paths and Redshift input |
| `tax_reg_calc_mcla_coef_*` | Step 7 — MCLA coefficient output path |
| `tax_reg_calc_*` (generate csv) | Step 8 — AA delivery bucket and archive bucket |
| `tax_reg_mysql_*` | Step 9 — S3 source paths, Aurora database/table names |

Glue scripts are deployed to `s3://s3-atp-3victors-{env}-use1-3v-glue-etl/<job-name>.py`.

---

## Redshift Schema Reference

The SQL DDL for creating the Redshift external schema (`tax_reg`) pointing at the Glue catalog is at:
`docs/sql/tax_reg_schema-3vprod.sql`
