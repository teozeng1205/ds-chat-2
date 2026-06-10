# Workflow: Sales POC Pipeline (input-file generation)

**Repo:** `ds-priceeye-data-collection` — `source/sales-poc-input-generator/` (input files) and
`source/sales-poc-market-generator/` (market data). **Env:** this is a POC pipeline that runs in
**GOLD/dev**, not prod — there is no `prod.sales_poc.*`. Query it via `local.sales_poc.*`
(analytics Glue), or the MySQL config via `federated_sales_poc.*` / `sales_poc_mysql.*`
(analytics) / `local.federated_sales_poc.*` (core). Glue DB: `…-sales_poc_db`.

## Stage 1 — Input-file generation  ·  `sales-poc-input-generator`

Entry: `src/main.py`. Component scripts: `create_carrier_input.py`, `create_customer_list.py`,
`populate_segment_data.py`, `populate_visual_defaults.py`, writer `src/dao/mysql_writer.py`.

### Reads — sales_poc MySQL config (for creating input files)
The generator reads/populates the `sales_poc` **MySQL** config tables and assembles them into the
carrier input requests:

| Table (`sales_poc.*` MySQL) | Used for |
|---|---|
| **`segment`** | segment definitions — `segmentName, customer, regionId, dateRangeId, carrierGroupId, cabinGroupId` (populated/read by `populate_segment_data.py`) |
| `region`, `geography`, `geography_entry` | market geography |
| `date_range` | depart/return date windows |
| `carrier_group`, `cabin_group`, `triptype_group` | carrier/cabin/trip-type groupings |
| `atpco_airlines`, `parent_airlines`, `new_customer_list` | airline + customer seed lists |
| `oag_valid_markets` | valid market filter |
| `input_request`, `output_status`, `visual_defaults` | request config + run/visual state |

**Reading the segments from MySQL:** `sales_poc.segment` is the segment table; `populate_segment_data.py`
`DELETE`s a customer's rows then re-`INSERT`s segment rows keyed by
`segmentName, customer, regionId, dateRangeId, carrierGroupId, cabinGroupId`. Other steps then
join `segment` to `region`/`date_range`/`carrier_group`/`cabin_group` to expand each segment into
concrete carrier input requests.

### Writes — the input files
`create_carrier_input.py` writes one parquet per carrier/day:
`s3://…-sales-poc/input_requests/v1/{year}/{month}/{day}/{carrier_code}/input_data.parquet`
→ Glue **`sales_poc.input_requests_v1`** (21 cols; `input_requests_v2` = 23 cols). These are the
"input files" PriceEye later collects against.

### Building segments FROM an input file (used in market-level analysis)
`populate_segment_data(carrier_code, input_df)` (called from `create_carrier_input.py:417`, defined in
`populate_segment_data.py`) **derives the sales_poc config from the input file itself** and cross-products it
into segment definitions:
1. Clears the carrier's existing `segment` / `region` / `geography` / `geography_entry` / `date_range` /
   `cabin_group` rows (all keyed by `customer = <carrier_code>`).
2. From `input_df`: builds `geography` (+`geography_entry`) from the distinct origin/destination
   `*_country_code`s; builds `region` from the distinct origin→destination country pairs (`"{orig}-{dest}"`);
   builds `date_range` and `cabin_group`.
3. Builds **`sales_poc.segment`** as the cross-product **region × date_range × cabin_group** (× "All Carriers"),
   naming each `"{regionName} | {dateRangeName} | All Carriers | {cabinGroupName}"`, inserting
   `segmentName, customer, regionId, dateRangeId, carrierGroupId, cabinGroupId`.

So segments are **built from the input file**, not hand-authored. These segment definitions
(`sales_poc.segment` / `analytics.segment`) drive the downstream **market-/segment-level analysis** and are read
by the alerts `palerts-generator` (`SELECT segmentId, segmentName FROM analytics.segment WHERE customer = …`,
`ds-priceeye-analytics/source/04-Alerts/palerts-generator/src/dao/priceeye_reader.py`).

## Stage 2 — Market data  ·  `sales-poc-market-generator`

Entry: `src/main.py`. Produces **`sales_poc.market_data_v1/v2`** and
**`sales_poc.missing_carrier_markets_v1`** (Glue `sales_poc_db`).

## Tables

| Table | Kind | Cols | Notes |
|---|---|---|---|
| `sales_poc.segment` (MySQL / `federated_sales_poc.segment`) | config (read) | 8 | segment defs; input to input-file generation |
| `sales_poc.input_requests_v1` / `_v2` | Glue output | 21 / 23 | the generated input files; S3 `input_requests/v1/<Y>/<M>/<D>/<carrier>/` |
| `sales_poc.market_data_v1` / `_v2` | Glue output | 19 / 37 | market data |
| `sales_poc.missing_carrier_markets_v1` | Glue output | 12 | markets with no carrier coverage |
| `sales_poc.*` other config (region, date_range, carrier_group, …) | MySQL | — | seeds joined to `segment` to build requests |

Partition registration (`partition_details`, GOLD bucket): `input_requests_v1`, `market_data_v1`,
`missing_carrier_markets_v1` with `partition_order = sales_date,customer`.

## Health / debugging signals
- Empty `sales_poc.input_request` in DEV is often a deliberate clean (only `priceeye.input_request`
  was meant to be cleared) — confirm env before assuming a generator failure.
- No input files for a carrier/day: check the `sales-poc-input-generator` run and that `segment`
  (and its `region`/`date_range`/`carrier_group`/`cabin_group` refs) is populated for that customer.
- `sales_poc.*` is GOLD/dev — do not look for it under `prod.*`.
