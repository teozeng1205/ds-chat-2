# ds-threevictors

> A shared Python utility library providing common AWS integrations — S3, Glue, Secrets Manager, EventBridge, MySQL, and Redshift — for use across Three Victors analytics services.

> **Production branch**: `master` (this document was generated from the `develop` branch; the documented state may differ slightly from production)

---

## Overview

`ds-threevictors` is a pip-installable Python package (`threevictors`) that centralises boilerplate AWS client code so that individual analytics components don't have to re-implement it. It is not a pipeline itself — it is a dependency consumed by other repos (ECS tasks, Lambda functions, Glue jobs) that run the actual data pipelines.

The library provides:
- A config-server pattern for reading `.properties` files stored in S3
- Authenticated database connectivity to MySQL and Redshift
- Helpers for S3, Glue partition management, Secrets Manager, STS, and EventBridge

---

## Package Structure

```
threevictors/                  ← installable package root
├── config_reader/             ← reads .properties files from S3 config bucket
├── dao/
│   ├── mysql_connector.py     ← abstract base for MySQL connections
│   └── redshift_connector.py  ← abstract base for Redshift connections
├── glue_util/                 ← AWS Glue partition management
├── notification_util/         ← EventBridge event publishing
├── s3_util/                   ← S3 get / upload / delete helpers
├── secrets_manager/           ← AWS Secrets Manager retrieval
└── sts_util/                  ← AWS STS caller identity
```

---

## Dependency Graph

```
Consumer Service (ECS / Lambda / Glue Job)
        │
        ├── ConfigReader ──────────────────► S3: config-server-{env}/default/*.properties
        │       │                                       (macro expansion via macros.properties)
        │       └── S3Util (internal)
        │
        ├── MySQLConnector (abstract) ──────► ConfigReader → database-*.properties
        │                                   └── SecretsManager (if 'authentication' key present)
        │
        ├── RedshiftConnector (abstract) ───► ConfigReader → database-*.properties
        │                                   └── SecretsManager (if 'authentication' key present)
        │
        ├── GlueUtil ──────────────────────► AWS Glue API (get table schema, create partition)
        │
        ├── NotificationPublisher ─────────► AWS EventBridge (put_events, up to 255 KB, 3 retries)
        │
        ├── S3Util ────────────────────────► AWS S3 (get/upload/delete objects, list buckets)
        │
        ├── SecretsManager ────────────────► AWS Secrets Manager (get_secret_value)
        │
        └── StsUtil ───────────────────────► AWS STS (get_caller_identity)
```

---

## Modules

### `config_reader` — Configuration Reader

**Class**: `ConfigReader`

**What it does**: Locates the environment's S3 configuration bucket (any bucket prefixed with `config-server-`) and reads `.properties` files from the `default/` prefix. Before returning a parsed properties dict, it resolves `${MACRO_NAME}` placeholders using values from `macros.properties` in the same bucket. Macros are cached for 60 seconds before a refresh. This gives all consuming services a single, environment-aware source of truth for connection strings, feature flags, and other configuration.

**Dependencies**: `s3_util.S3Util`

**Input**:
- S3: `config-server-{env}/default/<properties-file>.properties`
- S3: `config-server-{env}/default/macros.properties` (for macro substitution)

**Output**: `dict` of key/value pairs from the requested properties file, with macros expanded.

**Key methods**:

| Method | Description |
|--------|-------------|
| `read_properties(properties_file_name)` | Returns parsed dict from a `.properties` file in the config bucket |
| `replace_macros(file_content, macros)` | Substitutes `${MACRO}` tokens; refreshes macro cache if stale (>60 s) |
| `find_configuration_bucket()` | Lists all S3 buckets with prefix `config-server-`; exits on failure |

---

### `dao.mysql_connector` — MySQL Connector

**Class**: `MySQLConnector` (abstract base class)

**What it does**: Provides a reusable MySQL connection lifecycle for consuming services. Subclasses implement `get_properties_filename()` to point at a `.properties` file (e.g., `database-priceeye-reader.properties`) in the config server. On init, it reads host, port, database name, and credentials — fetching credentials from Secrets Manager when an `authentication` key is present in the properties file; otherwise reading `username`/`password` directly.

**Dependencies**: `config_reader.ConfigReader`, `secrets_manager.SecretsManager`, `mysql-connector-python`

**Input**:
- Config: `.properties` file referenced by the subclass (e.g., `database-priceeye-reader.properties`)
- Secrets Manager: secret named by the `authentication` property (format: `user:password`)

**Key methods**:

| Method | Description |
|--------|-------------|
| `get_properties_filename()` | Abstract — subclass must implement; returns the `.properties` file name |
| `get_connection()` | Returns the active `mysql.connector.connection` object |
| `close()` | Closes the connection |

**Usage pattern**:
```python
class MyDAO(MySQLConnector):
    def get_properties_filename(self):
        return "database-priceeye-reader.properties"
```

---

### `dao.redshift_connector` — Redshift Connector

**Class**: `RedshiftConnector` (abstract base class)

**What it does**: Identical pattern to `MySQLConnector` but targets Amazon Redshift (Serverless or provisioned). Reads connection config from the config server and resolves credentials via Secrets Manager when the `authentication` key is present.

**Dependencies**: `config_reader.ConfigReader`, `secrets_manager.SecretsManager`, `redshift-connector`

**Input**:
- Config: `.properties` file referenced by the subclass (e.g., `database-analytics-redshift-serverless-writer.properties`)
- Secrets Manager: secret named by the `authentication` property (format: `user:password`)

**Key methods**:

| Method | Description |
|--------|-------------|
| `get_properties_filename()` | Abstract — subclass must implement |
| `get_connection()` | Returns the active Redshift connection object |
| `close()` | Closes the connection |

---

### `glue_util` — Glue Partition Manager

**Class**: `GlueUtil`

**What it does**: Manages AWS Glue table partitions on behalf of analytics pipeline components. After a component writes new Parquet data to S3, it calls `add_partition` to register the new partition in the Glue catalog so it is immediately queryable by Athena. Handles the `AlreadyExistsException` gracefully (logs and continues). Internally derives the partition StorageDescriptor from the existing table schema, so callers only need to supply partition values and the S3 path.

**Dependencies**: `boto3` (Glue client)

**Key methods**:

| Method | Description |
|--------|-------------|
| `get_current_schema(database, table)` | Fetches table metadata (formats, location, SerdeInfo, partition keys) from Glue |
| `add_partition(database, table, partition_path, partition_values)` | Registers a new partition; silently skips if partition already exists |
| `generate_partition_input(table_data, partition_location, partition_values)` | Builds a single partition input dict from table metadata |
| `generate_partition_input_list(...)` | Wraps `generate_partition_input` in a list (batch variant) |

**Example** (from tests):
```python
glue = GlueUtil()
glue.add_partition(
    "glue-atp-3victors-{env}-use1-analytics_db",
    "competitive_position",
    "s3://s3-atp-3victors-{env}-use1-competitive-position/v1/AA/2025/02/11/",
    ["20250211", "AA"]
)
```

---

### `notification_util` — EventBridge Notification Publisher

**Class**: `NotificationPublisher`

**What it does**: Publishes structured JSON events to an AWS EventBridge custom bus. Enforces a 255 KB per-entry size limit (dropping events that exceed it) and retries up to 3 times with a 10-second backoff on failure. Uses a fluent builder pattern (`with_bus`, `with_source`, `with_detail_type`) so callers configure the publisher once and reuse it across multiple `publish()` calls.

**Dependencies**: `boto3` (EventBridge client)

**Key methods**:

| Method | Description |
|--------|-------------|
| `with_bus(bus)` | Sets the EventBridge bus name |
| `with_source(source)` | Sets the event source string |
| `with_detail_type(detail_type)` | Sets the `detail-type` for event routing |
| `publish(event)` | Publishes a dict as a JSON event using the configured bus/source/detail-type |
| `publish_specific(event, detail_type, bus, source)` | One-off publish with explicit routing params |
| `send_events(entries)` | Low-level send with retry logic (up to 3 attempts, 10 s delay) |
| `get_event_request_entry_size(entry)` | Static — computes entry byte size per EventBridge spec |

**Known EventBridge buses / sources** (from tests and cross-repo usage):
- Bus: `data-pipeline`
- Source examples: `threevictors.ecs.analytics`
- Detail-type examples: `MarketLevel`

**Usage pattern**:
```python
publisher = (NotificationPublisher()
    .with_bus("data-pipeline")
    .with_source("threevictors.ecs.analytics")
    .with_detail_type("MarketLevel"))

publisher.publish({"task": "completed", "date": "20250211"})
```

---

### `s3_util` — S3 Utilities

**Class**: `S3Util`

**What it does**: Thin wrapper around the boto3 S3 client, providing null-safe helpers for the most common S3 operations used by analytics components. Used internally by `ConfigReader` to fetch properties files, and directly by pipeline components to read/write data files.

**Dependencies**: `boto3` (S3 client)

**Key methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_object` | `(bucket, key) → str` | Reads and UTF-8-decodes an S3 object |
| `upload_object` | `(bucket, key, content: str\|bytes)` | Writes a string or bytes object via `put_object` |
| `upload_object_bytes` | `(bucket, key, data)` | Streams bytes via `upload_fileobj` |
| `upload_file` | `(bucket, key, file_name)` | Uploads a local file path |
| `delete_object` | `(bucket, key)` | Deletes an S3 object |
| `find_buckets_with_prefix` | `(prefix) → list[str]` | Lists bucket names matching a prefix |

---

### `secrets_manager` — Secrets Manager Client

**Class**: `SecretsManager`

**What it does**: Retrieves secrets from AWS Secrets Manager with structured error handling for all common error codes (`ResourceNotFoundException`, `InvalidRequestException`, `DecryptionFailure`, etc.). Returns either the secret string or binary value.

**Dependencies**: `boto3` (Secrets Manager client)

**Key methods**:

| Method | Description |
|--------|-------------|
| `get_secret(secret)` | Returns the secret value as a string or bytes |
| `get_secret_dict(secret)` | Parses the secret string as JSON and returns a dict |

---

### `sts_util` — STS Utilities

**Class**: `StsUtil`

**What it does**: Minimal wrapper around AWS STS for retrieving the current caller identity. Used by other components to dynamically determine the AWS account ID at runtime.

**Dependencies**: `boto3` (STS client)

**Key methods**:

| Method | Description |
|--------|-------------|
| `get_account_id()` | Returns the 12-digit AWS account ID string |
| `get_caller_identity()` | Returns the full STS `GetCallerIdentity` response dict |

---

## Installation & Distribution

The package is built with `setuptools` and distributed as `ds-threevictors` via the internal PyPI or direct pip install:

```
pip install ds-threevictors
```

**Runtime dependencies** (`requirements.txt`):

| Package | Purpose |
|---------|---------|
| `boto3` | All AWS service clients (S3, Glue, EventBridge, STS, Secrets Manager) |
| `redshift-connector` | Redshift database connectivity |
| `mysql-connector-python` | MySQL database connectivity |

**Python**: requires `>= 3.10`

**Current version**: `1.6+snapshot` (develop branch)

---

## Config Server Pattern

All connection configuration lives in an S3 bucket named `config-server-{env}` under the `default/` prefix. Each consuming service references a named `.properties` file. Macro tokens (`${KEY}`) in property values are resolved against `default/macros.properties`.

```
config-server-{env}/
└── default/
    ├── macros.properties                               ← global macro definitions
    ├── database-priceeye-reader.properties             ← MySQL connection for PriceEye reader
    ├── database-analytics-redshift-serverless-writer.properties  ← Redshift writer
    ├── PEPartitionRuleUpdaterLambda.properties         ← Lambda-specific config
    └── ...                                             ← one file per consuming service
```

A typical `.properties` file contains:

```properties
database.host=${MYSQL_HOST}
database.port=3306
database.name=priceeye
authentication=secret/priceeye-reader-credentials   # Secrets Manager secret ID
```

---

## Known Glue Infrastructure (from cross-repo usage)

| Database | Example Table | S3 Location |
|----------|--------------|-------------|
| `glue-atp-3victors-{env}-use1-analytics_db` | `competitive_position` | `s3://s3-atp-3victors-{env}-use1-competitive-position/v1/` |

---

## Testing

Tests live in `tests/` and are integration tests that require live AWS credentials and the target environment to be accessible. Run with:

```bash
pytest tests/ -v
```

Logging is configured via `pytest.ini` to emit `INFO`-level output to the console.

| Test file | What it exercises |
|-----------|-----------------|
| `test_s3_util.py` | `S3Util.get_object` against the dev config server bucket |
| `test_glue_util.py` | `GlueUtil.add_partition` against the dev Glue catalog |
| `test_notification_util.py` | `NotificationPublisher.publish` to the `data-pipeline` bus |
| `test_mysql_connector.py` | Full MySQL connection lifecycle via `database-priceeye-reader.properties` |
| `test_redshift_connector.py` | Full Redshift connection lifecycle via the analytics serverless writer properties |
