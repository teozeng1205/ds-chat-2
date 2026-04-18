---
name: python_venv
description: Pre-installed Python venv libraries (pandas, boto3, threevictors) and how to use them for direct data access from bash.
keywords: [python, venv, pandas, numpy, pyarrow, matplotlib, seaborn, boto3, duckdb, threevictors, redshift_connector, mysql_connector, s3_util]
---

## Python environment

The bash session has the ds-chat-2 backend venv pre-activated. `python3` gives you:
pandas, numpy, pyarrow, matplotlib, seaborn, boto3, duckdb, **threevictors**.

**threevictors** — ATPCO's internal data access library (same connectors as execute_sql / fetch_s3):

```python
# Redshift (requires valid AWS credentials, same as execute_sql)
from threevictors.dao import redshift_connector
reader = redshift_connector.RedshiftConnector()   # auto-detects analytics vs core

# Or use the project wrappers for named clusters:
import sys; sys.path.insert(0, '/path/to/chatkit/backend')
from app.investigation.datasources import AnalyticsRedshiftReader, CoreRedshiftReader
df = AnalyticsRedshiftReader().query(
    "SELECT * FROM prod.analytics.market_level_anomalies "
    "WHERE sales_date = 20260310 AND customer = 'B6' LIMIT 100"
)

# MySQL
from app.investigation.datasources import PriceEyeMySQLReader
df = PriceEyeMySQLReader().query("SELECT * FROM priceeye.site LIMIT 50")

# S3 (direct, beyond what fetch_s3 tool handles)
from threevictors.s3_util import s3_util
s3 = s3_util.S3Util()
keys = s3.find_keys_with_prefix(
    's3-atp-3victors-3vdev-use1-anomaly-datasets',
    'market-level/v4/B6/2026/03/',
)
```

Requires valid AWS credentials. If `execute_sql` works, `threevictors` will too.
