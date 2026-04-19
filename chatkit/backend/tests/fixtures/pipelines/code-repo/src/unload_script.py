"""Fixture — Redshift UNLOAD + COPY patterns. Attribution falls back to the
repo because no known stage is in the path."""

sql = """
UNLOAD ('SELECT * FROM analytics.market_level_anomalies_v4')
  TO 's3://s3-atp-3victors3vprod-use1-anomaly-datasets/market-level/unload/'
  IAM_ROLE 'arn:aws:iam::123:role/redshift';

COPY analytics.competitive_position
  FROM 's3://s3-atp-3victors3vprod-use1-competitive-position/v2/'
  IAM_ROLE 'arn:aws:iam::123:role/redshift'
  FORMAT AS PARQUET;
"""

# Also some glue-assets noise that should NOT become an edge
temp = "s3://aws-glue-assets-123456789012-us-east-1/temporary/"
cfg  = "s3://config-server-3vprod/default/app.properties"
