"""Fixture — looks like a real generator. File lives under a
`market-level-generator/` folder so Pass 3 attributes to that stage."""

import boto3

def main():
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket="s3-atp-3victors3vprod-use1-anomaly-datasets",
        Key="market-level/v4/output.parquet",
    )
    # Also reads some S3
    df = pd.read_parquet("s3://s3-atp-3victors3vprod-use1-competitive-position/v2/")
    df.to_parquet("s3://s3-atp-3victors3vprod-use1-anomaly-datasets/market-level/v4/")
