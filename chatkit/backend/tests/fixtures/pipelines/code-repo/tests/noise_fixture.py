"""Fixture — lives under `tests/` so the whole file should be skipped."""

import boto3
s3 = boto3.client("s3")
# This s3 write should NOT be picked up because we skip `tests/` dirs
s3.put_object(Bucket="s3-atp-3victors3vprod-use1-should-not-appear", Key="v1/")
