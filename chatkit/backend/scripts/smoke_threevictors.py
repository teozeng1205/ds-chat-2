#!/usr/bin/env python3
"""Basic ds-threevictors connectivity smoke tests for 3VDEV-style profiles.

Usage:
  python scripts/smoke_threevictors.py --profile 3VDEV
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
from threevictors.dao import mysql_connector, redshift_connector
from threevictors.s3_util import s3_util


def bootstrap_creds(profile: str) -> None:
    proc = subprocess.run(
        ["granted", "credential-process", "--profile", profile, "--auto-login"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    os.environ["AWS_ACCESS_KEY_ID"] = payload["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = payload["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = payload["SessionToken"]
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


class _RedshiftReader(redshift_connector.RedshiftConnector):
    def get_properties_filename(self):
        return "database-analytics-redshift-serverless-reader.properties"


class _MySQLReader(mysql_connector.MySQLConnector):
    def get_properties_filename(self):
        return "database-priceeye-reader.properties"


def smoke_redshift() -> dict:
    reader = None
    try:
        reader = _RedshiftReader()
        with reader.get_connection().cursor() as cur:
            cur.execute("SELECT current_date AS current_date, current_timestamp AS current_ts LIMIT 1")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        frame = pd.DataFrame(rows, columns=cols)
        return {
            "ok": True,
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "sample": frame.head(1).to_dict(orient="records"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass


def smoke_mysql() -> dict:
    reader = None
    try:
        reader = _MySQLReader()
        with reader.get_connection().cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        frame = pd.DataFrame(rows, columns=cols)
        return {
            "ok": True,
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "sample": frame.head(1).to_dict(orient="records"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass


def smoke_s3() -> dict:
    try:
        client = s3_util.S3Util()
        response = client.s3_client.list_objects_v2(
            Bucket="s3-atp-3victors-3vdev-use1-collection-anomalies",
            Prefix="collection-customer/v1/",
            MaxKeys=5,
        )
        keys = [item.get("Key") for item in response.get("Contents", []) if item.get("Key")]
        return {
            "ok": True,
            "keys_found": len(keys),
            "sample_keys": keys,
            "is_truncated": bool(response.get("IsTruncated", False)),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="3VDEV", help="AWS profile for granted credential-process")
    args = parser.parse_args()

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "environment": "3VDEV",
    }

    try:
        bootstrap_creds(args.profile)
    except Exception as exc:
        output["bootstrap"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(output, indent=2, default=str))
        return 1

    output["bootstrap"] = {"ok": True}
    output["redshift"] = smoke_redshift()
    output["mysql"] = smoke_mysql()
    output["s3"] = smoke_s3()

    print(json.dumps(output, indent=2, default=str))
    passed = all(output[name].get("ok") for name in ("bootstrap", "redshift", "mysql", "s3"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
