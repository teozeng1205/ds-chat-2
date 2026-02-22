"""Lazy wrappers around ds-threevictors connectors (Redshift, MySQL, S3)."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any

import pandas as pd


class ThreeVictorsDependencyError(RuntimeError):
    """Raised when ds-threevictors is unavailable in the current Python env."""


def _import_module(path: str):
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as exc:
        raise ThreeVictorsDependencyError(
            "Missing ds-threevictors dependency. Install/enable `threevictors` in backend runtime."
        ) from exc


def _resolve_redshift_connector_cls():
    module = _import_module("threevictors.dao.redshift_connector")
    connector_cls = getattr(module, "RedshiftConnector", None)
    if connector_cls is None or not inspect.isclass(connector_cls):
        raise ThreeVictorsDependencyError("Unable to locate RedshiftConnector in threevictors.dao.redshift_connector")
    return connector_cls


def _resolve_mysql_connector_cls():
    dao_module = _import_module("threevictors.dao")
    if not hasattr(dao_module, "__path__"):
        raise ThreeVictorsDependencyError("Unable to inspect threevictors.dao for MySQL connectors")

    for _, module_name, _ in pkgutil.iter_modules(dao_module.__path__):
        if "mysql" not in module_name.lower():
            continue
        module = _import_module(f"threevictors.dao.{module_name}")
        for attr_name in dir(module):
            candidate = getattr(module, attr_name)
            if not inspect.isclass(candidate):
                continue
            if "connector" not in attr_name.lower() or "mysql" not in attr_name.lower():
                continue
            if hasattr(candidate, "get_connection"):
                return candidate

    raise ThreeVictorsDependencyError("Unable to locate a MySQL connector class in threevictors.dao.*")


def _resolve_s3_util_cls():
    module = _import_module("threevictors.s3_util.s3_util")
    util_cls = getattr(module, "S3Util", None)
    if util_cls is None or not inspect.isclass(util_cls):
        raise ThreeVictorsDependencyError("Unable to locate S3Util in threevictors.s3_util.s3_util")
    return util_cls


@dataclass
class ThreeVictorsConfig:
    redshift_properties: str = "database-analytics-redshift-serverless-reader.properties"
    mysql_properties: str = "database-priceeye-reader.properties"
    environment: str = "3VDEV"


class ThreeVictorsClient:
    """Simple query/file facade that enforces ds-threevictors usage."""

    def __init__(self, config: ThreeVictorsConfig | None = None):
        self.config = config or ThreeVictorsConfig()

    def _new_redshift_reader(self):
        connector_cls = _resolve_redshift_connector_cls()
        properties_filename = self.config.redshift_properties

        class _Reader(connector_cls):
            def get_properties_filename(self):
                return properties_filename

        return _Reader()

    def _new_mysql_reader(self):
        connector_cls = _resolve_mysql_connector_cls()
        properties_filename = self.config.mysql_properties

        class _Reader(connector_cls):
            def get_properties_filename(self):
                return properties_filename

        return _Reader()

    def _query_df(self, reader: Any, query: str) -> pd.DataFrame:
        with reader.get_connection().cursor() as cursor:
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description] if cursor.description else []
            records = cursor.fetchall() if cursor.description else []
            return pd.DataFrame(records, columns=colnames)

    def query_redshift(self, query: str) -> pd.DataFrame:
        reader = self._new_redshift_reader()
        try:
            return self._query_df(reader, query)
        finally:
            close_fn = getattr(reader, "close", None)
            if callable(close_fn):
                close_fn()

    def query_mysql(self, query: str) -> pd.DataFrame:
        reader = self._new_mysql_reader()
        try:
            return self._query_df(reader, query)
        finally:
            close_fn = getattr(reader, "close", None)
            if callable(close_fn):
                close_fn()

    def mysql_table_columns(self, table_name: str) -> list[str]:
        try:
            frame = self.query_mysql(f"SELECT * FROM {table_name} LIMIT 0")
            return [str(c) for c in frame.columns]
        except Exception:
            try:
                desc = self.query_mysql(f"DESCRIBE {table_name}")
                if "Field" in desc.columns:
                    return [str(v) for v in desc["Field"].tolist()]
            except Exception:
                return []
        return []

    def new_s3_client(self):
        util_cls = _resolve_s3_util_cls()
        return util_cls()

    def list_s3_keys(self, bucket: str, prefix: str) -> list[str]:
        client = self.new_s3_client()
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = client.s3_client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                key = item.get("Key")
                if key:
                    keys.append(str(key))
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        return sorted(keys)

    def read_s3_object_bytes(self, bucket: str, key: str) -> bytes:
        client = self.new_s3_client()

        # Prefer direct boto body for binary-safe parquet reads.
        try:
            response = client.s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read()
            if isinstance(body, bytes):
                return body
        except Exception:
            pass

        payload = client.get_object(bucket, key)
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return payload.encode("utf-8")
        return bytes(payload)

    def read_s3_table(self, bucket: str, key: str, format_hint: str = "auto") -> pd.DataFrame:
        blob = self.read_s3_object_bytes(bucket, key)
        lower = key.lower()
        chosen = format_hint.lower()
        if chosen == "auto":
            if lower.endswith(".parquet"):
                chosen = "parquet"
            elif lower.endswith(".csv"):
                chosen = "csv"
            else:
                chosen = "csv"

        if chosen == "parquet":
            return pd.read_parquet(BytesIO(blob))

        text = blob.decode("utf-8", errors="replace")
        return pd.read_csv(StringIO(text))
