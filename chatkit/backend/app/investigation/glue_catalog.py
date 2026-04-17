"""Live Glue Data Catalog client — a small, stateless wrapper.

Exposes `get_table`, `get_partitions`, `list_tables`, `list_databases`,
and `discover_table` (search across all databases by table name).

Keeps the surface tiny so it can later replace the stale 2 MB JSON
snapshot as the source of truth for table metadata. All operations are
read-only.

Not wired into `inspect_table` yet — additive module.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)


_TABLE_REF = re.compile(r"^\s*(?:([\w-]+)\.)?([\w-]+)\s*$")


@dataclass(frozen=True)
class GlueColumn:
    name: str
    type: str
    comment: Optional[str] = None


@dataclass(frozen=True)
class GlueTable:
    database: str
    name: str
    owner: Optional[str]
    location: Optional[str]
    table_type: Optional[str]
    parameters: dict[str, Any]
    columns: tuple[GlueColumn, ...]
    partition_keys: tuple[GlueColumn, ...]
    created: Optional[str]
    updated: Optional[str]

    @property
    def qualified(self) -> str:
        return f"{self.database}.{self.name}"

    @property
    def partition_key_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.partition_keys)


@dataclass
class GluePartition:
    values: tuple[str, ...]
    location: Optional[str]
    created: Optional[str]
    updated: Optional[str]
    parameters: dict[str, Any] = field(default_factory=dict)


# ── Client ──


class GlueCatalog:
    """Thin, lazy boto3-backed Glue catalog client.

    A single client is reused across calls; boto3 clients are
    thread-safe per the AWS docs. Region defaults to whatever the
    session resolves (us-east-1 in 3VDEV).
    """

    def __init__(self, region_name: str | None = None, session: Any | None = None) -> None:
        self._region = region_name
        self._session = session
        self._client: Any | None = None
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                import boto3  # lazy — only required when someone actually calls Glue
                if self._session is not None:
                    self._client = self._session.client("glue", region_name=self._region)
                else:
                    self._client = boto3.client("glue", region_name=self._region)
            return self._client

    # ── Read APIs ──

    def list_databases(self) -> list[str]:
        client = self._get_client()
        out: list[str] = []
        paginator = client.get_paginator("get_databases")
        for page in paginator.paginate():
            out.extend(db["Name"] for db in page.get("DatabaseList", []))
        return out

    def list_tables(self, database: str, name_filter: str | None = None) -> list[str]:
        client = self._get_client()
        paginator = client.get_paginator("get_tables")
        kwargs: dict[str, Any] = {"DatabaseName": database}
        if name_filter:
            kwargs["Expression"] = name_filter
        names: list[str] = []
        for page in paginator.paginate(**kwargs):
            names.extend(t["Name"] for t in page.get("TableList", []))
        return names

    def get_table(self, database: str, name: str) -> Optional[GlueTable]:
        client = self._get_client()
        try:
            resp = client.get_table(DatabaseName=database, Name=name)
        except client.exceptions.EntityNotFoundException:
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("glue get_table(%s.%s) failed: %s", database, name, exc)
            return None
        return _table_from_api(resp.get("Table") or {})

    def get_partitions(
        self,
        database: str,
        table: str,
        *,
        expression: str | None = None,
        max_results: int = 200,
    ) -> list[GluePartition]:
        """Return (up to max_results) partitions for a table.

        `expression` is a Glue filter expression like `sales_date > 20260101`.
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {"DatabaseName": database, "TableName": table}
        if expression:
            kwargs["Expression"] = expression
        out: list[GluePartition] = []
        paginator = client.get_paginator("get_partitions")
        try:
            for page in paginator.paginate(**kwargs):
                for p in page.get("Partitions", []):
                    out.append(_partition_from_api(p))
                    if len(out) >= max_results:
                        return out
        except client.exceptions.EntityNotFoundException:
            return []
        except Exception as exc:  # noqa: BLE001
            log.warning("glue get_partitions(%s.%s) failed: %s", database, table, exc)
            return out
        return out

    def discover_table(self, ref: str, databases: Iterable[str] | None = None) -> list[GlueTable]:
        """Find a table by (possibly schema-qualified) name across databases.

        - `ref` may be "db.table" or just "table".
        - If `databases` is None, searches all databases (one Glue call per
          database's name-filter, so it scales reasonably).
        """
        match = _TABLE_REF.match(ref)
        if not match:
            return []
        db_hint, name = match.group(1), match.group(2)

        if db_hint:
            found = self.get_table(db_hint, name)
            return [found] if found else []

        candidate_dbs = list(databases) if databases else self.list_databases()
        hits: list[GlueTable] = []
        for db in candidate_dbs:
            try:
                names = self.list_tables(db, name_filter=re.escape(name))
            except Exception:
                continue
            if name not in names:
                continue
            found = self.get_table(db, name)
            if found is not None:
                hits.append(found)
        return hits


# ── Helpers ──


def _columns(raw: Iterable[dict[str, Any]]) -> tuple[GlueColumn, ...]:
    return tuple(
        GlueColumn(
            name=str(c.get("Name") or ""),
            type=str(c.get("Type") or ""),
            comment=(c.get("Comment") if c.get("Comment") else None),
        )
        for c in raw or []
    )


def _isoformat(val: Any) -> Optional[str]:
    if val is None:
        return None
    iso = getattr(val, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            return str(val)
    return str(val)


def _table_from_api(t: dict[str, Any]) -> GlueTable:
    storage = t.get("StorageDescriptor") or {}
    return GlueTable(
        database=str(t.get("DatabaseName") or ""),
        name=str(t.get("Name") or ""),
        owner=t.get("Owner"),
        location=storage.get("Location"),
        table_type=t.get("TableType"),
        parameters=dict(t.get("Parameters") or {}),
        columns=_columns(storage.get("Columns") or []),
        partition_keys=_columns(t.get("PartitionKeys") or []),
        created=_isoformat(t.get("CreateTime")),
        updated=_isoformat(t.get("UpdateTime")),
    )


def _partition_from_api(p: dict[str, Any]) -> GluePartition:
    storage = p.get("StorageDescriptor") or {}
    return GluePartition(
        values=tuple(str(v) for v in (p.get("Values") or [])),
        location=storage.get("Location"),
        created=_isoformat(p.get("CreationTime")),
        updated=_isoformat(p.get("LastAccessTime")),
        parameters=dict(p.get("Parameters") or {}),
    )


# ── Module singleton ──

_DEFAULT: GlueCatalog | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_catalog() -> GlueCatalog:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = GlueCatalog()
        return _DEFAULT
