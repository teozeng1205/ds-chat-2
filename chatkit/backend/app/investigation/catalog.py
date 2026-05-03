"""Local provider/site/customer code catalog.

KB V2 owns document, table, task, and lineage retrieval. This module now
keeps only the lightweight code catalog used by entity resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalCodeCatalog:
    """Local provider/site/customer code catalog from common_codes.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, Any] | None = None
        self._mtime: float = 0.0

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"providers": [], "sites": [], "customers": []}
        mtime = self.path.stat().st_mtime
        if self._cache is not None and mtime <= self._mtime:
            return self._cache
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        self._cache = payload
        self._mtime = mtime
        return payload

    @staticmethod
    def _normalize(raw: str) -> str:
        return "".join(ch for ch in str(raw).upper() if ch.isalnum())

    def resolve(self, token: str) -> tuple[str, str] | None:
        payload = self._load()
        buckets = [
            ("providers", "provider"),
            ("sites", "site"),
            ("customers", "customer"),
        ]
        norm = self._normalize(token)
        if not norm:
            return None

        for bucket, entity in buckets:
            for row in payload.get(bucket, []) or []:
                if isinstance(row, str):
                    canonical = row.strip().upper()
                    aliases: list[str] = []
                else:
                    canonical = str(row.get("code", "")).strip().upper()
                    aliases = [str(v).strip().upper() for v in (row.get("aliases", []) or [])]
                for candidate in [canonical, *aliases]:
                    if self._normalize(candidate) == norm:
                        return entity, canonical
        return None

    def rows(self) -> list[tuple[str, str, str]]:
        payload = self._load()
        out: list[tuple[str, str, str]] = []
        for key, entity in [
            ("providers", "provider"),
            ("sites", "site"),
            ("customers", "customer"),
        ]:
            for row in payload.get(key, []) or []:
                if isinstance(row, str):
                    code = row.strip().upper()
                else:
                    code = str(row.get("code", "")).strip().upper()
                if code:
                    out.append((code, entity, "common_codes"))
        return out


__all__ = ["LocalCodeCatalog"]
