"""Entity resolution for provider/site/customer codes."""

from __future__ import annotations

import re
from typing import Any

from .catalog import LocalCodeCatalog
from .datasources import DatasourceRegistry

_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


class EntityResolver:
    """Resolve entities from local common codes, then MySQL fallback."""

    def __init__(self, *, catalog: LocalCodeCatalog, registry: DatasourceRegistry) -> None:
        self.catalog = catalog
        self.registry = registry

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for match in _TOKEN_RE.finditer(text or ""):
            raw = match.group(0).strip()
            if not raw:
                continue
            norm = "".join(ch for ch in raw.upper() if ch.isalnum())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(raw)
        return out

    def resolve(self, text: str, *, sales_date_hint: str | None = None) -> dict[str, Any]:
        del sales_date_hint
        providers: list[str] = []
        sites: list[str] = []
        customers: list[str] = []
        unknown: list[str] = []

        tokens = self._tokenize(text)
        for token in tokens:
            from_common = self.catalog.resolve(token)
            if from_common is None:
                unknown.append(token)
                continue
            entity_type, canonical = from_common
            if entity_type == "provider" and canonical not in providers:
                providers.append(canonical)
            elif entity_type == "site" and canonical not in sites:
                sites.append(canonical)
            elif entity_type == "customer" and canonical not in customers:
                customers.append(canonical)

        if unknown:
            lookup = self.registry.mysql_lookup_codes(unknown)
            for key, target in [("providers", providers), ("sites", sites), ("customers", customers)]:
                for code in lookup.get(key, []):
                    if code not in target:
                        target.append(code)

        for sep in ["|", "/", ":"]:
            if sep in text:
                for chunk in text.split(sep):
                    from_common = self.catalog.resolve(chunk)
                    if not from_common:
                        continue
                    entity_type, canonical = from_common
                    if entity_type == "provider" and canonical not in providers:
                        providers.append(canonical)
                    if entity_type == "site" and canonical not in sites:
                        sites.append(canonical)

        return {
            "providers": providers,
            "sites": sites,
            "customers": customers,
            "unknown_tokens": unknown,
            "all_tokens": tokens,
        }


__all__ = ["EntityResolver"]
