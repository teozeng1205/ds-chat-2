"""Entity resolution for provider/site/customer codes."""

from __future__ import annotations

import re
from typing import Any

from .catalog import LocalCodeCatalog
from .datasources import DatasourceRegistry

_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")
_PIPE_SEP_RE = re.compile(r"[|/:]")


class EntityResolver:
    """Resolve entities from local common codes, then MySQL fallback with caching."""

    def __init__(self, *, catalog: LocalCodeCatalog, registry: DatasourceRegistry) -> None:
        self.catalog = catalog
        self.registry = registry
        self._mysql_cache: dict[str, dict[str, list[str]]] = {}

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

    def _mysql_lookup_cached(self, tokens: list[str]) -> dict[str, list[str]]:
        cache_key = "|".join(sorted(t.upper() for t in tokens))
        if cache_key in self._mysql_cache:
            return self._mysql_cache[cache_key]
        result = self.registry.mysql_lookup_codes(tokens)
        self._mysql_cache[cache_key] = result
        return result

    def resolve(self, text: str, *, sales_date_hint: str | None = None) -> dict[str, Any]:
        providers: list[str] = []
        sites: list[str] = []
        customers: list[str] = []
        unknown: list[str] = []

        # Handle pipe/slash/colon separated pairs (e.g. "QL2|AV")
        for sep_match in _PIPE_SEP_RE.finditer(text):
            sep = sep_match.group(0)
            for chunk in text.split(sep):
                chunk = chunk.strip()
                if not chunk:
                    continue
                from_common = self.catalog.resolve(chunk)
                if from_common is None:
                    continue
                entity_type, canonical = from_common
                if entity_type == "provider" and canonical not in providers:
                    providers.append(canonical)
                elif entity_type == "site" and canonical not in sites:
                    sites.append(canonical)
                elif entity_type == "customer" and canonical not in customers:
                    customers.append(canonical)

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
            lookup = self._mysql_lookup_cached(unknown)
            for key, target in [("providers", providers), ("sites", sites), ("customers", customers)]:
                for code in lookup.get(key, []):
                    if code not in target:
                        target.append(code)

        return {
            "providers": providers,
            "sites": sites,
            "customers": customers,
            "unknown_tokens": unknown,
            "all_tokens": tokens,
            "sales_date_hint": sales_date_hint,
        }


__all__ = ["EntityResolver"]
