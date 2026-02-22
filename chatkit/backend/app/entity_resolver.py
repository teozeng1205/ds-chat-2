"""Provider/site/customer resolver with common-code and MySQL fallback."""

from __future__ import annotations

import re
from typing import Any

from .knowledge_base import KnowledgeBaseService
from .nextgen_types import EntityResolution
from .threevictors_client import ThreeVictorsClient, ThreeVictorsDependencyError


_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


class EntityResolver:
    def __init__(self, kb: KnowledgeBaseService, tv_client: ThreeVictorsClient):
        self.kb = kb
        self.tv_client = tv_client
        self._mysql_columns_cache: dict[str, list[str]] = {}

    @staticmethod
    def _tokenize(raw_codes: str) -> list[str]:
        tokens = [match.group(0).strip() for match in _TOKEN_RE.finditer(raw_codes or "")]
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            upper = token.upper()
            if upper in seen:
                continue
            seen.add(upper)
            deduped.append(token)
        return deduped

    @staticmethod
    def _normalize(code: str) -> str:
        return "".join(ch for ch in code.upper() if ch.isalnum())

    def resolve_codes(self, raw_codes: str, cache: dict[str, Any] | None = None) -> list[EntityResolution]:
        cache_store = cache if isinstance(cache, dict) else {}
        out: list[EntityResolution] = []
        for token in self._tokenize(raw_codes):
            norm = self._normalize(token)
            if norm in cache_store:
                out.append(EntityResolution.model_validate(cache_store[norm]))
                continue

            resolved = self._resolve_one(token)
            cache_store[norm] = resolved.model_dump(mode="json")
            out.append(resolved)

        return out

    def _resolve_one(self, token: str) -> EntityResolution:
        normalized = self._normalize(token)
        from_common = self.kb.resolve_common_code(token)
        if from_common is not None:
            entity_type, canonical = from_common
            return EntityResolution(
                input_code=token,
                normalized_code=normalized,
                entity_type=entity_type,  # type: ignore[arg-type]
                canonical_value=canonical,
                confidence=1.0,
                source="common_codes",
                candidates=[canonical],
                ambiguous=False,
            )

        mysql_result = self._resolve_from_mysql(token)
        if mysql_result is not None:
            return mysql_result

        return EntityResolution(
            input_code=token,
            normalized_code=normalized,
            entity_type="unknown",
            canonical_value=None,
            confidence=0.0,
            source="unknown",
            candidates=[],
            ambiguous=False,
        )

    def _resolve_from_mysql(self, token: str) -> EntityResolution | None:
        table_candidates = [
            ("provider", "priceeye.provider", ["providercode", "provider_code", "code", "provider"]),
            ("site", "priceeye.site", ["sitecode", "site_code", "code", "site"]),
            ("customer", "priceeye.customer", ["customer", "customercode", "customer_code", "code"]),
        ]

        for entity_type, table_name, preferred_columns in table_candidates:
            try:
                matched = self._lookup_mysql_code(table_name, preferred_columns, token)
            except ThreeVictorsDependencyError:
                return None
            except Exception:
                continue

            if not matched:
                continue

            return EntityResolution(
                input_code=token,
                normalized_code=self._normalize(token),
                entity_type=entity_type,  # type: ignore[arg-type]
                canonical_value=matched[0],
                confidence=0.9 if len(matched) == 1 else 0.6,
                source="mysql_fallback",
                candidates=matched,
                ambiguous=len(matched) > 1,
            )

        return None

    def _lookup_mysql_code(self, table_name: str, preferred_columns: list[str], token: str) -> list[str]:
        columns = self._mysql_columns_cache.get(table_name)
        if columns is None:
            columns = [col.lower() for col in self.tv_client.mysql_table_columns(table_name)]
            self._mysql_columns_cache[table_name] = columns

        candidate_col = next((name for name in preferred_columns if name.lower() in columns), None)
        if not candidate_col:
            return []

        escaped = token.replace("'", "''")
        query = (
            f"SELECT DISTINCT {candidate_col} AS code "
            f"FROM {table_name} "
            f"WHERE UPPER({candidate_col}) = UPPER('{escaped}') LIMIT 5"
        )
        frame = self.tv_client.query_mysql(query)
        if "code" not in frame.columns:
            return []
        values = [str(v).strip().upper() for v in frame["code"].tolist() if str(v).strip()]
        return sorted(set(values))
