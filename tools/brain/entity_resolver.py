"""Entity resolution — map raw entity mentions to canonical forms.

Resolves "Alice", "my friend Alice" → canonical "Alice" by maintaining
an alias table backed by FalkorDB.  Used at commit time to prevent
duplicate graph nodes, and at search time to resolve query entities.

Resolution cascade (first match wins):
    1. Exact / case-insensitive lookup in cached alias table
    2. Word-level containment  ("Alice" ⊂ words of "my friend Alice")
    3. Fuzzy match  (SequenceMatcher ratio ≥ FUZZY_THRESHOLD)
    4. No match → register as new canonical entry

Gated behind ``ENTITY_RESOLVER=1`` environment variable (default off).
"""

from __future__ import annotations

import logging
import os
from difflib import SequenceMatcher

from brain import _state

logger = logging.getLogger(__name__)

ENTITY_RESOLVER_ENABLED = os.environ.get("ENTITY_RESOLVER", "0") == "1"
FUZZY_THRESHOLD = float(os.environ.get("ENTITY_RESOLVER_FUZZY", "0.85"))


class EntityResolver:
    def __init__(self) -> None:
        self._cache: dict[str, str] = {}  # lowercase_name → canonical_name
        self._loaded_graph: str = ""  # which graph is cached

    def _load_cache(self, graph_name: str) -> None:
        target = graph_name or _state.GRAPH_NAME
        if self._loaded_graph == target and self._cache:
            return
        try:
            redis_client = _state.get_falkordb()
            result = redis_client.execute_command(
                "GRAPH.QUERY",
                target,
                "MATCH (n) WHERE n.canonical IS NOT NULL RETURN n.canonical, n.name, labels(n)[0]",
            )
            self._cache.clear()
            for row in result[1] or []:
                canonical = row[0] if isinstance(row[0], str) else row[0].decode()
                name = row[1] if isinstance(row[1], str) else row[1].decode()
                self._cache[canonical.lower()] = canonical
                if name:
                    self._cache[name.lower()] = canonical
            self._loaded_graph = target
        except Exception as exc:
            logger.warning("Failed to load entity cache: %s", exc)

    def invalidate_cache(self) -> None:
        self._cache.clear()
        self._loaded_graph = ""

    def resolve(
        self,
        raw_entities: list[tuple[str, str]],
        context: str,
        graph_name: str = "",
    ) -> list[tuple[str, str, str]]:
        """Return ``[(raw_name, canonical_name, entity_type), ...]``."""
        if not ENTITY_RESOLVER_ENABLED:
            return [(name, name, etype) for name, etype in raw_entities]

        self._load_cache(graph_name)
        resolved: list[tuple[str, str, str]] = []

        for raw_name, entity_type in raw_entities:
            canonical = self._find_canonical(raw_name)
            if canonical is None:
                canonical = raw_name
                self._cache[raw_name.lower()] = canonical
            resolved.append((raw_name, canonical, entity_type))

        return resolved

    def _find_canonical(self, name: str) -> str | None:
        lower = name.lower()

        # 1. Exact / case-insensitive match
        if lower in self._cache:
            return self._cache[lower]

        # 2. Word-level containment (avoids char-substring false positives like John/Jonathan)
        lower_words = set(lower.split())
        for cached_lower, canonical in self._cache.items():
            cached_words = set(cached_lower.split())
            if lower_words.issubset(cached_words) or cached_words.issubset(lower_words):
                self._cache[lower] = canonical
                return canonical

        # 3. Fuzzy match
        best_score = 0.0
        best_canonical: str | None = None
        for cached_lower, canonical in self._cache.items():
            ratio = SequenceMatcher(None, lower, cached_lower).ratio()
            if ratio > best_score and ratio >= FUZZY_THRESHOLD:
                best_score = ratio
                best_canonical = canonical

        if best_canonical is not None:
            self._cache[lower] = best_canonical
            return best_canonical

        return None


_resolver = EntityResolver()


def resolve(
    raw_entities: list[tuple[str, str]],
    context: str,
    graph_name: str = "",
) -> list[tuple[str, str, str]]:
    return _resolver.resolve(raw_entities, context, graph_name)


def invalidate_cache() -> None:
    _resolver.invalidate_cache()
