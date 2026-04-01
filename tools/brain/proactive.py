from __future__ import annotations

import time
from typing import Any, Optional

from brain import entities
from brain import search


def proactive_surface(
    context_messages: list[str],
    max_results: int = 3,
    active_entities: Optional[list[str]] = None,
    min_days_since_access: int = 7,
) -> list[dict[str, Any]]:
    if not context_messages:
        return []

    full_context = " ".join(context_messages[-3:])
    context_lower = full_context.lower()

    entity_names: list[str] = []
    if active_entities:
        entity_names.extend([str(value) for value in active_entities if str(value).strip()])
    entity_names.extend([name for name, _ in entities.extract_entities_fast(full_context)])

    dedup_entities = []
    seen_entities = set()
    for name in entity_names:
        key = name.lower().strip()
        if key and key not in seen_entities:
            seen_entities.add(key)
            dedup_entities.append(name)

    suggestions: list[dict[str, Any]] = []
    now = time.time()

    for entity in dedup_entities[:8]:
        try:
            related = search.qdrant_search(entity, limit=6)
        except Exception:
            related = []
        for row in related:
            text = row.get("text", "")
            if not text or text.lower() in context_lower:
                continue

            last_accessed_raw = row.get("last_accessed") or row.get("date")
            last_accessed = search.scoring._parse_date(last_accessed_raw) if last_accessed_raw else None
            if last_accessed is not None:
                days_old = (now - last_accessed.timestamp()) / 86400
                if days_old < min_days_since_access:
                    continue

            relevance = row.get("rerank_score", row.get("score", 0.0))
            try:
                relevance = float(relevance)
            except (TypeError, ValueError):
                relevance = 0.0

            suggestions.append(
                {
                    "text": text[:500],
                    "relevance": round(relevance, 3),
                    "source": row.get("source", ""),
                    "reason": f"Related to: {entity}",
                    "last_accessed": row.get("last_accessed", ""),
                }
            )

    unique: dict[str, dict[str, Any]] = {}
    for item in suggestions:
        key = item["text"][:120]
        if key not in unique or item["relevance"] > unique[key]["relevance"]:
            unique[key] = item

    return sorted(unique.values(), key=lambda value: value["relevance"], reverse=True)[:max_results]
