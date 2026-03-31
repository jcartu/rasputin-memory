from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional


def _parse_date(date_str: str) -> Optional[Any]:
    if not date_str:
        return None

    try:
        normalized = date_str
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def apply_temporal_decay(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)

    for row in results:
        parsed_date = _parse_date(row.get("date", ""))
        if not parsed_date:
            continue

        days_old = max((now - parsed_date).total_seconds() / 86400, 0)

        importance = row.get("importance", 50)
        if importance is None:
            importance = 50
        try:
            importance = int(importance)
        except (ValueError, TypeError):
            importance = 50

        if importance >= 80:
            base_half_life = 365
        elif importance >= 40:
            base_half_life = 60
        else:
            base_half_life = 14

        retrieval_count = row.get("retrieval_count", 0) or 0
        effective_half_life = base_half_life * (1 + 0.1 * min(retrieval_count, 20))

        stability = effective_half_life / math.log(2)
        decay_factor = math.exp(-days_old / stability)

        row["original_score"] = row["score"]
        row["score"] = round(row["score"] * (0.2 + 0.8 * decay_factor), 4)
        row["days_old"] = round(days_old, 1)
        row["effective_half_life"] = round(effective_half_life, 0)

    return sorted(results, key=lambda value: value["score"], reverse=True)


def apply_multifactor_scoring(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in results:
        importance = row.get("importance", 50)
        if importance is None:
            importance = 50
        try:
            importance = int(importance)
        except (ValueError, TypeError):
            importance = 50
        importance_norm = min(importance / 100.0, 1.0)

        source_weight = row.get("source_weight", 0.5)

        retrieval_count = row.get("retrieval_count", 0) or 0
        retrieval_boost = min(retrieval_count / 10.0, 1.0)

        days_old = row.get("days_old", 30)
        if days_old is not None and days_old < 7:
            recency_bonus = 1.0
        elif days_old is not None and days_old < 30:
            recency_bonus = 0.7
        else:
            recency_bonus = 0.4

        multiplier = (
            0.35 + 0.25 * importance_norm + 0.20 * recency_bonus + 0.10 * source_weight + 0.10 * retrieval_boost
        )

        row["pre_multifactor_score"] = row["score"]
        row["score"] = round(row["score"] * multiplier, 4)
        row["multifactor"] = round(multiplier, 3)

    return sorted(results, key=lambda value: value["score"], reverse=True)
