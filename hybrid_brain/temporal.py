"""
hybrid_brain.temporal — Ebbinghaus power-law temporal decay and multi-factor scoring.

Example::

    from hybrid_brain import TemporalDecay

    decay = TemporalDecay(half_life_days=30)
    results = decay.apply(results)           # modifies score in-place, re-sorts
    results = decay.multifactor(results)     # additional source/retrieval weighting
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO-ish date strings. Returns None on failure."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(date_str)[:19], fmt)
        except ValueError:
            pass
    return None


class TemporalDecay:
    """Ebbinghaus power-law temporal decay for memory results.

    Importance tiers control half-life:
    - importance ≥ 80 → 365-day half-life
    - importance 40–79 → 60-day half-life
    - importance < 40 → 14-day half-life

    Retrieval count (spaced-repetition) adds 10 % per access (capped at 20).
    Score floor is 20 % to prevent permanent invisibility of old critical memories.

    Parameters
    ----------
    half_life_days:
        Default half-life (days) — used only when importance is not present.

    Example::

        decay = TemporalDecay()
        scored = decay.apply(results)
    """

    SOURCE_WEIGHTS: Dict[str, float] = {
        "conversation": 0.9, "fact_extractor": 0.85, "chatgpt": 0.8,
        "perplexity": 0.75, "email": 0.6, "telegram": 0.7, "whatsapp": 0.65,
        "social_intel": 0.5, "web_page": 0.4, "benchmark_test": 0.1,
    }

    def __init__(self, half_life_days: int = 30) -> None:
        self.half_life_days = half_life_days

    def apply(self, results: List[Dict[str, Any]], now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Apply Ebbinghaus decay to *results* in-place and re-sort by score.

        Each result dict is expected to have ``score``, ``date``, ``importance``,
        and optionally ``retrieval_count``.
        """
        now = now or datetime.now()
        for r in results:
            dt = _parse_date(r.get("date", ""))
            if not dt:
                continue
            days_old = max((now - dt).total_seconds() / 86400, 0)
            importance = self._safe_int(r.get("importance", 50), 50)

            if importance >= 80:
                base_half_life = 365
            elif importance >= 40:
                base_half_life = 60
            else:
                base_half_life = 14

            retrieval_count = r.get("retrieval_count", 0) or 0
            effective_half_life = base_half_life * (1 + 0.1 * min(retrieval_count, 20))
            stability = effective_half_life / math.log(2)
            decay_factor = math.exp(-days_old / stability)

            r["original_score"] = r.get("score", 0)
            r["score"] = round(r.get("score", 0) * (0.2 + 0.8 * decay_factor), 4)
            r["days_old"] = round(days_old, 1)
            r["effective_half_life"] = round(effective_half_life, 0)

        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    def multifactor(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Multi-factor importance scoring combining similarity, importance,
        recency, source reliability, and retrieval frequency.

        Formula::

            score = vector_sim × (0.35 + 0.25×importance_norm
                                       + 0.20×recency
                                       + 0.10×source_weight
                                       + 0.10×retrieval_boost)
        """
        for r in results:
            importance = self._safe_int(r.get("importance", 50), 50)
            importance_norm = min(importance / 100.0, 1.0)
            source = r.get("source", "")
            source_weight = self.SOURCE_WEIGHTS.get(source, 0.5)
            if source_weight == 0.5 and "social_intel" in source:
                source_weight = 0.5
            retrieval_count = r.get("retrieval_count", 0) or 0
            retrieval_boost = min(retrieval_count / 10.0, 1.0)
            days_old = r.get("days_old", 30) or 30
            recency_bonus = 1.0 if days_old < 7 else (0.7 if days_old < 30 else 0.4)
            multiplier = (0.35 + 0.25 * importance_norm + 0.20 * recency_bonus
                          + 0.10 * source_weight + 0.10 * retrieval_boost)
            r["pre_multifactor_score"] = r.get("score", 0)
            r["score"] = round(r.get("score", 0) * multiplier, 4)
            r["multifactor"] = round(multiplier, 3)

        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    @staticmethod
    def _safe_int(val: Any, default: int) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return default
