"""Reciprocal Rank Fusion for multi-lane retrieval (Phase C).

Hindsight-parity port. Ties broken by first-seen lane. k=60 is TREC-standard;
do not tune as part of this phase (see RASPUTIN_HINDSIGHT_PARITY_PLAN §4.4).
"""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    result_lists: dict[str, list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists via reciprocal rank fusion.

    RRF score for document d: sum over lanes L of 1 / (k + rank_L(d)).

    Args:
        result_lists: dict mapping lane name -> ranked candidate list.
            Each candidate dict should have either a 'point_id' (preferred)
            or a 'text' field for deduplication.
        k: RRF smoothing constant. TREC-standard is 60.

    Returns:
        Flat list sorted by rrf_score desc. Each item carries:
            rrf_score (float, rounded to 6 decimals), rrf_rank (int, 1-based),
            source_ranks ({lane_name: rank}), origin="rrf_fusion", and all
            original payload fields from the first-seen instance of the candidate.
    """
    rrf_scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    all_cands: dict[str, dict[str, Any]] = {}

    def _key(item: dict[str, Any]) -> str:
        pid = item.get("point_id")
        if pid is not None and pid != "":
            return f"pid:{pid}"
        text = (item.get("text") or "").strip().lower()
        return f"txt:{text[:80]}"

    for lane_name, results in result_lists.items():
        for rank, item in enumerate(results, start=1):
            key = _key(item)
            if not key or key in ("pid:", "txt:"):
                continue
            all_cands.setdefault(key, item)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            source_ranks.setdefault(key, {})[lane_name] = rank

    merged: list[dict[str, Any]] = []
    sorted_keys = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    for rrf_rank, (key, score) in enumerate(sorted_keys, start=1):
        cand = dict(all_cands[key])  # copy; do not mutate caller's payload
        cand["rrf_score"] = round(score, 6)
        cand["rrf_rank"] = rrf_rank
        cand["source_ranks"] = dict(source_ranks[key])
        cand["origin"] = "rrf_fusion"
        merged.append(cand)
    return merged
