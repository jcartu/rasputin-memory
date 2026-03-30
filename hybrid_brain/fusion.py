"""
hybrid_brain.fusion — RRF (Reciprocal Rank Fusion) utilities.

Example::

    from hybrid_brain import RRFFusion

    fuser = RRFFusion(k=60)
    merged = fuser.fuse([vector_ranked_ids, bm25_ranked_ids])
"""

from __future__ import annotations

from typing import Dict, List


class RRFFusion:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Parameters
    ----------
    k:
        RRF constant (default 60 — as per the original paper).
        Higher k reduces the impact of rank differences.

    Example::

        fuser = RRFFusion(k=60)
        fused = fuser.fuse([["doc1", "doc3", "doc2"], ["doc2", "doc1"]])
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, rankings: List[List[str]]) -> List[str]:
        """Fuse multiple ranked lists into a single ranking.

        Parameters
        ----------
        rankings:
            List of ranked doc-ID lists (best-first).

        Returns
        -------
        Fused ranking (best-first).
        """
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def fuse_with_scores(self, rankings: List[List[str]]) -> List[tuple[str, float]]:
        """Like :meth:`fuse` but returns ``(doc_id, rrf_score)`` pairs."""
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
