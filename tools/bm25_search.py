#!/usr/bin/env python3
"""
BM25 Search Layer for Rasputin Memory Engine
Implements client-side BM25 scoring on top of Qdrant dense vector results.
Used for hybrid search: dense (semantic) + sparse (keyword) with RRF fusion.
"""

import re
import math
from collections import Counter
from typing import Any


# Simple BM25 implementation (no external deps needed)
class BM25Scorer:
    """Lightweight BM25 scorer for re-scoring retrieved passages."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r"\w+", text.lower())

    def score(self, query: str, documents: list[str]) -> list[float]:
        """
        Score documents against query using BM25.
        Returns list of scores (same order as documents).
        """
        query_terms = self.tokenize(query)
        if not query_terms or not documents:
            return [0.0] * len(documents)

        # Tokenize all documents
        doc_tokens = [self.tokenize(d) for d in documents]
        doc_lens = [len(t) for t in doc_tokens]
        avg_dl = sum(doc_lens) / len(doc_lens) if doc_lens else 1

        # Document frequency
        df: dict[str, int] = {}
        for tokens in doc_tokens:
            unique = set(tokens)
            for t in unique:
                df[t] = df.get(t, 0) + 1

        N = len(documents)
        scores = []

        for i, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            dl = doc_lens[i]
            score = 0.0

            for term in query_terms:
                if term not in tf:
                    continue

                # IDF component
                n = df.get(term, 0)
                idf = math.log((N - n + 0.5) / (n + 0.5) + 1)

                # TF component with length normalization
                freq = tf[term]
                tf_norm = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / avg_dl))

                score += idf * tf_norm

            scores.append(score)

        return scores


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_scores: list[float],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Fuse dense vector scores with BM25 scores using Reciprocal Rank Fusion.
    k=60 is the standard constant from the RRF paper.

    Returns results re-ordered by fused score.
    """
    if not dense_results:
        return []

    if len(dense_results) != len(bm25_scores):
        if len(bm25_scores) < len(dense_results):
            bm25_scores = [*bm25_scores, *([0.0] * (len(dense_results) - len(bm25_scores)))]
        else:
            bm25_scores = bm25_scores[: len(dense_results)]

    # Get dense ranking
    dense_ranked = sorted(range(len(dense_results)), key=lambda i: dense_results[i].get("score", 0), reverse=True)

    # Get BM25 ranking
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

    # RRF scores
    rrf_scores = [0.0] * len(dense_results)

    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] += 1.0 / (k + rank + 1)

    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] += 1.0 / (k + rank + 1)

    # Sort by RRF score
    fused_order = sorted(range(len(dense_results)), key=lambda i: rrf_scores[i], reverse=True)

    # Return re-ordered results with RRF score attached
    fused_results = []
    for idx in fused_order:
        result = dense_results[idx].copy()
        result["rrf_score"] = rrf_scores[idx]
        result["bm25_score"] = bm25_scores[idx]
        fused_results.append(result)

    return fused_results


# Singleton scorer
_scorer = BM25Scorer()


def hybrid_rerank(query: str, dense_results: list[dict[str, Any]], bm25_weight: float = 0.3) -> list[dict[str, Any]]:
    """
    Apply BM25 + RRF fusion to dense search results.
    Call this BEFORE the neural reranker for best results.

    Pipeline: Dense search → BM25+RRF hybrid → Neural reranker → Final results
    """
    if not dense_results:
        return []

    # Extract text from results for BM25 scoring
    # Results may have text at top-level (from qdrant_search) or nested under 'payload'
    documents = []
    for r in dense_results:
        p = r.get("payload") or {}
        parts = []
        # Top-level fields (from qdrant_search output)
        for field in ("title", "subject", "question"):
            v = r.get(field) or p.get(field)
            if v:
                parts.append(str(v))
        text = r.get("text") or p.get("text") or p.get("body") or ""
        if text:
            parts.append(text[:1000])
        documents.append(" ".join(parts))

    # Score with BM25
    bm25_scores = _scorer.score(query, documents)

    # Fuse with RRF
    fused_results = reciprocal_rank_fusion(dense_results, bm25_scores)
    if bm25_weight <= 0:
        return fused_results

    weight = max(0.0, min(1.0, bm25_weight))
    bm25_values = [float(r.get("bm25_score", 0.0)) for r in fused_results]
    bm25_min = min(bm25_values) if bm25_values else 0.0
    bm25_max = max(bm25_values) if bm25_values else 0.0
    bm25_span = bm25_max - bm25_min

    rescored = []
    for r in fused_results:
        dense_score = float(r.get("score", 0.0))
        bm25_score = float(r.get("bm25_score", 0.0))
        bm25_norm = ((bm25_score - bm25_min) / bm25_span) if bm25_span > 0 else 0.0
        item = r.copy()
        item["hybrid_score"] = (1.0 - weight) * dense_score + weight * bm25_norm
        rescored.append(item)

    return sorted(rescored, key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
