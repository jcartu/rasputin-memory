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
        df = Counter()
        for tokens in doc_tokens:
            unique = set(tokens)
            for t in unique:
                df[t] += 1

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
        result["bm25_score"] = bm25_scores[idx] if idx < len(bm25_scores) else 0
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
    documents = []
    for r in dense_results:
        p = r.get("payload", {})
        parts = []
        if p.get("subject"):
            parts.append(p["subject"])
        if p.get("title"):
            parts.append(p["title"])
        if p.get("question"):
            parts.append(p["question"])
        text = p.get("text", p.get("body", ""))
        if text:
            parts.append(text[:1000])
        documents.append(" ".join(parts))

    # Score with BM25
    bm25_scores = _scorer.score(query, documents)

    # Fuse with RRF
    fused_results = reciprocal_rank_fusion(dense_results, bm25_scores)
    if bm25_weight <= 0:
        return fused_results
    return fused_results
