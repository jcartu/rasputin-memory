from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_NAME = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_MAX_LENGTH = int(os.environ.get("CROSS_ENCODER_MAX_LENGTH", "512"))
_BATCH_SIZE = int(os.environ.get("CROSS_ENCODER_BATCH_SIZE", "32"))

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder: %s", _MODEL_NAME)
        _model = CrossEncoder(_MODEL_NAME, max_length=_MAX_LENGTH, device="cpu")
        logger.info("Cross-encoder loaded (CPU)")
        return _model
    except ImportError:
        logger.warning("sentence-transformers not installed, cross-encoder disabled")
        return None
    except Exception as e:
        logger.error("Failed to load cross-encoder: %s", e)
        return None


def is_available() -> bool:
    return _load_model() is not None


def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    model = _load_model()
    if model is None or not results:
        return results[:top_k] if top_k else results

    pairs = []
    for r in results:
        doc_text = (r.get("text") or "")[:1000]
        date = r.get("date") or ""
        if date and len(date) >= 10:
            doc_text = f"[Date: {date[:10]}] {doc_text}"
        source = r.get("source") or ""
        if source:
            doc_text = f"[Source: {source}] {doc_text}"
        pairs.append([query, doc_text])

    try:
        raw_scores = model.predict(pairs, batch_size=_BATCH_SIZE)
    except Exception as e:
        logger.error("Cross-encoder predict failed: %s", e)
        return results[:top_k] if top_k else results

    for r, raw_score in zip(results, raw_scores):
        score = float(raw_score)
        if math.isnan(score):
            score = 0.0
        r["ce_score"] = round(1.0 / (1.0 + math.exp(-score)), 6)
        r["ce_raw"] = round(score, 4)

    results.sort(key=lambda x: x.get("ce_score", 0), reverse=True)
    return results[:top_k] if top_k else results


def rerank_with_recency(
    query: str,
    results: list[dict[str, Any]],
    top_k: int | None = None,
    recency_alpha: float = 0.2,
) -> list[dict[str, Any]]:
    results = rerank(query, results, top_k=None)
    now = datetime.now(timezone.utc)

    for r in results:
        ce = r.get("ce_score", 0.5)
        recency = 0.5
        date_str = r.get("date") or ""
        if date_str:
            try:
                if date_str.endswith("Z"):
                    date_str = date_str[:-1] + "+00:00"
                dt = datetime.fromisoformat(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                days_ago = max((now - dt).total_seconds() / 86400, 0)
                recency = max(0.1, min(1.0, 1.0 - (days_ago / 365)))
            except (ValueError, TypeError):
                pass

        recency_boost = 1.0 + recency_alpha * (recency - 0.5)
        r["recency"] = round(recency, 4)
        r["recency_boost"] = round(recency_boost, 4)
        r["final_score"] = round(ce * recency_boost, 6)

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return results[:top_k] if top_k else results
