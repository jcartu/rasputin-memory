from __future__ import annotations

import threading
import time
import importlib
from datetime import datetime
from typing import Any, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from brain import _state
from brain import embedding
from brain import entities as entities_module
from brain import graph
from brain import scoring

import re as _re

safe_import = importlib.import_module("pipeline._imports").safe_import

_TOKEN_RE = _re.compile(r"\w+", _re.UNICODE)

try:
    from bm25_search import hybrid_rerank as bm25_rerank
except ModuleNotFoundError:
    from tools.bm25_search import hybrid_rerank as bm25_rerank  # type: ignore[no-redef]

_query_expansion = safe_import("pipeline.query_expansion", "tools.pipeline.query_expansion")
expand_queries = _query_expansion.expand_queries


def _neural_rerank(query: str, results: list[dict[str, Any]], top_k: Optional[int] = None) -> list[dict[str, Any]]:
    if not results:
        return results

    passages = []
    for row in results:
        text = row.get("text", "")[:1000]
        source = row.get("source", "")
        title = row.get("title", "")
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if source:
            parts.append(f"Source: {source}")
        parts.append(text)
        passages.append(" | ".join(parts))

    try:
        response = _state.requests.post(
            _state.RERANKER_URL,
            json={"query": query, "passages": passages},
            timeout=_state.RERANKER_TIMEOUT,
        )
        response.raise_for_status()
        scores = response.json().get("scores", [])

        if len(scores) != len(results):
            return results

        for idx, row in enumerate(results):
            row["rerank_score"] = scores[idx]

        reranked = sorted(results, key=lambda value: value.get("rerank_score", 0), reverse=True)
        return reranked[:top_k] if top_k else reranked
    except Exception as error:
        _state.logger.error("Reranker error: %s", error)
        return results


def qdrant_search(query: str, limit: int = 10, source_filter: Optional[str] = None) -> list[dict[str, Any]]:
    try:
        vector = embedding.get_embedding(query)
    except Exception as error:
        _state.logger.error("Qdrant embedding error: %s", error)
        return []

    search_filter = None
    if source_filter:
        search_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_filter))])

    results = _state.qdrant.query_points(  # type: ignore[attr-defined]  # qdrant-client>=1.9.0
        collection_name=_state.COLLECTION,
        query=vector,
        limit=limit,
        query_filter=search_filter,
        with_payload=True,
    )

    out = []
    for point in results.points:
        payload = point.payload or {}
        out.append(
            {
                "score": round(point.score, 4),
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "date": payload.get("date", ""),
                "title": payload.get("title", ""),
                "url": payload.get("url", ""),
                "domain": payload.get("domain", ""),
                "importance": payload.get("importance", 50),
                "retrieval_count": payload.get("retrieval_count", 0),
                "last_accessed": payload.get("last_accessed", ""),
                "point_id": point.id,
                "origin": "qdrant",
            }
        )

    return scoring.apply_temporal_decay(out)


def hybrid_search(
    query: str,
    limit: int = 10,
    graph_hops: int = 2,
    source_filter: Optional[str] = None,
    expand: bool = True,
) -> dict[str, Any]:
    start = time.time()

    queries = [query]
    if expand:
        queries = expand_queries(query, max_expansions=5)

    fetch_limit = limit * 4
    all_qdrant_results = []
    for expanded_query in queries:
        all_qdrant_results.extend(qdrant_search(expanded_query, limit=fetch_limit, source_filter=source_filter))

    deduped_by_text: dict[str, dict[str, Any]] = {}
    for item in all_qdrant_results:
        text_key = (item.get("text") or "").strip().lower()
        if not text_key:
            text_key = f"point:{item.get('point_id', '')}"
        current = deduped_by_text.get(text_key)
        if current is None or item.get("score", 0) > current.get("score", 0):
            deduped_by_text[text_key] = item

    qdrant_results = sorted(deduped_by_text.values(), key=lambda value: value.get("score", 0), reverse=True)
    graph_results = graph.graph_search(query, hops=graph_hops, limit=limit)

    bm25_applied = False
    if _state.BM25_AVAILABLE and qdrant_results:
        try:
            qdrant_results = bm25_rerank(query, qdrant_results)
            bm25_applied = True
        except Exception as error:
            _state.logger.error("BM25 rerank error: %s", error)

    if qdrant_results and any("source_weight" in row for row in qdrant_results):
        qdrant_results = scoring.apply_multifactor_scoring(qdrant_results)

    # Keyword overlap boosting: boost results that share distinctive tokens with the query
    query_tokens = set(_TOKEN_RE.findall(query.lower()))
    _stopwords = {"the", "a", "an", "is", "was", "are", "were", "do", "does", "did",
                  "what", "where", "when", "how", "who", "which", "that", "this",
                  "in", "on", "at", "to", "for", "of", "with", "by", "from", "and",
                  "or", "not", "it", "we", "he", "she", "they", "i", "you", "my",
                  "about", "know", "now", "has", "have", "had", "be", "been"}
    query_content_tokens = query_tokens - _stopwords
    if query_content_tokens:
        score_key = "hybrid_score" if any("hybrid_score" in r for r in qdrant_results) else "score"
        for row in qdrant_results:
            text_tokens = set(_TOKEN_RE.findall((row.get("text") or "").lower()))
            overlap = query_content_tokens & text_tokens
            if overlap:
                overlap_ratio = len(overlap) / len(query_content_tokens)
                boost = 1.0 + 4.0 * overlap_ratio  # up to 5x for full overlap
                row[score_key] = row.get(score_key, row.get("score", 0)) * boost
                row["keyword_boosted"] = True
        qdrant_results = sorted(
            qdrant_results,
            key=lambda v: v.get("hybrid_score") if v.get("hybrid_score") is not None else v.get("score", 0),
            reverse=True,
        )

    # Entity boosting: when query mentions known entities, boost results that contain those entities
    # Higher boost when the queried entity is the PRIMARY entity in the text (entity focus ratio)
    query_entities = [name for name, _type in entities_module.extract_entities_fast(query)]
    if query_entities:
        score_key = "hybrid_score" if any("hybrid_score" in r for r in qdrant_results) else "score"
        query_entity_names_lower = {e.lower() for e in query_entities}
        for row in qdrant_results:
            text = row.get("text") or ""
            text_lower = text.lower()
            matched = [e for e in query_entities if e.lower() in text_lower]
            if matched:
                # Count all entities in the text
                all_text_entities = entities_module.extract_entities_fast(text)
                total_entities = max(len(all_text_entities), 1)
                matched_count = sum(1 for name, _ in all_text_entities if name.lower() in query_entity_names_lower)
                # Focus ratio: 1.0 if text only has query entities, lower if text has many other entities
                focus_ratio = matched_count / total_entities
                # Position bonus: entity appearing earlier = text is more about that entity
                position_bonus = 0.0
                for e in matched:
                    pos = text_lower.find(e.lower())
                    if pos >= 0:
                        # Normalize position: 0.0 at end, 1.0 at start
                        text_len = max(len(text_lower), 1)
                        position_bonus = max(position_bonus, 1.0 - (pos / text_len))
                # Boost scales with focus and position
                boost = 1.5 + 1.0 * focus_ratio + 0.5 * position_bonus
                current = row.get(score_key, row.get("score", 0))
                row[score_key] = current * boost
                row["entity_boosted"] = True
                row["entity_focus"] = round(focus_ratio, 2)
        qdrant_results = sorted(
            qdrant_results,
            key=lambda v: v.get("hybrid_score") if v.get("hybrid_score") is not None else v.get("score", 0),
            reverse=True,
        )

    graph_memory_results = [row for row in graph_results if row.get("source") in ("graph_memory", "graph_keyword")]
    graph_context_results = [row for row in graph_results if row.get("source") == "graph_context"]

    for graph_row in graph_memory_results:
        hop_count = graph_row.get("graph_hop", 1)
        graph_row["score"] = 0.8 if hop_count == 1 else 0.5

    all_candidates = list(qdrant_results[: limit * 2]) + graph_memory_results

    neural_applied = False
    reranker_up = embedding.is_reranker_available()
    if reranker_up and all_candidates:
        pre_count = len(all_candidates)
        all_candidates = _neural_rerank(query, all_candidates[: limit * 3], top_k=limit)
        neural_applied = len(all_candidates) <= pre_count and any(
            row.get("rerank_score") is not None for row in all_candidates
        )
    else:
        # Use hybrid_score (BM25-adjusted) when available, fall back to cosine score
        all_candidates = sorted(
            all_candidates,
            key=lambda value: value.get("hybrid_score") if value.get("hybrid_score") is not None else value.get("score", 0),
            reverse=True,
        )[:limit]

    merged = all_candidates[:limit]

    if graph_context_results and len(merged) < limit:
        for graph_context in graph_context_results[: limit - len(merged)]:
            graph_context["score"] = 0.1
            merged.append(graph_context)

    graph_enrichment = graph.enrich_with_graph(merged, limit=5)

    elapsed = time.time() - start

    _update_access_tracking([row for row in merged if row.get("origin") != "graph"])

    return {
        "query": query,
        "elapsed_ms": round(elapsed * 1000, 1),
        "results": merged,
        "graph_context": graph_enrichment,
        "graph_results": graph_results,
        "graph_enrichment": graph_enrichment,
        "stats": {
            "expanded_queries": len(queries),
            "query_expansion_enabled": bool(expand),
            "qdrant_hits": len([row for row in merged if row.get("origin") != "graph"]),
            "graph_hits": len(graph_results),
            "graph_merged": len([row for row in merged if row.get("origin") == "graph"]),
            "graph_enriched_entities": len(graph_enrichment),
            "bm25_reranked": bm25_applied,
            "neural_reranked": neural_applied,
        },
    }


def _update_access_tracking(results: list[dict[str, Any]]) -> None:
    now = datetime.now().isoformat()

    def _do_update() -> None:
        for row in results:
            point_id = row.get("point_id")
            if point_id is None:
                continue
            try:
                points = _state.qdrant.retrieve(
                    collection_name=_state.COLLECTION,
                    ids=[point_id],
                    with_payload=True,
                )
                if points:
                    payload = points[0].payload or {}
                    current_count = payload.get("retrieval_count", 0) or 0
                    _state.qdrant.set_payload(
                        collection_name=_state.COLLECTION,
                        points=[point_id],
                        payload={
                            "last_accessed": now,
                            "retrieval_count": current_count + 1,
                        },
                    )
            except Exception:
                pass

    try:
        thread = threading.Thread(target=_do_update, daemon=True)
        thread.start()
    except Exception:
        pass
