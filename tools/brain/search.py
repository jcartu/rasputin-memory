from __future__ import annotations

import threading
import time
import importlib
from datetime import datetime
from typing import Any, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from brain import _state
from brain import embedding
from brain import entities
from brain import graph
from brain import scoring

safe_import = importlib.import_module("pipeline._imports").safe_import

try:
    from bm25_search import hybrid_rerank as bm25_rerank
except ModuleNotFoundError:
    from tools.bm25_search import hybrid_rerank as bm25_rerank

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

    results = _state.qdrant.query_points(
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

    out = scoring.apply_temporal_decay(out)
    out = scoring.apply_multifactor_scoring(out)
    return out


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

    deduped_by_text = {}
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
        all_candidates = sorted(all_candidates, key=lambda value: value.get("score", 0), reverse=True)[:limit]

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
