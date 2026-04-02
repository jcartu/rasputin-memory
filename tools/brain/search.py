from __future__ import annotations

import json
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


def _llm_rerank(query: str, results: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
    if not _state.ANTHROPIC_API_KEY or not results:
        return results[:top_k]

    passages = []
    candidate_pool = results[:30]
    for i, row in enumerate(candidate_pool):
        text = (row.get("text") or "")[:500]
        source = row.get("source", "")
        date = row.get("date", "")
        passages.append(f"[{i}] ({source}, {date}) {text}")

    prompt = f"""You are ranking memory search results by relevance to a query.

Query: {query}

Passages:
{chr(10).join(passages)}

Return a JSON array of passage indices ordered by relevance (most relevant first). Include only passages that are actually relevant to the query. Example: [3, 0, 7, 12]

Respond with ONLY the JSON array, nothing else."""

    try:
        import urllib.request

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": _state.LLM_RERANKER_MODEL,
                    "max_tokens": 200,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode(),
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("x-api-key", _state.ANTHROPIC_API_KEY)
        req.add_header("anthropic-version", "2023-06-01")

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        text = data["content"][0]["text"].strip()
        indices = json.loads(text)

        reranked = []
        seen = set()
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(candidate_pool) and idx not in seen:
                row = candidate_pool[idx].copy()
                row["llm_rerank_position"] = len(reranked)
                reranked.append(row)
                seen.add(idx)

        for i, row in enumerate(results):
            if i not in seen and len(reranked) < top_k:
                reranked.append(row)

        return reranked[:top_k]
    except Exception as error:
        _state.logger.warning("LLM reranker failed, falling back to original order: %s", error)
        return results[:top_k]


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

    graph_memory_results = [row for row in graph_results if row.get("source") in ("graph_memory", "graph_keyword")]
    graph_context_results = [row for row in graph_results if row.get("source") == "graph_context"]

    for graph_row in graph_memory_results:
        hop_count = graph_row.get("graph_hop", 1)
        graph_row["score"] = 0.8 if hop_count == 1 else 0.5

    all_candidates = list(qdrant_results[: limit * 2]) + graph_memory_results

    neural_applied = False
    llm_applied = False
    ranking_score_key = "hybrid_score" if any("hybrid_score" in row for row in all_candidates) else "score"
    if all_candidates:
        rerank_pool = all_candidates[: limit * 3]
        if _state.LLM_RERANKER_ENABLED and _state.ANTHROPIC_API_KEY:
            all_candidates = _llm_rerank(query, rerank_pool, top_k=limit)
            llm_applied = any(row.get("llm_rerank_position") is not None for row in all_candidates)
            if llm_applied:
                top_span = max(len(all_candidates), 1)
                for idx, row in enumerate(all_candidates):
                    llm_position = row.get("llm_rerank_position")
                    if llm_position is None:
                        llm_position = idx + top_span
                        row["llm_rerank_position"] = llm_position
                    row["llm_rerank_score"] = float((2 * top_span) - int(llm_position))
                ranking_score_key = "llm_rerank_score"
        elif embedding.is_reranker_available():
            pre_count = len(all_candidates)
            all_candidates = _neural_rerank(query, rerank_pool, top_k=limit)
            neural_applied = len(all_candidates) <= pre_count and any(
                row.get("rerank_score") is not None for row in all_candidates
            )
            if neural_applied:
                ranking_score_key = "rerank_score"
        else:
            all_candidates = sorted(
                all_candidates,
                key=lambda value: float(value.get(ranking_score_key) or value.get("score", 0)),
                reverse=True,
            )[:limit]

    query_tokens = set(_TOKEN_RE.findall(query.lower()))
    _stopwords = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "are",
        "were",
        "do",
        "does",
        "did",
        "what",
        "where",
        "when",
        "how",
        "who",
        "which",
        "that",
        "this",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "and",
        "or",
        "not",
        "it",
        "we",
        "he",
        "she",
        "they",
        "i",
        "you",
        "my",
        "about",
        "know",
        "now",
        "has",
        "have",
        "had",
        "be",
        "been",
    }
    query_content_tokens = query_tokens - _stopwords
    if query_content_tokens:
        for row in all_candidates:
            text_tokens = set(_TOKEN_RE.findall((row.get("text") or "").lower()))
            overlap = query_content_tokens & text_tokens
            if overlap:
                overlap_ratio = len(overlap) / len(query_content_tokens)
                boost = 1.0 + 9.0 * (overlap_ratio**3)
                row[ranking_score_key] = row.get(ranking_score_key, row.get("score", 0)) * boost
                row["keyword_boosted"] = True

    query_entities = [name for name, _type in entities_module.extract_entities_fast(query)]
    if query_entities:
        query_entity_names_lower = {e.lower() for e in query_entities}
        for row in all_candidates:
            text = row.get("text") or ""
            text_lower = text.lower()
            matched = [e for e in query_entities if e.lower() in text_lower]
            if matched:
                all_text_entities = entities_module.extract_entities_fast(text)
                total_entities = max(len(all_text_entities), 1)
                matched_count = sum(1 for name, _ in all_text_entities if name.lower() in query_entity_names_lower)
                focus_ratio = matched_count / total_entities
                position_bonus = 0.0
                for e in matched:
                    pos = text_lower.find(e.lower())
                    if pos >= 0:
                        text_len = max(len(text_lower), 1)
                        position_bonus = max(position_bonus, 1.0 - (pos / text_len))
                boost = 2.0 + 5.0 * (focus_ratio**0.5) + 2.0 * position_bonus
                current = row.get(ranking_score_key, row.get("score", 0))
                row[ranking_score_key] = current * boost
                row["entity_boosted"] = True
                row["entity_focus"] = round(focus_ratio, 2)

    all_candidates = sorted(
        all_candidates,
        key=lambda v: float(
            v.get(ranking_score_key, 0)
            or v.get("llm_rerank_score", 0)
            or v.get("rerank_score", 0)
            or v.get("hybrid_score", 0)
            or v.get("score", 0)
        ),
        reverse=True,
    )

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
            "llm_reranked": llm_applied,
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
