from __future__ import annotations

import json
import concurrent.futures
import time
from datetime import datetime, timezone
from typing import Any, Optional

import re as _re

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from brain import _state
from brain import embedding
from brain import graph
from brain import scoring

import os as _os

SCORE_BREAKDOWN = _os.environ.get("SCORE_BREAKDOWN", "0") == "1"
CROSS_ENCODER_ENABLED = _os.environ.get("CROSS_ENCODER", "1") == "1"

# Phase B — four-partition parallel retrieval (Hindsight parity). Env vars are read at
# call time inside _four_lane_search() / hybrid_search() so tests and runtime toggles
# take effect without module reload.

_lane_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="lane")

_access_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="access-track")

_YEAR_PATTERN = _re.compile(r"\b(20[0-2]\d|19\d{2})\b")
_MONTH_NAMES = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


def _extract_date_range(query: str) -> tuple[str | None, str | None]:
    years = _YEAR_PATTERN.findall(query)
    if not years:
        return None, None

    min_year = min(years)
    max_year = max(years)

    query_lower = query.lower()
    months_found = [m for m in _MONTH_NAMES if m in query_lower]

    if months_found and len(years) == 1:
        month = _MONTH_NAMES[months_found[0]]
        return f"{years[0]}-{month}-01", f"{years[0]}-{month}-28"

    return f"{min_year}-01-01", f"{max_year}-12-31"


def expand_queries(query: str, max_expansions: int = 5) -> list[str]:
    return [query]


def bm25_rerank(_query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

        reranked: list[dict[str, Any]] = []
        seen: set[int] = set()
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


def qdrant_search(
    query: str,
    limit: int = 10,
    source_filter: Optional[str] = None,
    collection: Optional[str] = None,
    chunk_type_filter: Optional[str] = None,
    fact_type_filter: Optional[str] = None,
) -> list[dict[str, Any]]:
    try:
        vector = embedding.get_embedding(query)
    except Exception as error:
        _state.logger.error("Qdrant embedding error: %s", error)
        return []

    conditions: list = []
    if source_filter:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
    if chunk_type_filter:
        conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type_filter)))
    if fact_type_filter:
        conditions.append(FieldCondition(key="fact_type", match=MatchValue(value=fact_type_filter)))
    search_filter = Filter(must=conditions) if conditions else None

    target_collection = collection or _state.COLLECTION

    results = _state.qdrant.query_points(  # type: ignore[attr-defined]  # qdrant-client>=1.9.0
        collection_name=target_collection,
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
                "chunk_type": payload.get("chunk_type", ""),
                "fact_type": payload.get("fact_type", ""),
                "origin": "qdrant",
            }
        )

    if chunk_type_filter == "fact" and _os.environ.get("FACT_TEMPORAL_RANGES", "0") == "1":
        start_date, end_date = _extract_date_range(query)
        if start_date and end_date:
            existing_ids = {r["point_id"] for r in out}
            try:
                temporal_filter = Filter(
                    must=[
                        FieldCondition(key="chunk_type", match=MatchValue(value="fact")),
                        FieldCondition(
                            key="occurred_start",
                            range=Range(gte=start_date, lte=end_date),  # type: ignore[arg-type]
                        ),
                    ]
                )
                temporal_results = _state.qdrant.query_points(
                    collection_name=target_collection,
                    query=vector,
                    query_filter=temporal_filter,
                    limit=5,
                    with_payload=True,
                )
                for point in temporal_results.points:
                    if point.id in existing_ids:
                        continue
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
                            "chunk_type": payload.get("chunk_type", ""),
                            "fact_type": payload.get("fact_type", ""),
                            "origin": "qdrant_temporal",
                        }
                    )
            except Exception:
                pass

    # Demote inference facts so they don't displace direct factual answers
    for row in out:
        if row.get("chunk_type") == "fact" and row.get("fact_type") == "inference":
            row["score"] = row["score"] * 0.9

    if SCORE_BREAKDOWN:
        for row in out:
            row["_sb"] = {"dense_score": row.get("score", 0)}
    return out


_FACTUAL_PREFIXES = ("what is", "what was", "when did", "who is", "how many", "where did", "what does", "how old")


def _decompose_query_intent(query: str) -> list[str]:
    if any(query.lower().startswith(p) for p in _FACTUAL_PREFIXES):
        return [query]

    anthropic_key = _state.ANTHROPIC_API_KEY

    prompt = (
        f'A user says: "{query}"\n\n'
        "What implicit topics, goals, or constraints from PRIOR conversations might be relevant? "
        "List 2-3 short search queries.\n\nReturn ONLY a JSON array of strings."
    )

    if anthropic_key:
        try:
            import urllib.request as _urllib_req

            body = json.dumps(
                {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 200,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode()

            req = _urllib_req.Request(
                "https://api.anthropic.com/v1/messages",
                data=body,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("x-api-key", anthropic_key)
            req.add_header("anthropic-version", "2023-06-01")

            with _urllib_req.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            raw = data["content"][0]["text"].strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                intents = json.loads(raw[start:end])
                if isinstance(intents, list):
                    return [query] + [str(i) for i in intents[:3]]
        except Exception:
            pass

    return [query]


def _four_lane_search(
    query: str,
    collection: Optional[str],
    source_filter: Optional[str],
) -> dict[str, list[dict[str, Any]]]:
    """Hindsight-parity parallel retrieval: 4 semantic partitions.

    Four lanes: window + fact_world + fact_experience + fact_inference.
    BM25 remains as a rerank-stage fusion signal via tools/bm25_search.py;
    a standalone FTS5-backed BM25 lane is deferred (see docs/BACKLOG.md).

    Returns dict {lane_name -> ordered candidate list}. Callers flat-concat
    (this phase) or RRF-fuse (Phase C).
    """
    # Read budgets at call time so env overrides take effect without reload.
    lane_windows = int(_os.environ.get("BENCH_LANE_WINDOWS", "45"))
    lane_fact_w = int(_os.environ.get("BENCH_LANE_FACT_W", "8"))
    lane_fact_e = int(_os.environ.get("BENCH_LANE_FACT_E", "4"))
    lane_fact_i = int(_os.environ.get("BENCH_LANE_FACT_I", "3"))

    def _window_lane() -> list[dict[str, Any]]:
        return qdrant_search(
            query,
            limit=lane_windows,
            source_filter=source_filter,
            collection=collection,
            chunk_type_filter="window",
        )

    def _fact_world_lane() -> list[dict[str, Any]]:
        return qdrant_search(
            query,
            limit=lane_fact_w,
            source_filter=source_filter,
            collection=collection,
            chunk_type_filter="fact",
            fact_type_filter="world",
        )

    def _fact_experience_lane() -> list[dict[str, Any]]:
        return qdrant_search(
            query,
            limit=lane_fact_e,
            source_filter=source_filter,
            collection=collection,
            chunk_type_filter="fact",
            fact_type_filter="experience",
        )

    def _fact_inference_lane() -> list[dict[str, Any]]:
        return qdrant_search(
            query,
            limit=lane_fact_i,
            source_filter=source_filter,
            collection=collection,
            chunk_type_filter="fact",
            fact_type_filter="inference",
        )

    futures = {
        "window": _lane_pool.submit(_window_lane),
        "world": _lane_pool.submit(_fact_world_lane),
        "exp": _lane_pool.submit(_fact_experience_lane),
        "inf": _lane_pool.submit(_fact_inference_lane),
    }
    results: dict[str, list[dict[str, Any]]] = {}
    for name, fut in futures.items():
        try:
            results[name] = fut.result(timeout=10)
        except Exception as exc:
            _state.logger.warning("Lane %s failed/timeout: %s", name, exc)
            results[name] = []
    return results


def _resolve_contradictions_in_results(results: list[dict[str, Any]]) -> None:
    supersede_map: dict[Any, set[Any]] = {}
    for row in results:
        supersedes = row.get("supersedes") or []
        pid = row.get("point_id")
        if pid and supersedes:
            supersede_map[pid] = set(supersedes)

    if not supersede_map:
        return

    superseded_ids = set()
    for targets in supersede_map.values():
        superseded_ids.update(targets)

    for row in results:
        pid = row.get("point_id")
        if pid and pid in superseded_ids:
            row["contradiction_demoted"] = True
            for score_key in ("rerank_score", "hybrid_score", "score"):
                if score_key in row:
                    row[score_key] = row[score_key] * 0.3
                    break

    results.sort(
        key=lambda v: float(v.get("rerank_score", 0) or v.get("hybrid_score", 0) or v.get("score", 0)),
        reverse=True,
    )


def hybrid_search(
    query: str,
    limit: int = 10,
    graph_hops: int = 2,
    source_filter: Optional[str] = None,
    expand: bool = True,
    collection: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> dict[str, Any]:
    start = time.time()

    queries = [query]
    if expand:
        expanded = expand_queries(query, max_expansions=5)
        if expanded:
            seen: set[str] = set()
            queries = [q for q in expanded if not (q in seen or seen.add(q))]  # type: ignore[func-returns-value]

    fetch_limit = limit * 4
    all_qdrant_results: list[dict[str, Any]] = []

    if _os.environ.get("FOUR_LANE", "0") == "1":
        # Phase B: four-partition parallel retrieval across expanded queries.
        # Phase C: optionally fuse lanes via RRF when RRF_FUSION=1.
        rrf_enabled = _os.environ.get("RRF_FUSION", "0") == "1"
        for expanded_query in queries:
            lane_results = _four_lane_search(expanded_query, collection, source_filter)
            if rrf_enabled:
                from brain.fusion import reciprocal_rank_fusion

                fused = reciprocal_rank_fusion(lane_results, k=60)
                # Cap at limit*2+60, matching parity plan §4.3 pre-CE budget
                all_qdrant_results.extend(fused[: min(limit * 2 + 60, 120)])
            else:
                for lane_name, lane_hits in lane_results.items():
                    all_qdrant_results.extend(lane_hits)
    else:
        # Legacy: single qdrant_search per expanded query.
        for expanded_query in queries:
            qdrant_kwargs: dict[str, Any] = {
                "limit": fetch_limit,
                "source_filter": source_filter,
            }
            if collection:
                qdrant_kwargs["collection"] = collection
            if chunk_type:
                qdrant_kwargs["chunk_type_filter"] = chunk_type
            all_qdrant_results.extend(qdrant_search(expanded_query, **qdrant_kwargs))

    constraint_hits: list[dict[str, Any]] = []
    try:
        from brain import constraints as _constraints_mod

        if _constraints_mod.CONSTRAINTS_ENABLED:
            intent_queries = _decompose_query_intent(query)
            for iq in intent_queries[1:]:
                iq_kwargs: dict[str, Any] = {"limit": 20, "source_filter": source_filter}
                if collection:
                    iq_kwargs["collection"] = collection
                all_qdrant_results.extend(qdrant_search(iq, **iq_kwargs))

            constraint_col = f"{(collection or _state.COLLECTION)}_constraints"
            all_search_queries = list(dict.fromkeys(queries + intent_queries))[:8]
            for sq in all_search_queries:
                try:
                    vec = embedding.get_embedding(sq)
                    hits = _state.qdrant.query_points(
                        collection_name=constraint_col,
                        query=vec,
                        limit=15,
                        with_payload=True,
                    )
                    for point in hits.points if hasattr(hits, "points") else []:
                        p = point.payload or {}
                        constraint_hits.append(
                            {
                                "score": round(point.score * 0.9, 4),
                                "text": p.get("parent_text", ""),
                                "constraint_summary": p.get("constraint_summary", ""),
                                "source": p.get("source", ""),
                                "date": p.get("date", ""),
                                "point_id": point.id,
                                "origin": "constraint",
                            }
                        )
                except Exception:
                    pass
    except Exception:
        pass

    all_qdrant_results.extend(constraint_hits)

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

    if qdrant_results and any("source_weight" in row for row in qdrant_results):
        qdrant_results = scoring.apply_multifactor_scoring(qdrant_results)

    graph_memory_results = [row for row in graph_results if row.get("source") in ("graph_memory", "graph_keyword")]
    graph_context_results = [row for row in graph_results if row.get("source") == "graph_context"]

    for graph_row in graph_memory_results:
        hop_count = graph_row.get("graph_hop", 1)
        graph_row["score"] = 0.8 if hop_count == 1 else 0.5

    all_candidates = list(qdrant_results[: limit * 2]) + graph_memory_results

    if _os.environ.get("KNN_LINKS", "0") == "1":
        try:
            from brain import knn_links

            seed_ids = [
                r["point_id"] for r in all_candidates[:20] if r.get("point_id") and r.get("chunk_type") == "fact"
            ]
            if seed_ids:
                existing_pids = {r["point_id"] for r in all_candidates if r.get("point_id")}
                knn_expanded = knn_links.expand_seeds(
                    collection=collection or _state.COLLECTION,
                    seed_ids=seed_ids,
                    exclude_ids=existing_pids,
                )
                if knn_expanded:
                    all_candidates.extend(knn_expanded)
                    seen_pids: set[Any] = set()
                    deduped_candidates: list[dict[str, Any]] = []
                    for r in all_candidates:
                        pid = r.get("point_id")
                        if pid and pid not in seen_pids:
                            seen_pids.add(pid)
                            deduped_candidates.append(r)
                        elif not pid:
                            deduped_candidates.append(r)
                    all_candidates = deduped_candidates
        except Exception as exc:
            _state.logger.warning("kNN expansion failed: %s", exc)

    ranking_score_key = "hybrid_score" if any("hybrid_score" in row for row in all_candidates) else "score"
    if SCORE_BREAKDOWN:
        for row in all_candidates:
            sb = row.setdefault("_sb", {})
            sb["pre_rerank_score"] = row.get(ranking_score_key, row.get("score", 0))
    ce_applied = False
    if all_candidates and CROSS_ENCODER_ENABLED:
        try:
            from brain import cross_encoder as _ce

            if _ce.is_available():
                all_candidates = _ce.rerank_with_recency(query, all_candidates, top_k=limit)
                ce_applied = True
                ranking_score_key = "final_score"
        except ImportError:
            pass

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

    _resolve_contradictions_in_results(merged)

    if graph_context_results and len(merged) < limit:
        for graph_context in graph_context_results[: limit - len(merged)]:
            graph_context["score"] = 0.1
            merged.append(graph_context)

    graph_enrichment = graph.enrich_with_graph(merged, limit=5)

    if SCORE_BREAKDOWN:
        for row in merged:
            sb = row.setdefault("_sb", {})
            sb["final_score"] = row.get(ranking_score_key, row.get("score", 0))
            sb["reranker_score"] = row.get("rerank_score") or row.get("llm_rerank_score")
            sb["contradiction_demoted"] = row.get("contradiction_demoted", False)

    elapsed = time.time() - start

    _update_access_tracking([row for row in merged if row.get("origin") != "graph"], collection=collection)

    return {
        "query": query,
        "elapsed_ms": round(elapsed * 1000, 1),
        "results": merged,
        "graph_context": graph_enrichment,
        "graph_results": graph_results,
        "stats": {
            "expanded_queries": len(queries),
            "query_expansion_enabled": bool(expand),
            "qdrant_hits": len([row for row in merged if row.get("origin") != "graph"]),
            "graph_hits": len(graph_results),
            "graph_merged": len([row for row in merged if row.get("origin") == "graph"]),
            "graph_enriched_entities": len(graph_enrichment),
            "graph_hop_2_connections": sum(len(v) for v in graph_enrichment.values()),
            "constraint_hits": len(constraint_hits),
            "cross_encoder_reranked": ce_applied,
        },
    }


def _update_access_tracking(results: list[dict[str, Any]], collection: Optional[str] = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    target_collection = collection or _state.COLLECTION

    def _do_batch() -> None:
        ids = [r["point_id"] for r in results if r.get("point_id")]
        if ids:
            try:
                _state.qdrant.set_payload(
                    collection_name=target_collection,
                    points=ids,
                    payload={"last_accessed": now},
                )
            except Exception:
                pass

    try:
        _access_pool.submit(_do_batch)
    except Exception:
        pass
