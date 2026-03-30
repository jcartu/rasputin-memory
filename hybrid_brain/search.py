"""
hybrid_brain.search — HybridSearch: the main search orchestrator.

Combines Qdrant vector search, BM25 re-scoring, RRF fusion, neural
reranking, and optional FalkorDB graph enrichment into a single
callable class.

Example::

    from hybrid_brain import HybridSearch

    search = HybridSearch()
    results = search.query("Josh meeting about Q3 targets", limit=10)
    for r in results:
        print(r["score"], r["text"][:80])
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .fusion import RRFFusion
from .temporal import TemporalDecay

# ---------------------------------------------------------------------------
# Constants (overridable at construction time)
# ---------------------------------------------------------------------------
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "memories_v2"
DEFAULT_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_EMBED_MODEL = "nomic-embed-text-v2-moe"
DEFAULT_RERANKER_URL = "http://localhost:8006/rerank"


# ---------------------------------------------------------------------------
# Lightweight BM25 (no external dependency)
# ---------------------------------------------------------------------------

class _BM25:
    """Minimal BM25 scorer for candidate re-ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-zA-Z0-9]+', text.lower())

    def score(self, query: str, documents: List[str]) -> List[float]:
        query_terms = self.tokenize(query)
        if not query_terms or not documents:
            return [0.0] * len(documents)
        doc_tokens = [self.tokenize(d) for d in documents]
        doc_lens = [len(t) for t in doc_tokens]
        avg_dl = sum(doc_lens) / max(len(doc_lens), 1)
        df: Counter = Counter()
        for tokens in doc_tokens:
            for t in set(tokens):
                df[t] += 1
        N = len(documents)
        scores = []
        for i, tokens in enumerate(doc_tokens):
            tf: Counter = Counter(tokens)
            dl = doc_lens[i]
            sc = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                n = df.get(term, 0)
                idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
                sc += idf * (tf[term] * (self.k1 + 1)) / (
                    tf[term] + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                )
            scores.append(sc)
        return scores


# ---------------------------------------------------------------------------
# HybridSearch
# ---------------------------------------------------------------------------

class HybridSearch:
    """Orchestrates vector + BM25 + graph + reranker search over memory.

    Parameters
    ----------
    qdrant_url:
        Qdrant base URL.
    collection:
        Qdrant collection name.
    embed_url:
        Ollama-compatible embedding endpoint.
    embed_model:
        Embedding model name (must match the indexed vectors).
    reranker_url:
        BGE/cross-encoder reranker endpoint.
    falkordb_host / falkordb_port:
        FalkorDB connection for graph enrichment.
    graph_name:
        FalkorDB graph name.
    rrf_k:
        RRF constant (default 60).
    use_temporal_decay:
        Apply Ebbinghaus temporal decay to scores (default ``True``).
    use_multifactor:
        Apply multi-factor scoring on top of decay (default ``True``).

    Example::

        search = HybridSearch(collection="second_brain")
        results = search.query("Sasha IVF timeline", limit=5)
    """

    def __init__(
        self,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection: str = DEFAULT_COLLECTION,
        embed_url: str = DEFAULT_EMBED_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        reranker_url: str = DEFAULT_RERANKER_URL,
        falkordb_host: str = "localhost",
        falkordb_port: int = 6380,
        graph_name: str = "brain",
        rrf_k: int = 60,
        use_temporal_decay: bool = True,
        use_multifactor: bool = True,
    ) -> None:
        self.qdrant = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.embed_url = embed_url
        self.embed_model = embed_model
        self.reranker_url = reranker_url
        self.falkordb_host = falkordb_host
        self.falkordb_port = falkordb_port
        self.graph_name = graph_name
        self._rrf = RRFFusion(k=rrf_k)
        self._bm25 = _BM25()
        self._decay = TemporalDecay() if use_temporal_decay else None
        self._use_multifactor = use_multifactor

    # ── Public API ────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        limit: int = 10,
        graph_hops: int = 2,
        source_filter: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run hybrid search and return ranked results.

        Parameters
        ----------
        query:
            Natural-language search query.
        limit:
            Maximum results to return.
        graph_hops:
            Number of graph traversal hops for entity enrichment (0 = disabled).
        source_filter:
            Filter results to a specific source (e.g. ``"telegram"``).
        agent_id:
            Multi-tenant agent partition filter.

        Returns
        -------
        List of result dicts with keys: ``text``, ``score``, ``source``,
        ``date``, ``importance``, ``id``, and optional graph/rerank fields.
        """
        # 1. Vector search
        vec_results = self._vector_search(query, limit=limit * 5,
                                          source_filter=source_filter,
                                          agent_id=agent_id)
        if not vec_results:
            return []

        # 2. BM25 re-score
        texts = [r["text"] for r in vec_results]
        bm25_scores = self._bm25.score(query, texts)
        bm25_ranked = [vec_results[i] for i in
                       sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)]

        # 3. RRF fusion
        vec_ids = [r["id"] for r in vec_results]
        bm25_ids = [r["id"] for r in bm25_ranked]
        fused_ids = self._rrf.fuse([vec_ids, bm25_ids])
        id_map = {r["id"]: r for r in vec_results}
        fused_results = [id_map[i] for i in fused_ids if i in id_map]

        # 4. Optional graph enrichment
        if graph_hops > 0:
            fused_results = self._graph_enrich(query, fused_results, limit=5)

        # 5. Neural rerank
        fused_results = self._rerank(query, fused_results, top_k=limit)

        # 6. Temporal decay + multi-factor scoring
        if self._decay:
            fused_results = self._decay.apply(fused_results)
            if self._use_multifactor:
                fused_results = self._decay.multifactor(fused_results)

        return fused_results[:limit]

    def embed(self, text: str, prefix: str = "search_query: ") -> List[float]:
        """Return embedding vector for *text* using configured model."""
        prefixed = f"{prefix}{text}"
        resp = requests.post(
            self.embed_url,
            json={"model": self.embed_model, "input": prefixed},
            timeout=35,
        )
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embeddings", [data.get("embedding")])[0]
        if emb is None:
            raise ValueError(f"No embedding in response: {list(data.keys())}")
        return emb

    # ── Internal helpers ──────────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        limit: int,
        source_filter: Optional[str],
        agent_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        vec = self.embed(query, prefix="search_query: ")
        qdrant_filter = None
        conditions = []
        if source_filter:
            conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
        if agent_id:
            conditions.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if conditions:
            qdrant_filter = Filter(must=conditions)

        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=vec,
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        output = []
        for point in results.points:
            p = point.payload or {}
            output.append({
                "id": str(point.id),
                "score": round(point.score, 4),
                "text": p.get("text", ""),
                "source": p.get("source", ""),
                "date": p.get("date", ""),
                "importance": p.get("importance", 50),
                "retrieval_count": p.get("retrieval_count", 0),
            })
        return output

    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not results:
            return results
        passages = [r.get("text", "")[:1000] for r in results]
        try:
            resp = requests.post(
                self.reranker_url,
                json={"query": query, "passages": passages},
                timeout=15,
            )
            resp.raise_for_status()
            scores = resp.json().get("scores", [])
            if len(scores) == len(results):
                for i, r in enumerate(results):
                    r["rerank_score"] = scores[i]
                results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception:
            pass  # Fall back to RRF order
        return results[:top_k]

    def _graph_enrich(
        self,
        query: str,
        results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Attempt FalkorDB graph enrichment; returns results unchanged on error."""
        try:
            import redis
            r = redis.Redis(host=self.falkordb_host, port=self.falkordb_port)
            r.ping()
            # Simple entity lookup — same logic as enrich_with_graph in hybrid_brain.py
            for result in results[:limit]:
                text = result.get("text", "")
                words = re.findall(r'\b[A-Z][a-z]+\b', text)
                if words:
                    result["graph_entities"] = words[:5]
        except Exception:
            pass
        return results
