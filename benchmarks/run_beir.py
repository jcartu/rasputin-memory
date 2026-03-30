#!/usr/bin/env python3
"""
BEIR Benchmark Script for Rasputin Memory Hybrid Search
========================================================
Evaluates hybrid search (vector + BM25 + reranker) vs vector-only baseline
on BEIR datasets using local infrastructure:
  - Ollama nomic-embed-text-v2-moe (port 11434)
  - Qdrant (port 6333) — TEMPORARY collections only
  - BGE reranker (port 8006)

Usage:
    python3 benchmarks/run_beir.py [--datasets scifact nfcorpus] [--cleanup]

IMPORTANT: Never touches the 'second_brain' collection.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, VectorParams,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text-v2-moe"
EMBED_DIM = 768
QDRANT_URL = "http://localhost:6333"
RERANKER_URL = "http://localhost:8006/rerank"
BATCH_SIZE = 32          # docs per embed batch
COLLECTION_PREFIX = "beir_bench_"  # temp collections — never "second_brain"

qdrant = QdrantClient(url=QDRANT_URL)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_batch(texts: List[str], prefix: str = "search_document: ") -> List[List[float]]:
    """Embed a list of texts with nomic-embed-text-v2-moe."""
    prefixed = [f"{prefix}{t[:4096]}" for t in texts]
    resp = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": prefixed}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("embeddings", [data.get("embedding")])


def embed_single(text: str, prefix: str = "search_query: ") -> List[float]:
    return embed_batch([text], prefix=prefix)[0]


# ---------------------------------------------------------------------------
# BM25 (inline — matches bm25_search.py in repo)
# ---------------------------------------------------------------------------

class BM25Scorer:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'[a-zA-Z0-9]+', text.lower())

    def score(self, query: str, documents: List[str]) -> List[float]:
        query_terms = self.tokenize(query)
        if not query_terms or not documents:
            return [0.0] * len(documents)
        doc_tokens = [self.tokenize(d) for d in documents]
        doc_lens = [len(t) for t in doc_tokens]
        avg_dl = sum(doc_lens) / len(doc_lens) if doc_lens else 1
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
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion over multiple ranked lists of doc IDs."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def neural_rerank(query: str, doc_ids: List[str], texts: Dict[str, str], top_k: int = 10) -> List[str]:
    """Rerank using bge-reranker-v2-m3. Falls back to input order."""
    passages = [texts.get(d, "")[:1000] for d in doc_ids]
    try:
        resp = requests.post(RERANKER_URL, json={"query": query, "passages": passages}, timeout=15)
        resp.raise_for_status()
        scores = resp.json().get("scores", [])
        if len(scores) == len(doc_ids):
            paired = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
            return [p[0] for p in paired][:top_k]
    except Exception as e:
        print(f"  [reranker] fallback ({e})", flush=True)
    return doc_ids[:top_k]


# ---------------------------------------------------------------------------
# BEIR dataset loading
# ---------------------------------------------------------------------------

def load_beir_dataset(dataset_name: str, split: str = "test") -> Tuple[Dict, Dict, Dict]:
    """Load BEIR dataset. Returns (corpus, queries, qrels)."""
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        print("Install beir: pip install beir")
        sys.exit(1)

    data_path = Path(f"/tmp/beir_data/{dataset_name}")
    if not data_path.exists():
        print(f"  Downloading {dataset_name}...", flush=True)
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = Path(util.download_and_unzip(url, "/tmp/beir_data"))
    else:
        print(f"  Using cached {dataset_name}", flush=True)

    loader = GenericDataLoader(str(data_path))
    corpus, queries, qrels = loader.load(split=split)
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_collection(collection_name: str, corpus: Dict) -> Dict[str, str]:
    """Index corpus into a temporary Qdrant collection. Returns id→text map."""
    print(f"  Building collection '{collection_name}' ({len(corpus)} docs)...", flush=True)

    # Safety: never touch second_brain
    assert "second_brain" not in collection_name, "SAFETY: never touch second_brain!"

    # Recreate collection
    try:
        qdrant.delete_collection(collection_name)
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    doc_ids = list(corpus.keys())
    id_to_text: Dict[str, str] = {}
    points: List[PointStruct] = []

    # We need integer IDs for Qdrant; keep mapping
    str_to_int: Dict[str, int] = {d: i for i, d in enumerate(doc_ids)}
    int_to_str: Dict[int, str] = {i: d for d, i in str_to_int.items()}

    for batch_start in range(0, len(doc_ids), BATCH_SIZE):
        batch_ids = doc_ids[batch_start:batch_start + BATCH_SIZE]
        batch_texts = []
        for doc_id in batch_ids:
            doc = corpus[doc_id]
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            id_to_text[doc_id] = text
            batch_texts.append(text)

        vectors = embed_batch(batch_texts, prefix="search_document: ")

        for doc_id, vec in zip(batch_ids, vectors):
            points.append(PointStruct(
                id=str_to_int[doc_id],
                vector=vec,
                payload={"doc_id": doc_id, "text": id_to_text[doc_id][:2000]},
            ))

        if len(points) >= 200:
            qdrant.upsert(collection_name=collection_name, points=points)
            points = []
            print(f"    indexed {batch_start + len(batch_ids)}/{len(doc_ids)}", flush=True)

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)

    # Store mapping in collection info payload (just use a file)
    mapping_path = f"/tmp/beir_mapping_{collection_name}.json"
    with open(mapping_path, "w") as f:
        json.dump({"str_to_int": str_to_int, "int_to_str": {str(k): v for k, v in int_to_str.items()}}, f)

    print(f"  Collection '{collection_name}' ready.", flush=True)
    return id_to_text, str_to_int, int_to_str


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def _resolve_id(pt_id, int_to_str: Dict) -> str:
    """Resolve Qdrant point ID (int or str) to original doc string ID."""
    # int_to_str has integer keys; Qdrant may return int or str
    if isinstance(pt_id, str):
        pt_id = int(pt_id)
    return int_to_str[pt_id]


def retrieve_vector_only(query: str, collection_name: str, int_to_str: Dict, top_k: int = 100) -> List[str]:
    """Vector-only retrieval."""
    vec = embed_single(query, prefix="search_query: ")
    results = qdrant.query_points(collection_name=collection_name, query=vec, limit=top_k, with_payload=True)
    return [_resolve_id(p.id, int_to_str) for p in results.points]


def retrieve_hybrid(query: str, collection_name: str, int_to_str: Dict,
                    id_to_text: Dict, top_k: int = 10, use_reranker: bool = True) -> List[str]:
    """Full hybrid: vector + BM25 RRF + optional reranker."""
    # 1. Vector retrieval (get more for fusion)
    vec = embed_single(query, prefix="search_query: ")
    results = qdrant.query_points(collection_name=collection_name, query=vec, limit=100, with_payload=True)
    vec_ranked = [_resolve_id(p.id, int_to_str) for p in results.points]
    vec_texts = {_resolve_id(p.id, int_to_str): p.payload.get("text", "") for p in results.points}

    # 2. BM25 on retrieved candidates
    bm25 = BM25Scorer()
    candidate_ids = vec_ranked[:100]
    candidate_texts = [vec_texts.get(d, id_to_text.get(d, "")) for d in candidate_ids]
    bm25_scores = bm25.score(query, candidate_texts)
    bm25_ranked = [candidate_ids[i] for i in sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)]

    # 3. RRF fusion
    fused = rrf_fuse([vec_ranked, bm25_ranked])[:top_k * 3]

    # 4. Reranker
    if use_reranker:
        all_texts = {d: id_to_text.get(d, vec_texts.get(d, "")) for d in fused}
        reranked = neural_rerank(query, fused, all_texts, top_k=top_k)
        return reranked

    return fused[:top_k]


# ---------------------------------------------------------------------------
# NDCG / Recall / MRR evaluation
# ---------------------------------------------------------------------------

def evaluate(run: Dict[str, List[str]], qrels: Dict, k_values: List[int] = [10, 100]) -> Dict:
    """Compute NDCG@k, Recall@k, MRR@k for a retrieval run."""
    metrics: Dict[str, float] = {}

    for k in k_values:
        ndcg_sum = recall_sum = mrr_sum = 0.0
        n = 0

        for qid, ranked_docs in run.items():
            rel = qrels.get(qid, {})
            if not rel:
                continue
            n += 1
            retrieved = ranked_docs[:k]

            # NDCG@k
            dcg = sum(
                rel.get(doc_id, 0) / math.log2(rank + 2)
                for rank, doc_id in enumerate(retrieved)
            )
            ideal = sorted(rel.values(), reverse=True)[:k]
            idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
            ndcg_sum += dcg / idcg if idcg > 0 else 0

            # Recall@k
            relevant = set(d for d, s in rel.items() if s > 0)
            retrieved_set = set(retrieved)
            recall_sum += len(retrieved_set & relevant) / len(relevant) if relevant else 0

            # MRR@k
            for rank, doc_id in enumerate(retrieved):
                if rel.get(doc_id, 0) > 0:
                    mrr_sum += 1.0 / (rank + 1)
                    break

        metrics[f"NDCG@{k}"] = round(ndcg_sum / n, 4) if n > 0 else 0.0
        metrics[f"Recall@{k}"] = round(recall_sum / n, 4) if n > 0 else 0.0
        metrics[f"MRR@{k}"] = round(mrr_sum / n, 4) if n > 0 else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(dataset_name: str, max_queries: int = 100) -> Dict:
    """Run full benchmark for one BEIR dataset."""
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset: {dataset_name} (max {max_queries} queries)", flush=True)
    print(f"{'='*60}", flush=True)

    corpus, queries, qrels = load_beir_dataset(dataset_name)

    # Limit corpus for indexing speed (use full corpus, limit queries for eval)
    print(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)} | Using {min(max_queries, len(queries))} queries", flush=True)

    collection_name = f"{COLLECTION_PREFIX}{dataset_name}"
    id_to_text, str_to_int, int_to_str = build_collection(collection_name, corpus)

    # Sample queries
    query_ids = list(qrels.keys())[:max_queries]
    eval_queries = {qid: queries[qid] for qid in query_ids if qid in queries}

    # --- Vector only ---
    print("\n  Running vector-only retrieval...", flush=True)
    vec_run: Dict[str, List[str]] = {}
    for qid, qtext in eval_queries.items():
        vec_run[qid] = retrieve_vector_only(qtext, collection_name, int_to_str, top_k=100)

    vec_metrics = evaluate(vec_run, qrels, k_values=[10, 100])

    # --- Hybrid (full) ---
    print("  Running hybrid retrieval (vector + BM25 + reranker)...", flush=True)
    hybrid_run: Dict[str, List[str]] = {}
    for i, (qid, qtext) in enumerate(eval_queries.items()):
        hybrid_run[qid] = retrieve_hybrid(qtext, collection_name, int_to_str, id_to_text, top_k=100, use_reranker=True)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(eval_queries)} queries done", flush=True)

    hybrid_metrics = evaluate(hybrid_run, qrels, k_values=[10, 100])

    return {
        "dataset": dataset_name,
        "corpus_size": len(corpus),
        "queries_evaluated": len(eval_queries),
        "vector_only": vec_metrics,
        "hybrid_full": hybrid_metrics,
        "collection_name": collection_name,
    }


def cleanup_collections(collection_names: List[str]):
    """Delete temporary benchmark collections."""
    for name in collection_names:
        assert "second_brain" not in name, "SAFETY: never delete second_brain!"
        try:
            qdrant.delete_collection(name)
            print(f"  Deleted collection: {name}")
        except Exception as e:
            print(f"  Warning: could not delete {name}: {e}")


def write_results(results: List[Dict], output_path: Path):
    """Write RESULTS.md with benchmark table."""
    lines = [
        "# BEIR Benchmark Results",
        "",
        "Evaluates Rasputin Memory hybrid search pipeline on BEIR datasets.",
        "",
        "## Infrastructure",
        "- **Embeddings:** Ollama `nomic-embed-text-v2-moe` (768-dim, localhost:11434)",
        "- **Vector DB:** Qdrant (localhost:6333) — temporary collections",
        "- **Reranker:** BGE `bge-reranker-v2-m3` (localhost:8006)",
        "- **Fusion:** RRF (Reciprocal Rank Fusion) k=60",
        "",
        "## Results",
        "",
    ]

    for r in results:
        ds = r["dataset"]
        lines.append(f"### {ds}")
        lines.append(f"- Corpus: {r['corpus_size']:,} documents")
        lines.append(f"- Queries evaluated: {r['queries_evaluated']}")
        lines.append("")
        lines.append("| Pipeline | NDCG@10 | Recall@10 | Recall@100 |")
        lines.append("|----------|---------|-----------|------------|")

        vo = r["vector_only"]
        hf = r["hybrid_full"]
        lines.append(f"| Vector Only | {vo['NDCG@10']:.4f} | {vo['Recall@10']:.4f} | {vo['Recall@100']:.4f} |")
        lines.append(f"| **Hybrid Full** | **{hf['NDCG@10']:.4f}** | **{hf['Recall@10']:.4f}** | **{hf['Recall@100']:.4f}** |")

        # Delta
        dn = round(hf["NDCG@10"] - vo["NDCG@10"], 4)
        dr10 = round(hf["Recall@10"] - vo["Recall@10"], 4)
        dr100 = round(hf["Recall@100"] - vo["Recall@100"], 4)
        def sign(x):
            return f"+{x}" if x >= 0 else str(x)
        lines.append(f"| Delta | {sign(dn)} | {sign(dr10)} | {sign(dr100)} |")
        lines.append("")

    lines.extend([
        "## Notes",
        "- Hybrid Full = Vector + BM25 (RRF) + Neural Reranker",
        "- All collections are temporary (prefix `beir_bench_`) and cleaned up after benchmarking",
        "- The `second_brain` collection is never touched",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "## Reproduction",
        "```bash",
        "python3 benchmarks/run_beir.py --datasets scifact nfcorpus",
        "```",
    ])

    output_path.write_text("\n".join(lines))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEIR benchmark for Rasputin Memory")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"],
                        help="BEIR dataset names")
    parser.add_argument("--max-queries", type=int, default=50,
                        help="Max queries per dataset (default: 50 for speed)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep temporary collections after benchmarking")
    parser.add_argument("--output", default="benchmarks/RESULTS.md")
    args = parser.parse_args()

    all_results = []
    collections_created = []

    for dataset in args.datasets:
        result = run_benchmark(dataset, max_queries=args.max_queries)
        all_results.append(result)
        collections_created.append(result["collection_name"])

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    write_results(all_results, output_path)

    if not args.no_cleanup:
        print("\nCleaning up temporary collections...")
        cleanup_collections(collections_created)
    else:
        print(f"\nKept collections: {collections_created}")

    print("\nDone.")
