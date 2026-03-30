# BEIR Benchmark Results

Evaluates Rasputin Memory hybrid search pipeline on BEIR datasets.

## Infrastructure
- **Embeddings:** Ollama `nomic-embed-text-v2-moe` (768-dim, localhost:11434)
- **Vector DB:** Qdrant (localhost:6333) — temporary collections
- **Reranker:** BGE `bge-reranker-v2-m3` (localhost:8006)
- **Fusion:** RRF (Reciprocal Rank Fusion) k=60

## Results

### scifact
- Corpus: 5,183 documents
- Queries evaluated: 50

| Pipeline | NDCG@10 | Recall@10 | Recall@100 |
|----------|---------|-----------|------------|
| Vector Only | 0.8230 | 0.8860 | 0.9800 |
| **Hybrid Full** | **0.8336** | **0.8660** | **0.9800** |
| Delta | +0.0106 | -0.02 | +0.0 |

### nfcorpus
- Corpus: 3,633 documents
- Queries evaluated: 50

| Pipeline | NDCG@10 | Recall@10 | Recall@100 |
|----------|---------|-----------|------------|
| Vector Only | 0.3710 | 0.1779 | 0.3136 |
| **Hybrid Full** | **0.3323** | **0.1581** | **0.3136** |
| Delta | -0.0387 | -0.0198 | +0.0 |

## Notes
- Hybrid Full = Vector + BM25 (RRF) + Neural Reranker
- All collections are temporary (prefix `beir_bench_`) and cleaned up after benchmarking
- The `second_brain` collection is never touched
- Generated: 2026-03-30 15:21 UTC

## Reproduction
```bash
python3 benchmarks/run_beir.py --datasets scifact nfcorpus
```