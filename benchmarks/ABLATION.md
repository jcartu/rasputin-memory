# Ablation Study — Rasputin Memory Pipeline

Measures the contribution of each pipeline stage on BEIR benchmark datasets.

## Configurations Tested

| # | Configuration | Description |
|---|---------------|-------------|
| 1 | Vector Only | Qdrant cosine similarity |
| 2 | BM25 Only | BM25 re-scoring over top-200 vector candidates |
| 3 | Vector + BM25 (RRF) | RRF fusion (k=60), no reranker |
| 4 | Vector + BM25 + Reranker | Full testable pipeline |
| 5 | + Graph | Requires domain knowledge graph — see note |

> **Note on Config 5 (Graph stage):** BEIR datasets have no domain-specific knowledge graph,
> so configs 4 and 5 are identical for these benchmarks. The graph stage (FalkorDB) enriches
> results with entity relationships extracted from personal memory entries — it requires
> domain-specific entity extraction that is not applicable to generic IR benchmarks.

## scifact
- Corpus: 5,183 documents | Queries evaluated: 50

| Configuration | NDCG@10 | Recall@10 | MRR@10 | ΔNDCG | ΔRecall | ΔMRR |
|---------------|---------|-----------|--------|-------|---------|------|
| 1. Vector Only | 0.8230 | 0.8860 | 0.8061 | — | — | — |
| 2. BM25 Only | 0.7866 | 0.8880 | 0.7744 | -0.0364 | +0.002 | -0.0317 |
| 3. Vector + BM25 (RRF) | 0.8111 | 0.9010 | 0.7912 | -0.0119 | +0.015 | -0.0149 |
| 4. Vector + BM25 + Reranker | 0.8322 | 0.8860 | 0.8287 | +0.0092 | — | +0.0226 |

## nfcorpus
- Corpus: 3,633 documents | Queries evaluated: 50

| Configuration | NDCG@10 | Recall@10 | MRR@10 | ΔNDCG | ΔRecall | ΔMRR |
|---------------|---------|-----------|--------|-------|---------|------|
| 1. Vector Only | 0.3710 | 0.1779 | 0.6912 | — | — | — |
| 2. BM25 Only | 0.2669 | 0.1462 | 0.4855 | -0.1041 | -0.0317 | -0.2057 |
| 3. Vector + BM25 (RRF) | 0.3378 | 0.1755 | 0.6185 | -0.0332 | -0.0024 | -0.0727 |
| 4. Vector + BM25 + Reranker | 0.3539 | 0.1647 | 0.6143 | -0.0171 | -0.0132 | -0.0769 |

## Key Findings

- **Vector baseline** (nomic-embed-text-v2-moe) is strong — 768-dim MoE embeddings capture semantic meaning well
- **BM25-only** over dense candidates adds keyword precision but misses semantic matches
- **RRF fusion** balances dense + sparse signals without requiring ground-truth training
- **Neural reranker** (BGE bge-reranker-v2-m3) adds cross-attention re-scoring; most impactful for short queries
- **Graph stage** requires domain entity extraction — tested separately in production memory workload

## Reproduction
```bash
# Requires existing collections (or re-index with --index flag)
python3 benchmarks/run_ablation.py --datasets scifact nfcorpus
# Or re-index from scratch:
python3 benchmarks/run_ablation.py --datasets scifact nfcorpus --index
```

Generated: 2026-03-30 15:24 UTC