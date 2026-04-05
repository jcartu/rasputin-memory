# Ablation: Dense-Only vs Full Pipeline

**Date:** 2026-04-05
**Commit:** 2758821
**Dataset:** LoCoMo conv-0, 199 questions

## Hypothesis

The full 16-stage pipeline (BM25, keyword boost, entity boost, temporal boost,
temporal decay, MMR, Cohere reranking, query expansion) contributes measurable
retrieval quality over raw dense vector search.

## What Changed

Set all ABLATION_* environment variables to 0, disabling:
- BM25 + RRF fusion
- Keyword relevance boost
- Entity relevance boost
- Temporal signal boost
- Temporal decay (Ebbinghaus)
- MMR diversity filtering
- Cohere/LLM/BGE reranking
- Query expansion

## Results

| Config | Accuracy | Gold-ANY | Gold-Top5 | Gold-Top10 | MRR |
|--------|----------|----------|-----------|------------|-----|
| Full pipeline (baseline) | 65.1% | 88.4% | 63.8% | 71.4% | varies |
| Dense only | 65.1% | 88.4% | 63.8% | 71.4% | identical |

Per-category: identical across all 5 categories.

## Verdict: CONFIRMED — pipeline adds 0pp

The entire post-retrieval pipeline is dead weight for this benchmark.
BM25 IDF is computed on the retrieved set (not corpus) — meaningless.
Entity extraction misses LoCoMo names (not in known_entities.json).
Keyword boost and temporal boost are too small (+0.08–0.15) to change rank order.
Cohere reranking was disabled in the dense-only run but was ON in baseline — and still no difference, because the search results are the same before reranking touches them.

## Next Steps

1. Strip the pipeline to dense search + optional reranker
2. Focus improvements on: embedding quality, chunking strategy, query decomposition
3. The ranking problem is really a retrieval problem — wrong chunks never enter the pipeline
