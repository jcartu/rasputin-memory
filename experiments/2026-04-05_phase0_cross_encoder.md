# Phase 0: Local Cross-Encoder Reranker

**Date:** 2026-04-05
**Commit:** fdfe37e
**Model:** cross-encoder/ms-marco-MiniLM-L-6-v2 (22MB, CPU, 76ms/60 docs)
**Config:** Dense search + cross-encoder reranking, no other pipeline stages

## Hypothesis

A neural cross-encoder scoring query-document pairs will rank results far better
than cosine similarity, improving Gold-in-Top-5 and accuracy. Hindsight uses this
as their primary scoring signal.

## Results

| Metric | Baseline | Cross-Encoder | Delta |
|--------|----------|---------------|-------|
| Non-adv accuracy | 65.1% | 63.2% | -1.9pp |
| Overall accuracy | 51.3% | 50.3% | -1.0pp |

Per-category:
| Category | Baseline | CE | Delta |
|----------|----------|-----|-------|
| single-hop | 46.9% | 43.8% | -3.1pp |
| multi-hop | 61.5% | 53.8% | -7.7pp |
| temporal | 59.5% | 59.5% | 0pp |
| open-domain | 77.1% | 75.7% | -1.4pp |
| adversarial | 6.4% | 6.4% | 0pp |

## Verdict: NO IMPROVEMENT at 60-chunk context

The cross-encoder reranks chunks within the top-60, but the answer model
sees ALL 60 chunks regardless of order. Reranking is irrelevant when the
context window is this large.

This matches the Cohere reranker finding (also 0pp at 60 chunks).

Reranking would only matter if context is reduced to top-5 or top-10.
The feature is kept (gated behind CROSS_ENCODER=1) for future use with
smaller context windows.

## Key Insight

The consistent finding across 5 experiments (pipeline ablation, Cohere,
qwen3 768d, qwen3 4096d, cross-encoder) is that NOTHING in the ranking
pipeline matters at 60-chunk context. The bottleneck is:
1. What gets INTO the top-60 (retrieval ceiling = 88.4% Gold-in-ANY)
2. Whether the answer model can extract from 60 chunks (generation quality)

The next lever is CHUNKING — changing what text gets embedded as a single
vector. This is Phase 3 in the plan (chunking strategy experiments).
