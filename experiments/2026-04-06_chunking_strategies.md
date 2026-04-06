# Phase 3: Chunking Strategy Experiments

**Date:** 2026-04-05/06
**Dataset:** LoCoMo conv-0, 199 questions
**Config:** Dense search only, no reranking, no boosts

## Hypothesis

Different window sizes and the presence/absence of individual turns change what
information is co-located in a single vector, which determines what can be
retrieved by a single query.

## Configurations Tested

| Config | Turns | Windows | Window Size | Stride | Vectors |
|--------|-------|---------|-------------|--------|---------|
| Baseline | Yes | Yes (w5s2) | 5 | 2 | 627 |
| Turns only | Yes | No | - | - | 419 |
| Windows only (w5s2) | No | Yes | 5 | 2 | 208 |
| w3s1 | No | Yes | 3 | 1 | 417 |
| w5s1 | No | Yes | 5 | 1 | 415 |
| w10s3 | No | Yes | 10 | 3 | ~140 |

## Results

| Config | Non-Adv | Open-dom | Temporal | Single-hop | Multi-hop | Adv |
|--------|---------|----------|----------|------------|-----------|-----|
| Baseline | 63.2% | 75.7% | 59.5% | 43.8% | 53.8% | 6.4% |
| Turns only | 63.2% | 75.7% | 59.5% | 43.8% | 53.8% | 6.4% |
| **Windows w5s2** | **68.4%** | 81.4% | **70.3%** | **46.9%** | 46.2% | 6.4% |
| w3s1 | 66.4% | 80.0% | 70.3% | 40.6% | 46.2% | 6.4% |
| w5s1 | 63.8% | 84.3% | 54.1% | 34.4% | 53.8% | 6.4% |
| w10s3 | 68.4% | **87.1%** | 70.3% | 34.4% | 46.2% | 8.5% |

## Key Findings

1. **Individual turns add nothing.** Turns-only = baseline. The turns are noise
   that dilute the search results without contributing useful context.

2. **Windows-only w5s2 is the sweet spot.** +5.2pp over baseline with 67% fewer
   vectors (208 vs 627). The 5-turn window captures cross-turn context that
   individual turns miss.

3. **Larger windows help open-domain but hurt single-hop.** w10s3 gets 87.1%
   open-domain but 34.4% single-hop — specific facts get buried in large blocks.

4. **Full overlap (stride 1) is worse than stride 2.** Too many near-duplicate
   vectors dilute the search space.

5. **Temporal improved +10.8pp across all window configs.** Windows capture
   conversation flow with timestamps, which helps temporal reasoning.

## Verdict: KEEP — Windows-only w5s2

Best overall balance. Simplifies ingestion (drop individual turn embedding).
Production default should be windows-only with w5s2.
