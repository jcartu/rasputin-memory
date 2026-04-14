# Wider Retrieval Pool (75w+25f vs 45w+15f)

**Date:** 2026-04-14
**Baseline:** v0.9 production (74.2% non-adv, 45w+15f+10bm25)
**Dataset:** LoCoMo 10-conv, 1986 questions

## Hypothesis

All embedding similarity scores are in the 0.03-0.05 range for this dataset —
the initial dense retrieval is essentially random at these similarity levels.
The Qwen3-Reranker does all meaningful ranking. A wider initial pool gives the
reranker more candidates, increasing the chance that the gold fact is present
for the CE to find.

## What Changed

Increased per-query-variant retrieval from 45 windows + 15 facts to
75 windows + 25 facts. BM25 stays at 10. Total candidates per variant:
110 (up from 70). The Qwen3-Reranker still selects the best 60 for the
answer model — only the pre-reranker pool changes.

## Results (full 10-conv, production mode)

| Category | Default (45w+15f) | Wide (75w+25f) | Δ |
|----------|-------------------|----------------|---|
| non-adv  | 74.2%             | 74.1%          | −0.1pp |
| single-hop | 54.3%           | **58.5%**      | **+4.2pp** |
| open-domain | 84.8%          | 83.6%          | −1.2pp |
| temporal | 71.3%             | 70.4%          | −0.9pp |
| multi-hop | 49.0%            | 49.0%          | 0.0pp |

## Analysis

Single-hop improved +4.2pp because specific factual questions ("What is X's job?")
benefit from seeing more candidates — the correct fact is more likely to appear in
a pool of 110 than 70, and the Qwen3 CE identifies it accurately.

Open-domain regressed −1.2pp because broader questions get more noise candidates
that compete with the correct answer. The CE has to distinguish from a larger
set of plausible-but-wrong results.

Overall accuracy is flat (74.1% vs 74.2%) — the gains and losses cancel out.

## Recommendation

Ship as a **configurable option**, not the default:
- `BENCH_LANE_WINDOWS=75 BENCH_LANE_FACTS=25` for single-hop-heavy workloads
- Default remains `45w+15f` for balanced accuracy across all categories

## Status

Shipped as tuning option in v0.9.0.
