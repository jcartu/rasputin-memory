# Phase 4a: Three-Lane Search — BM25 Keyword Lane via Qdrant Text Index

**Date:** 2026-04-07
**Dataset:** LoCoMo conv-0, 199 questions
**Config:** Dense search + cross-encoder, windows-only w5s2, FACT_EXTRACTION=1, BENCH_TWO_LANE=1, BENCH_BM25_LANE=1

## Hypothesis

The retrieval oracle from experiment 10 shows 27% of single-hop failures have
the gold answer "not in top-60" — dense search completely misses them. Adding
a keyword-based retrieval lane (BM25) could recover these misses by finding
chunks that contain matching keywords but have different semantic embeddings.

## Implementation

Added Qdrant full-text payload index on the `text` field with word tokenizer.
After two-lane dense search, a supplementary `MatchText` query finds windows
containing all extracted keywords from the question (stop-word filtered). These
keyword-matched windows fill the BM25 lane slots.

To keep total context at 60 chunks, dense windows were reduced from 45 to 35:
- 35 dense windows + 15 dense facts + 10 BM25 keyword windows = 60 total

## Configurations Tested

| Config | Dense Windows | Dense Facts | BM25 Windows | Total |
|--------|--------------|-------------|--------------|-------|
| Two-Lane (best) | 45 | 15 | 0 | 60 |
| **Three-Lane** | **35** | **15** | **10** | **60** |

## Results

| Config | Non-Adv | Open-dom | Temporal | Single-hop | Multi-hop | Adv |
|--------|---------|----------|----------|------------|-----------|-----|
| Baseline (63.2%) | 63.2% | 75.7% | 59.5% | 43.8% | 53.8% | 6.4% |
| Two-Lane 45+15 | 69.7% | 82.9% | 73.0% | 43.8% | 53.8% | 6.4% |
| **Three-Lane** | **65.8%** | **77.1%** | **75.7%** | **34.4%** | **53.8%** | **10.6%** |

*Delta vs Two-Lane:*

| Metric | Delta |
|--------|-------|
| Non-Adv | **-3.9pp** |
| Open-domain | -5.8pp |
| Temporal | +2.7pp |
| Single-hop | **-9.4pp** |
| Multi-hop | 0.0pp |

## Retrieval Oracle

| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 42 | 2 (4%) | 1 (2%) | 39 (92%) |
| multi-hop | 6 | 5 (83%) | 1 (16%) | 0 (0%) |
| open-domain | 16 | 2 (12%) | 1 (6%) | 13 (81%) |
| single-hop | 21 | 8 (38%) | 2 (9%) | 11 (52%) |
| temporal | 9 | 0 (0%) | 1 (11%) | 8 (88%) |

## Key Findings

1. **Reducing dense windows from 45→35 is catastrophic for single-hop.**
   Single-hop dropped from 43.8% to 34.4% (-9.4pp). This confirms that dense
   window coverage is the single most important factor for precision questions.

2. **MatchText (AND logic) is too restrictive for keyword search.** Requiring
   ALL stop-word-filtered keywords to appear in a single window means very
   few chunks match. Most BM25 slots go unfilled, resulting in fewer than
   60 chunks in context.

3. **Open-domain hurt most (-5.8pp).** Open-domain questions are general and
   benefit from broad window coverage. Reducing from 45 to 35 dense windows
   directly reduces recall for these questions.

4. **Temporal unaffected (+2.7pp).** Temporal questions rely on facts, which
   were kept at 15 slots. The 1-answer improvement (28 vs 27) is noise.

## Verdict: REVERT — BM25 keyword lane hurts when it displaces dense windows

The MatchText approach has two fundamental problems:
1. AND logic is too restrictive — most keyword queries match zero windows
2. Displacing dense window slots to make room for BM25 directly hurts retrieval

## What Would Work Instead

If BM25 is to help, it must NOT displace dense window slots:
- **Additive approach**: Keep 45+15 two-lane, ADD BM25 results as bonus (total > 60)
- **Sparse vectors with RRF fusion**: Qdrant's prefetch API can combine dense + sparse
  search in one query with Reciprocal Rank Fusion, without slot allocation tradeoffs
- **MatchTextAny (OR logic)**: Less restrictive than MatchText, but returns too many
  low-quality matches without ranking

The two-lane 45w+15f configuration remains the best result at 69.7%.
