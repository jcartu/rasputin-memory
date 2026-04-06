# Phase 2c: Two-Lane Retrieval — Facts with Cap

**Date:** 2026-04-06
**Dataset:** LoCoMo conv-0, 199 questions
**Config:** Dense search + cross-encoder, windows-only w5s2, FACT_EXTRACTION=1, FACT_CAP=15

## Hypothesis

Fact extraction helps temporal (+18.9pp uncapped) but destroys single-hop
(-21.9pp uncapped) because 744 facts dilute the 208 windows in search results.
Capping facts at 15 per query should preserve enough windows for context while
adding fact precision for temporal/open-domain.

## Implementation

Added `apply_fact_cap(results, max_facts)` to the benchmark. After
`multi_query_search` returns mixed results from the combined collection
(745 facts + 208 windows = 953 vectors), the cap preserves sort order but
allows at most 15 facts through. Remaining slots are windows.

Also added `chunk_type` to `qdrant_search` output (production change) for
reliable fact identification.

## Configurations Tested

| Config | Facts | Fact Cap | Context | Vectors |
|--------|-------|----------|---------|---------|
| Baseline | None | - | 60 | 208 |
| Uncapped facts | 744 | None | 60 | 952 |
| **Cap-15** | **745** | **15** | **60** | **953** |

## Environment

```
FACT_EXTRACTION=1  BENCH_FACT_CAP=15
BENCH_CONTEXT_CHUNKS=60  BENCH_SEARCH_LIMIT=60  CROSS_ENCODER=1
CHUNK_TURNS=0  CHUNK_WINDOWS=1  CHUNK_WINDOW_SIZE=5  CHUNK_STRIDE=2
ABLATION_BM25=0  ABLATION_KEYWORD_BOOST=0  ABLATION_ENTITY_BOOST=0
ABLATION_TEMPORAL_BOOST=0  ABLATION_TEMPORAL_DECAY=0  ABLATION_MMR=0
ABLATION_RERANKER=0  ABLATION_QUERY_EXPAND=0
```

## Results

| Config | Non-Adv | Open-dom | Temporal | Single-hop | Multi-hop | Adv |
|--------|---------|----------|----------|------------|-----------|-----|
| Baseline (63.2%) | 63.2% | 75.7% | 59.5% | 43.8% | 53.8% | 6.4% |
| Win w5s2 (68.4%) | 68.4% | 81.4% | 70.3% | 46.9% | 46.2% | 6.4% |
| Uncapped facts | 63.2% | 78.6% | 78.4% | 25.0% | 30.8% | 6.4% |
| **Cap-15** | **66.4%** | **85.7%** | **70.3%** | **28.1%** | **46.2%** | **6.4%** |

*Delta vs baseline (63.2%):*

| Metric | Delta |
|--------|-------|
| Non-Adv | **+3.2pp** |
| Open-domain | **+10.0pp** |
| Temporal | **+10.8pp** |
| Single-hop | **-15.7pp** |
| Multi-hop | -7.6pp |

*Cap-15 vs uncapped facts:*

| Metric | Uncapped | Cap-15 | Improvement |
|--------|----------|--------|-------------|
| Single-hop | 25.0% | 28.1% | +3.1pp |
| Multi-hop | 30.8% | 46.2% | +15.4pp |
| Temporal | 78.4% | 70.3% | -8.1pp |

## Retrieval Oracle

| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 44 | 0 (0%) | 5 (11%) | 39 (88%) |
| multi-hop | 7 | 6 (85%) | 1 (14%) | 0 (0%) |
| open-domain | 10 | 2 (20%) | 2 (20%) | 6 (60%) |
| single-hop | 23 | 9 (39%) | 3 (13%) | 11 (47%) |
| temporal | 11 | 1 (9%) | 1 (9%) | 9 (81%) |

## Key Findings

1. **Cap helps but doesn't solve the root cause.** Single-hop improved from
   25.0% (uncapped) to 28.1% (cap-15), and multi-hop from 30.8% to 46.2%.
   But single-hop is still -15.7pp from baseline, violating the 3pp threshold.

2. **Open-domain loves facts.** +10.0pp over baseline. General questions benefit
   from having multiple angles (facts + windows) in the context.

3. **The cap trades temporal for single-hop.** Reducing facts from uncapped to 15
   dropped temporal from 78.4% to 70.3% (-8.1pp) while only improving single-hop
   by +3.1pp. Not a favorable trade at this ratio.

4. **Root cause: facts displace windows at the SEARCH level, not just the
   context level.** The cap only controls what reaches the answer model. But
   the server searches a combined collection (953 vectors) where facts
   outnumber windows 3.6:1. The server's top-60 is fact-dominated. Even
   after capping, only ~30 windows make it through multi-query merge
   (vs 60 at baseline). Single-hop answers that were in window positions
   31-60 are permanently lost.

5. **"Not in top-60" for single-hop: 39% vs ~29% baseline.** More retrieval
   misses because facts displaced windows from the server's top-60 results.

## Verdict: REVERT — Single-hop regression too large

The fact cap partially addresses the dilution problem (+3.2pp overall, multi-hop
recovered from 30.8% to 46.2%) but single-hop at -15.7pp is unacceptable.

## Next Step: Two-Lane SEARCH

The fix must operate at the search level, not just the context level. Add a
`chunk_type` filter to Qdrant search so the benchmark can do:
- Windows search: limit=45 (guaranteed window coverage)
- Facts search: limit=15 (bonus precision)
- Merge: 45 windows + 15 facts = 60 total

This ensures single-hop windows are never displaced by facts in the search
results. The ratio 45:15 can be tuned in follow-up experiments.
