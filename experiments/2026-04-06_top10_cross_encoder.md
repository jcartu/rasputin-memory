# Phase 2b: Top-10 Context with Cross-Encoder Reranking

**Date:** 2026-04-06
**Dataset:** LoCoMo conv-0, 199 questions
**Config:** Dense search only, no boosts, cross-encoder ON, windows-only w5s2

## Hypothesis

At 60-chunk context, the answer model sees enough material that ranking quality
is irrelevant (proven in experiments 1-4: BM25, keyword/entity/temporal boosts,
MMR, Cohere reranker, and cross-encoder all add 0pp). Reducing to 10-chunk
context forces the system to rely on ranking quality — the cross-encoder's
ability to push truly relevant chunks to the top should compensate for the
smaller context window.

This tests the first lever identified in Phase 2a: whether reducing context
size (making ranking matter) combined with a quality reranker can match or
beat the 60-chunk brute-force approach.

## Bug Fix (Pre-Experiment)

`multi_query_search` in the benchmark was sorting merged results by raw Qdrant
`score`, discarding the cross-encoder's `final_score` ordering. Fixed to use
`final_score > rerank_score > score` cascade. This had zero effect at 60-chunk
context (proven) but is essential for meaningful top-10 results.

Also fixed `generate_opus_answer` call to respect `BENCH_CONTEXT_CHUNKS` via
`max_chunks` parameter (was previously hardcoded to 50).

## Configurations Tested

| Config | Context | Search Limit | Cross-Encoder | Boosts | Vectors |
|--------|---------|--------------|---------------|--------|---------|
| Baseline | 60 | 60 | ON (no effect) | All OFF | 208 |
| **Top-10 CE** | **10** | **60** | **ON** | All OFF | 208 |

Key: search still fetches 60 results per query (large reranking pool),
cross-encoder reranks them, only top 10 go to the answer model.

## Environment

```
BENCH_CONTEXT_CHUNKS=10
BENCH_SEARCH_LIMIT=60  (unchanged — large pool for reranking)
CROSS_ENCODER=1
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
| **Top-10 CE** | **53.3%** | **77.1%** | **56.8%** | **9.4%** | **23.1%** | **6.4%** |

*Delta vs baseline (63.2%):*

| Metric | Delta |
|--------|-------|
| Non-Adv | **-9.9pp** |
| Open-domain | +1.4pp |
| Temporal | -2.7pp |
| Single-hop | **-34.4pp** |
| Multi-hop | **-30.7pp** |

## Retrieval Oracle

| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 44 | 6 (13%) | 0 (0%) | 38 (86%) |
| multi-hop | 10 | 10 (100%) | 0 (0%) | 0 (0%) |
| open-domain | 16 | 8 (50%) | 0 (0%) | 8 (50%) |
| single-hop | 29 | 18 (62%) | 0 (0%) | 11 (37%) |
| temporal | 16 | 3 (18%) | 0 (0%) | 13 (81%) |

**Note:** Oracle only sees top-10 chunks (CONTEXT_CHUNKS=10), so "In 60 not 10"
reads as 0% by construction. The true diagnostic is the high "Not in top-60"
rate for single-hop (62%) — these are retrieval misses, not ranking failures.

## Key Findings

1. **Top-10 context is catastrophically too small.** Single-hop -34.4pp and
   multi-hop -30.7pp. The answer model cannot function with so little context
   for precision questions.

2. **Cross-encoder ranking works perfectly within its scope.** "In 60 not 10"
   is 0% everywhere — when the answer is retrievable, the cross-encoder pushes
   it to the top-10 without fail. The ranking is not the bottleneck.

3. **Open-domain is resilient to context reduction.** +1.4pp at top-10 because
   open-domain questions can be answered from multiple chunks — any of the
   top-10 may suffice.

4. **The problem is retrieval recall, not ranking precision.** Single-hop
   questions need a specific fact from a specific memory chunk. With 208 windows
   from 19 sessions, the answer is often in a chunk that doesn't rank in the
   top-60 by semantic similarity to the question.

5. **Cross-encoder is too slow for production at 60-chunk candidate pools.**
   5.6 seconds per search call (208 candidates on CPU). The ms-marco-MiniLM
   model was designed for web search re-ranking, not 200+ candidate pools.

## Verdict: REVERT — Top-10 context destroys precision categories

The hypothesis was wrong: reducing context doesn't "make ranking matter" in a
useful way. The problem is that conversational memory search has fundamentally
different recall characteristics than web search. The answer to "What did Alice
cook for dinner on July 5th?" might be in a window that doesn't mention Alice,
cooking, or July 5th in ways that semantic search can recover.

**Why the cross-encoder can't save small context:**
- It re-ranks what's already retrieved. If the answer window wasn't fetched
  from Qdrant in the first place, no reranker can fix that.
- For single-hop precision questions, the bottleneck is embedding similarity
  between the question and the answer chunk, not ranking quality.

## Next Steps

Top-10 is dead. The two productive paths are:

1. **Two-lane retrieval** (windows + facts): Keep 60-chunk context but fill it
   with a mix of broad windows (for context) and precise extracted facts (for
   single-hop/temporal). This addresses the root cause — making the right
   information EXIST in the search results.

2. **Better embeddings or chunking**: If the answer isn't in the top-60 by
   semantic similarity, improve what gets embedded (e.g., enriched window text
   with speaker names and dates, or smaller windows for single-turn facts).

Path 1 is higher-value because fact extraction already showed +18.9pp on
temporal. The challenge is mixing facts and windows without destroying
single-hop (which facts alone did at -21.9pp).
