# Phase 2: Consolidation Observations (Three-Lane Search)

**Date:** 2026-04-09
**Baseline:** Phase 1 prompt routing (73.0% conv-0 non-adv)
**Dataset:** LoCoMo conv-0, 199 questions

## What Changed

Added consolidation engine: LLM (Groq llama-3.3-70b) reads all extracted facts
per conversation and synthesizes them into higher-level "observations" — stored
in `{collection}_obs` Qdrant collections.

Three-lane search: 35 windows + 10 facts + 15 observations = 60 candidates,
then cross-encoder reranking.

6,363 observations created across 10 conversations (avg 636/conv).

## Conv-0 Results (199 questions)

| Category | Phase 1 | Phase 2 | Change |
|----------|---------|---------|--------|
| non-adv  | 73.0%   | 72.4%   | -0.6pp |
| multi-hop | 76.9%  | 76.9%   | same   |
| temporal | 83.8%   | **89.2%** | **+5.4pp** |
| open-domain | 82.9% | 84.3% | +1.4pp |
| single-hop | 37.5% | **25.0%** | **-12.5pp** |

## Analysis

Temporal gains (+5.4pp) make sense — consolidated observations aggregate scattered
date mentions into coherent timelines.

Single-hop regression (-12.5pp) is the problem. Observations are broad summaries
that dilute the context for questions needing specific factual details. When a
single-hop question asks "What color was X?", the observation "X has many hobbies
including painting and pottery" doesn't help and pushes the specific answer out
of the top context.

## Next Steps

1. Filter observations by question type — only use obs lane for inference/temporal
2. Reduce obs lane slots from 15 to 5 for factual questions
3. Or: gate obs lane behind classify_question() — only for inference + temporal
