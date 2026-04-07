# Phase 2f: Two-Lane Search Without Cross-Encoder

**Date:** 2026-04-07
**Dataset:** LoCoMo conv-0, 199 questions
**Config:** Dense search only (no CE), windows-only w5s2, FACT_EXTRACTION=1, BENCH_TWO_LANE=1

## Hypothesis

The cross-encoder was proven to add 0pp at 60-chunk single-lane context
(experiment 4). If it also adds 0pp in two-lane search, we can disable it
for faster iteration (5.6s → ~0.5s per search call) and cheaper production.

## Results

| Config | Non-Adv | Open-dom | Temporal | Single-hop | Multi-hop | Adv |
|--------|---------|----------|----------|------------|-----------|-----|
| Baseline (63.2%) | 63.2% | 75.7% | 59.5% | 43.8% | 53.8% | 6.4% |
| Two-Lane + CE | 69.7% | 82.9% | 73.0% | 43.8% | 53.8% | 6.4% |
| **Two-Lane - CE** | **64.5%** | **80.0%** | **70.3%** | **28.1%** | **53.8%** | **8.5%** |

*Delta (no CE vs with CE):*

| Metric | Delta |
|--------|-------|
| Non-Adv | **-5.2pp** |
| Open-domain | -2.9pp |
| Temporal | -2.7pp |
| Single-hop | **-15.7pp** |
| Multi-hop | 0.0pp |

## Retrieval Oracle

| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 43 | 9 (20%) | 20 (46%) | 14 (32%) |
| multi-hop | 6 | 6 (100%) | 0 (0%) | 0 (0%) |
| open-domain | 14 | 5 (35%) | 5 (35%) | 4 (28%) |
| single-hop | 23 | 10 (43%) | 6 (26%) | 7 (30%) |
| temporal | 11 | 0 (0%) | 2 (18%) | 9 (81%) |

## Key Finding: Cross-Encoder IS Essential for Two-Lane Search

**Why CE adds 0pp at single-lane 60-chunk but +5.2pp at two-lane:**

The critical variable is the **selection ratio** — how many results the CE
must select from the candidate pool:

| Config | Limit | Pool Size | Selection Ratio | CE Impact |
|--------|-------|-----------|-----------------|-----------|
| Single-lane | 60 | 208 | 28.8% | 0pp |
| Two-lane windows | 45 | 208 | 21.6% | **+15.7pp single-hop** |
| Two-lane facts | 15 | 743 | 2.0% | ~+2.7pp temporal |

At 28.8% selection (single-lane), you grab most windows and ranking doesn't
matter — the answer model finds the needle in 60 haystacks. At 21.6%
(two-lane windows), you're excluding 78.4% of windows. Without CE, the
wrong windows get excluded and single-hop answers are lost.

**Retrieval oracle confirms:** "Not in top-60" for single-hop jumped from
27% (with CE) to 43% (without CE). The CE is literally selecting the right
windows into the top-45.

## Verdict: REVERT — Cross-encoder must stay enabled

The cross-encoder earns its 5.6s/call overhead by making the 45-window
selection 15.7pp better for single-hop. This is the mechanism that makes
two-lane search work: CE quality-selects the window subset, while the
separate fact lane adds temporal precision.

Production implication: CE is required for two-lane search. Future
optimization should focus on faster CE models or GPU inference, not
disabling CE.
