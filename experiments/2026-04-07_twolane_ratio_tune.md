# Phase 2e: Two-Lane Ratio Tuning — 50w+10f vs 45w+15f

**Date:** 2026-04-07
**Dataset:** LoCoMo conv-0, 199 questions
**Config:** Dense search + cross-encoder, windows-only w5s2, FACT_EXTRACTION=1, BENCH_TWO_LANE=1

## Hypothesis

Experiment 10 (45w+15f) showed 9 single-hop failures in the "in 60 not 10"
category (50% of wrong answers). More window slots (50 vs 45) could push
more answers into the retrievable range and improve single-hop.

## Configurations Tested

| Config | Window Slots | Fact Slots | Total |
|--------|-------------|------------|-------|
| Exp 10 | 45 | 15 | 60 |
| **Exp 11** | **50** | **10** | **60** |

## Results

| Config | Non-Adv | Open-dom | Temporal | Single-hop | Multi-hop | Adv |
|--------|---------|----------|----------|------------|-----------|-----|
| Baseline | 63.2% | 75.7% | 59.5% | 43.8% | 53.8% | 6.4% |
| 45w+15f | 69.7% | 82.9% | 73.0% | 43.8% | 53.8% | 6.4% |
| **50w+10f** | **69.7%** | **84.3%** | **75.7%** | **40.6%** | **46.2%** | **8.5%** |

*50w+10f vs 45w+15f delta:*

| Metric | 45w+15f | 50w+10f | Delta |
|--------|---------|---------|-------|
| Non-Adv | 69.7% | 69.7% | 0.0pp |
| Open-domain | 82.9% | 84.3% | +1.4pp |
| Temporal | 73.0% | 75.7% | +2.7pp |
| Single-hop | 43.8% | 40.6% | -3.2pp |
| Multi-hop | 53.8% | 46.2% | -7.6pp |

## Key Findings

1. **Overall score identical.** Both produce 106/152 non-adv (69.7%). The
   ratio change redistributes accuracy across categories but doesn't change
   the total.

2. **More windows did NOT help single-hop.** Counter to hypothesis, 50 window
   slots performed worse than 45 (-3.2pp). With only 32 single-hop questions,
   this is 1 answer difference (13 vs 14 correct) — likely noise.

3. **Fewer facts did NOT hurt temporal.** 10 fact slots actually scored
   higher than 15 on temporal (75.7% vs 73.0%). Again, 1 answer difference
   (28 vs 27) on 37 questions.

4. **Statistical noise dominates at this sample size.** Per-category shifts
   of 1-2 answers on 13-37 question pools are not significant. The identical
   overall score confirms the ratio doesn't meaningfully affect total quality.

## Verdict: KEEP 45:15 as default — no improvement from 50:10

The 45:15 ratio from exp 10 remains the best because:
- Identical overall performance
- Single-hop exactly at baseline (43.8%) vs borderline regression at 50:10 (40.6%)
- Multi-hop at baseline (53.8%) vs below at 50:10 (46.2%)
- The 3pp regression threshold is touched by 50:10 single-hop

The two-lane ratio is not a sensitive lever. Both 45:15 and 50:10 produce
the same total; the ratio mainly shuffles per-category accuracy.
Future optimization should focus on improving retrieval recall and
generation quality, not ratio tuning.
