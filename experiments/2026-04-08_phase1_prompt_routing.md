# Phase 1: Inference Prompt Routing

**Date:** 2026-04-08
**Commit:** 59a4787 (code), 2878d1a (conv-0 results)
**Dataset:** LoCoMo 10-conv, 1986 questions
**Baseline:** 67.5% non-adv (commit dbb6100)

## Hypothesis

71% of multi-hop failures are the model saying "I don't have enough information"
even with evidence in context. A single generic prompt can't serve all question
types — inference questions need encouragement to reason from evidence, temporal
questions need date-computation guidance, and factual questions need the existing
entity-swap tolerance.

## What Changed

Added `classify_question()` — regex-based classifier routing questions to three
prompt templates based on question type:

- **factual** (default): Existing adversarial-resistant prompt. Entity-swap tolerant.
- **inference**: "Draw reasonable conclusions from evidence. Only abstain when memories
  contain genuinely NOTHING related." Explicit examples of valid inference chains.
- **temporal**: "Use timestamps to compute actual dates. Enumerate distinct mentions
  when counting."

Controlled by `PROMPT_ROUTING=1` env var. No LLM calls — pure regex, zero latency.

## Conv-0 Results (199 questions)

| Category | Baseline | Phase 1 | Change |
|----------|----------|---------|--------|
| non-adv overall | 67.5% | **73.0%** | **+5.5pp** |
| multi-hop | 38.5% | **76.9%** | **+38.4pp** |
| temporal | 64.8% | **83.8%** | **+19.0pp** |
| single-hop | 37.2% | 37.5% | +0.3pp |
| open-domain | 81.9% | 82.9% | +1.0pp |
| adversarial | 2.5% | 14.9% | +12.4pp |

## 3-Conv Running Total (497 Qs)

| Category | Baseline | Phase 1 | Change |
|----------|----------|---------|--------|
| non-adv overall | 67.5% | **72.2%** | **+4.7pp** |
| multi-hop | 38.5% | **61.9%** | **+23.4pp** |
| temporal | 64.8% | **82.5%** | **+17.7pp** |
| single-hop | 37.2% | **44.6%** | **+7.4pp** |
| open-domain | 81.9% | 76.3% | -5.6pp |

## Analysis

Multi-hop gains are massive — the inference prompt stops the model from abstaining
when evidence is present but requires connecting dots. The explicit instruction
"Only say I don't have enough information if memories contain genuinely NOTHING"
directly addresses the 71% abstention rate.

Temporal gains come from the date-computation instructions — "yesterday relative to
timestamp 2023-05-08 = 2023-05-07" directly helps the model resolve relative dates.

Open-domain slight regression in 3-conv total may be noise or some factual questions
being misclassified as inference. Will monitor in full 10-conv.

## Status

Full 10-conv validation in progress. Will update with final numbers.
Success gate: non-adv ≥ 69%, multi-hop ≥ 50%.
Both gates already passed on conv-0.
