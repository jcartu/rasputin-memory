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

## Full 10-Conv Final Results (1986 Qs)

| Category | Baseline | Phase 1 | Change |
|----------|----------|---------|--------|
| non-adv overall | 67.5% | **69.1%** | **+1.6pp** |
| multi-hop | 38.5% | **55.2%** | **+16.7pp** |
| single-hop | 37.2% | **41.1%** | **+3.9pp** |
| temporal | 64.8% | **66.4%** | **+1.6pp** |
| open-domain | 81.9% | 81.1% | -0.8pp |
| adversarial | 2.5% | 11.7% | +9.2pp |

Per-conversation:
- conv-26: 73.0%, conv-30: 70.4%, conv-41: 72.4%, conv-42: 67.8%
- conv-43: 64.0%, conv-44: 64.2%, conv-47: 76.0%, conv-48: 73.8%
- conv-49: 64.1%, conv-50: 65.2%

## Analysis

Multi-hop gains confirmed at scale (+16.7pp). The inference prompt stops the model
from abstaining when evidence requires connecting dots.

Temporal gains smaller at scale (+1.6pp vs +19pp on conv-0) — conv-0 was likely an
outlier. The date-computation instructions help specific questions but don't
generalize as broadly.

Open-domain flat (-0.8pp) — prompt routing doesn't harm factual questions.
Single-hop improved +3.9pp — some single-hop questions benefit from inference prompt.

Note: conv-43 was rescored after OpenAI credit outage during original run.

## Result

**SHIPPED.** Success gates passed: non-adv ≥ 69% ✅, multi-hop ≥ 50% ✅.
