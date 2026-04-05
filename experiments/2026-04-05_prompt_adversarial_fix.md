# Prompt Fix: Remove Entity-Name Refusal + Timestamp Hint

**Date:** 2026-04-05
**Commit:** 5e617e3 (reverted to b9d9f77)
**Dataset:** LoCoMo conv-0, 199 questions

## Hypothesis

Removing the "refuse when entity names don't match" instruction will fix
adversarial over-abstention (77% of adversarial failures are answer failures
with gold in chunks). Adding timestamp resolution hint will fix temporal
generation failures (73% of temporal failures are answer failures).

## What Changed

- Removed: "If the question names a person or entity that doesn't appear in the
  memories, say so rather than substituting a similar entity."
- Added: "When memories contain timestamps like [2:24 pm on 14 August, 2023],
  use them together with relative phrases to compute actual dates."

## Results

| Category | Baseline | Prompt Fix | Delta |
|----------|----------|------------|-------|
| Non-adversarial | 65.1% | 61.8% | -3.3pp |
| Adversarial | 6.4% | 10.6% | +4.2pp |
| Temporal | 59.5% | 70.3% | +10.8pp |
| Single-hop | 46.9% | 31.2% | -15.7pp |
| Multi-hop | 61.5% | 38.5% | -23.0pp |
| Open-domain | 77.1% | 75.7% | -1.4pp |

## Verdict: REVERT — net regression

The targeted categories improved (adversarial +4.2pp, temporal +10.8pp) but
removing entity-name caution caused single-hop to collapse -15.7pp. The model
became reckless — answering with wrong entities instead of being cautious.

Multi-hop also dropped -23.0pp, likely because the timestamp hint constrained
the model's reasoning about cross-reference questions.

## Lesson

Prompt changes trade accuracy between categories. A fix for one failure mode
creates failures in others. The adversarial and temporal problems may need
per-category prompt routing, not a single prompt change.
