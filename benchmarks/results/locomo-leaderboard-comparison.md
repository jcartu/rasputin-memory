# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-04 19:59
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → gpt-4o-mini → gpt-4o-mini-2024-07-18 judge
**Mode:** compare | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 61.18%**
- Token F1: 0.00%

## Including adversarial
- All categories: 62.81%
- Adversarial only: 68.09%

## Per-Category Breakdown
- **adversarial**: 68.1% judge, 0.0% F1 (47 Qs)
- **multi-hop**: 92.3% judge, 0.0% F1 (13 Qs)
- **open-domain**: 72.9% judge, 0.0% F1 (70 Qs)
- **single-hop**: 40.6% judge, 0.0% F1 (32 Qs)
- **temporal**: 45.9% judge, 0.0% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 61.2% (93/152 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **61.18%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |