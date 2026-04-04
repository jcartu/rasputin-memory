# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-04 17:32
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 62.50%**
- Token F1: 15.70%

## Including adversarial
- All categories: 49.25%
- Adversarial only: 6.38%

## Per-Category Breakdown
- **adversarial**: 6.4% judge, 11.3% F1 (47 Qs)
- **multi-hop**: 46.2% judge, 7.7% F1 (13 Qs)
- **open-domain**: 74.3% judge, 20.0% F1 (70 Qs)
- **single-hop**: 43.8% judge, 8.7% F1 (32 Qs)
- **temporal**: 62.2% judge, 16.4% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 62.5% (95/152 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **62.50%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |