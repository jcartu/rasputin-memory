# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-04 20:52
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 61.84%**
- Token F1: 16.35%

## Including adversarial
- All categories: 48.74%
- Adversarial only: 6.38%

## Per-Category Breakdown
- **adversarial**: 6.4% judge, 12.6% F1 (47 Qs)
- **multi-hop**: 61.5% judge, 9.3% F1 (13 Qs)
- **open-domain**: 75.7% judge, 20.7% F1 (70 Qs)
- **single-hop**: 37.5% judge, 8.6% F1 (32 Qs)
- **temporal**: 56.8% judge, 17.3% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 61.8% (94/152 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **61.84%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |