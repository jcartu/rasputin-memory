# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-05 23:39
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 59.87%**
- Token F1: 15.57%

## Including adversarial
- All categories: 47.74%
- Adversarial only: 8.51%

## Per-Category Breakdown
- **adversarial**: 8.5% judge, 12.1% F1 (47 Qs)
- **multi-hop**: 46.2% judge, 7.4% F1 (13 Qs)
- **open-domain**: 71.4% judge, 19.8% F1 (70 Qs)
- **single-hop**: 40.6% judge, 9.4% F1 (32 Qs)
- **temporal**: 59.5% judge, 15.9% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 59.9% (91/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 43 | 10 (23%) | 9 (20%) | 24 (55%) |
| multi-hop | 7 | 7 (100%) | 0 (0%) | 0 (0%) |
| open-domain | 20 | 11 (55%) | 2 (10%) | 7 (35%) |
| single-hop | 19 | 4 (21%) | 4 (21%) | 11 (57%) |
| temporal | 15 | 3 (20%) | 1 (6%) | 11 (73%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **59.87%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |