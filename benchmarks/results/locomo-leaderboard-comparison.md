# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-11 03:20
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 63.16%**
- Token F1: 15.27%

## Including adversarial
- All categories: 51.76%
- Adversarial only: 14.89%

## Per-Category Breakdown
- **adversarial**: 14.9% judge, 12.4% F1 (47 Qs)
- **multi-hop**: 76.9% judge, 5.0% F1 (13 Qs)
- **open-domain**: 70.0% judge, 20.0% F1 (70 Qs)
- **single-hop**: 34.4% judge, 8.0% F1 (32 Qs)
- **temporal**: 70.3% judge, 16.2% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 63.2% (96/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 40 | 6 (15%) | 2 (5%) | 32 (80%) |
| multi-hop | 3 | 2 (66%) | 1 (33%) | 0 (0%) |
| open-domain | 21 | 8 (38%) | 4 (19%) | 9 (42%) |
| single-hop | 21 | 7 (33%) | 7 (33%) | 7 (33%) |
| temporal | 11 | 1 (9%) | 2 (18%) | 8 (72%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **63.16%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |