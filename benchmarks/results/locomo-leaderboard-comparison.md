# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-06 01:12
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 63.16%**
- Token F1: 16.04%

## Including adversarial
- All categories: 49.75%
- Adversarial only: 6.38%

## Per-Category Breakdown
- **adversarial**: 6.4% judge, 13.6% F1 (47 Qs)
- **multi-hop**: 53.8% judge, 7.3% F1 (13 Qs)
- **open-domain**: 75.7% judge, 20.3% F1 (70 Qs)
- **single-hop**: 43.8% judge, 8.5% F1 (32 Qs)
- **temporal**: 59.5% judge, 17.5% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 63.2% (96/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 44 | 1 (2%) | 9 (20%) | 34 (77%) |
| multi-hop | 6 | 5 (83%) | 1 (16%) | 0 (0%) |
| open-domain | 17 | 4 (23%) | 6 (35%) | 7 (41%) |
| single-hop | 18 | 6 (33%) | 4 (22%) | 8 (44%) |
| temporal | 15 | 3 (20%) | 1 (6%) | 11 (73%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **63.16%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |