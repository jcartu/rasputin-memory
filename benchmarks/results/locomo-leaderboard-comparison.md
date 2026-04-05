# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-05 05:26
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → gpt-4o-mini → gpt-4o-mini-2024-07-18 judge
**Mode:** compare | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 72.37%**
- Token F1: 28.64%

## Including adversarial
- All categories: 61.31%
- Adversarial only: 25.53%

## Per-Category Breakdown
- **adversarial**: 25.5% judge, 9.2% F1 (47 Qs)
- **multi-hop**: 38.5% judge, 5.9% F1 (13 Qs)
- **open-domain**: 80.0% judge, 39.6% F1 (70 Qs)
- **single-hop**: 68.8% judge, 16.6% F1 (32 Qs)
- **temporal**: 73.0% judge, 26.3% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 72.4% (110/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 35 | 2 (5%) | 7 (20%) | 26 (74%) |
| multi-hop | 8 | 4 (50%) | 3 (37%) | 1 (12%) |
| open-domain | 14 | 3 (21%) | 6 (42%) | 5 (35%) |
| single-hop | 10 | 2 (20%) | 3 (30%) | 5 (50%) |
| temporal | 10 | 2 (20%) | 0 (0%) | 8 (80%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **72.37%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |