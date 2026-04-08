# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-08 06:01
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 73.03%**
- Token F1: 16.26%

## Including adversarial
- All categories: 59.30%
- Adversarial only: 14.89%

## Per-Category Breakdown
- **adversarial**: 14.9% judge, 12.8% F1 (47 Qs)
- **multi-hop**: 76.9% judge, 5.6% F1 (13 Qs)
- **open-domain**: 82.9% judge, 21.5% F1 (70 Qs)
- **single-hop**: 37.5% judge, 9.2% F1 (32 Qs)
- **temporal**: 83.8% judge, 16.1% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 73.0% (111/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 40 | 0 (0%) | 4 (10%) | 36 (90%) |
| multi-hop | 3 | 3 (100%) | 0 (0%) | 0 (0%) |
| open-domain | 12 | 1 (8%) | 3 (25%) | 8 (66%) |
| single-hop | 20 | 7 (35%) | 4 (20%) | 9 (45%) |
| temporal | 6 | 0 (0%) | 1 (16%) | 5 (83%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **73.03%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |