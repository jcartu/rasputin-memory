# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-05 03:46
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 63.16%**
- Token F1: 15.74%

## Including adversarial
- All categories: 49.75%
- Adversarial only: 6.38%

## Per-Category Breakdown
- **adversarial**: 6.4% judge, 11.9% F1 (47 Qs)
- **multi-hop**: 53.8% judge, 7.8% F1 (13 Qs)
- **open-domain**: 74.3% judge, 19.6% F1 (70 Qs)
- **single-hop**: 40.6% judge, 8.9% F1 (32 Qs)
- **temporal**: 64.9% judge, 17.1% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 63.2% (96/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 44 | 2 (4%) | 8 (18%) | 34 (77%) |
| multi-hop | 6 | 5 (83%) | 1 (16%) | 0 (0%) |
| open-domain | 18 | 3 (16%) | 8 (44%) | 7 (38%) |
| single-hop | 19 | 6 (31%) | 3 (15%) | 10 (52%) |
| temporal | 13 | 3 (23%) | 1 (7%) | 9 (69%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **63.16%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |