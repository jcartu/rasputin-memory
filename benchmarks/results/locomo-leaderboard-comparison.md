# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-06 04:39
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 68.42%**
- Token F1: 16.50%

## Including adversarial
- All categories: 54.27%
- Adversarial only: 8.51%

## Per-Category Breakdown
- **adversarial**: 8.5% judge, 14.2% F1 (47 Qs)
- **multi-hop**: 46.2% judge, 7.0% F1 (13 Qs)
- **open-domain**: 87.1% judge, 22.6% F1 (70 Qs)
- **single-hop**: 34.4% judge, 8.2% F1 (32 Qs)
- **temporal**: 70.3% judge, 15.5% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 68.4% (104/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 43 | 0 (0%) | 3 (6%) | 40 (93%) |
| multi-hop | 7 | 3 (42%) | 2 (28%) | 2 (28%) |
| open-domain | 9 | 0 (0%) | 2 (22%) | 7 (77%) |
| single-hop | 21 | 2 (9%) | 1 (4%) | 18 (85%) |
| temporal | 11 | 2 (18%) | 0 (0%) | 9 (81%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **68.42%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |