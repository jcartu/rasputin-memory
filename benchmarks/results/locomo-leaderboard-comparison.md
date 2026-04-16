# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-16 06:49
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 239 (191 non-adversarial, 48 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 69.11%**
- Token F1: 17.32%

## Including adversarial
- All categories: 55.65%
- Adversarial only: 2.08%

## Per-Category Breakdown
- **adversarial**: 2.1% judge, 12.5% F1 (48 Qs)
- **multi-hop**: 50.0% judge, 5.1% F1 (10 Qs)
- **open-domain**: 78.0% judge, 19.7% F1 (118 Qs)
- **single-hop**: 47.6% judge, 14.8% F1 (21 Qs)
- **temporal**: 59.5% judge, 14.7% F1 (42 Qs)

## Per-Conversation
- **conv-48**: 69.1% (132/191 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 47 | 2 (4%) | 1 (2%) | 44 (93%) |
| multi-hop | 5 | 3 (60%) | 0 (0%) | 2 (40%) |
| open-domain | 26 | 3 (11%) | 4 (15%) | 19 (73%) |
| single-hop | 11 | 4 (36%) | 2 (18%) | 5 (45%) |
| temporal | 17 | 2 (11%) | 1 (5%) | 14 (82%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **69.11%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |