# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-05 14:13
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 61.84%**
- Token F1: 16.47%

## Including adversarial
- All categories: 49.75%
- Adversarial only: 10.64%

## Per-Category Breakdown
- **adversarial**: 10.6% judge, 11.0% F1 (47 Qs)
- **multi-hop**: 38.5% judge, 8.7% F1 (13 Qs)
- **open-domain**: 75.7% judge, 20.8% F1 (70 Qs)
- **single-hop**: 31.2% judge, 8.2% F1 (32 Qs)
- **temporal**: 70.3% judge, 18.1% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 61.8% (94/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 42 | 2 (4%) | 8 (19%) | 32 (76%) |
| multi-hop | 8 | 7 (87%) | 1 (12%) | 0 (0%) |
| open-domain | 17 | 6 (35%) | 7 (41%) | 4 (23%) |
| single-hop | 22 | 6 (27%) | 9 (40%) | 7 (31%) |
| temporal | 11 | 2 (18%) | 2 (18%) | 7 (63%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **61.84%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |