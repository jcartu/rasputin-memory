# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-09 18:39
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 69.08%**
- Token F1: 15.25%

## Including adversarial
- All categories: 53.77%
- Adversarial only: 4.26%

## Per-Category Breakdown
- **adversarial**: 4.3% judge, 7.7% F1 (47 Qs)
- **multi-hop**: 69.2% judge, 5.6% F1 (13 Qs)
- **open-domain**: 81.4% judge, 20.3% F1 (70 Qs)
- **single-hop**: 34.4% judge, 8.0% F1 (32 Qs)
- **temporal**: 75.7% judge, 15.3% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 69.1% (105/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 45 | 35 (77%) | 0 (0%) | 10 (22%) |
| multi-hop | 4 | 3 (75%) | 0 (0%) | 1 (25%) |
| open-domain | 13 | 1 (7%) | 0 (0%) | 12 (92%) |
| single-hop | 21 | 6 (28%) | 5 (23%) | 10 (47%) |
| temporal | 9 | 1 (11%) | 0 (0%) | 8 (88%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **69.08%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |