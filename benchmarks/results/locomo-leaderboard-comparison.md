# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-06 10:45
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 63.16%**
- Token F1: 17.06%

## Including adversarial
- All categories: 49.75%
- Adversarial only: 6.38%

## Per-Category Breakdown
- **adversarial**: 6.4% judge, 12.0% F1 (47 Qs)
- **multi-hop**: 30.8% judge, 6.7% F1 (13 Qs)
- **open-domain**: 78.6% judge, 21.8% F1 (70 Qs)
- **single-hop**: 25.0% judge, 6.9% F1 (32 Qs)
- **temporal**: 78.4% judge, 20.5% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 63.2% (96/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 44 | 2 (4%) | 7 (15%) | 35 (79%) |
| multi-hop | 9 | 9 (100%) | 0 (0%) | 0 (0%) |
| open-domain | 15 | 4 (26%) | 5 (33%) | 6 (40%) |
| single-hop | 24 | 7 (29%) | 6 (25%) | 11 (45%) |
| temporal | 8 | 2 (25%) | 0 (0%) | 6 (75%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **63.16%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |