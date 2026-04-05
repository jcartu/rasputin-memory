# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-05 04:42
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 56.58%**
- Token F1: 13.97%

## Including adversarial
- All categories: 44.72%
- Adversarial only: 6.38%

## Per-Category Breakdown
- **adversarial**: 6.4% judge, 11.2% F1 (47 Qs)
- **multi-hop**: 23.1% judge, 7.5% F1 (13 Qs)
- **open-domain**: 70.0% judge, 17.6% F1 (70 Qs)
- **single-hop**: 37.5% judge, 6.8% F1 (32 Qs)
- **temporal**: 59.5% judge, 15.6% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 56.6% (86/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 44 | 2 (4%) | 8 (18%) | 34 (77%) |
| multi-hop | 10 | 5 (50%) | 4 (40%) | 1 (10%) |
| open-domain | 21 | 4 (19%) | 7 (33%) | 10 (47%) |
| single-hop | 20 | 4 (20%) | 6 (30%) | 10 (50%) |
| temporal | 15 | 3 (20%) | 1 (6%) | 11 (73%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **56.58%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |