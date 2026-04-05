# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-05 11:46
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 65.13%**
- Token F1: 15.58%

## Including adversarial
- All categories: 51.26%
- Adversarial only: 6.38%

## Per-Category Breakdown
- **adversarial**: 6.4% judge, 13.0% F1 (47 Qs)
- **multi-hop**: 61.5% judge, 7.9% F1 (13 Qs)
- **open-domain**: 77.1% judge, 19.7% F1 (70 Qs)
- **single-hop**: 46.9% judge, 8.2% F1 (32 Qs)
- **temporal**: 59.5% judge, 16.8% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 65.1% (99/152 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 44 | 2 (4%) | 8 (18%) | 34 (77%) |
| multi-hop | 5 | 4 (80%) | 1 (20%) | 0 (0%) |
| open-domain | 16 | 3 (18%) | 7 (43%) | 6 (37%) |
| single-hop | 17 | 6 (35%) | 3 (17%) | 8 (47%) |
| temporal | 15 | 3 (20%) | 1 (6%) | 11 (73%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **65.13%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |