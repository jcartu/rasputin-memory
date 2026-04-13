# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-13 14:58
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 987 (778 non-adversarial, 209 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 73.52%**
- Token F1: 15.31%

## Including adversarial
- All categories: 60.69%
- Adversarial only: 12.92%

## Per-Category Breakdown
- **adversarial**: 12.9% judge, 10.3% F1 (209 Qs)
- **multi-hop**: 54.0% judge, 9.2% F1 (50 Qs)
- **open-domain**: 84.9% judge, 18.0% F1 (423 Qs)
- **single-hop**: 57.1% judge, 13.6% F1 (140 Qs)
- **temporal**: 64.2% judge, 11.7% F1 (165 Qs)

## Per-Conversation
- **conv-44**: 70.7% (87/123 excl. adv)
- **conv-47**: 76.0% (114/150 excl. adv)
- **conv-48**: 75.9% (145/191 excl. adv)
- **conv-49**: 73.1% (114/156 excl. adv)
- **conv-50**: 70.9% (112/158 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 182 | 4 (2%) | 5 (2%) | 173 (95%) |
| multi-hop | 23 | 17 (73%) | 1 (4%) | 5 (21%) |
| open-domain | 64 | 8 (12%) | 2 (3%) | 54 (84%) |
| single-hop | 60 | 23 (38%) | 8 (13%) | 29 (48%) |
| temporal | 59 | 11 (18%) | 5 (8%) | 43 (72%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **73.52%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |