# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-08 22:05
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 1986 (1540 non-adversarial, 446 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 69.09%**
- Token F1: 16.25%

## Including adversarial
- All categories: 56.19%
- Adversarial only: 11.66%

## Per-Category Breakdown
- **adversarial**: 11.7% judge, 11.3% F1 (446 Qs)
- **multi-hop**: 55.2% judge, 8.5% F1 (96 Qs)
- **open-domain**: 81.1% judge, 19.7% F1 (841 Qs)
- **single-hop**: 41.1% judge, 12.9% F1 (282 Qs)
- **temporal**: 66.4% judge, 12.6% F1 (321 Qs)

## Per-Conversation
- **conv-26**: 73.0% (111/152 excl. adv)
- **conv-30**: 70.4% (57/81 excl. adv)
- **conv-41**: 72.4% (110/152 excl. adv)
- **conv-42**: 67.8% (135/199 excl. adv)
- **conv-43**: 64.0% (114/178 excl. adv)
- **conv-44**: 64.2% (79/123 excl. adv)
- **conv-47**: 76.0% (114/150 excl. adv)
- **conv-48**: 73.8% (141/191 excl. adv)
- **conv-49**: 64.1% (100/156 excl. adv)
- **conv-50**: 65.2% (103/158 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 394 | 20 (5%) | 82 (20%) | 292 (74%) |
| multi-hop | 43 | 37 (86%) | 4 (9%) | 2 (4%) |
| open-domain | 159 | 30 (18%) | 22 (13%) | 107 (67%) |
| single-hop | 166 | 78 (46%) | 31 (18%) | 57 (34%) |
| temporal | 108 | 30 (27%) | 12 (11%) | 66 (61%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **69.09%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |