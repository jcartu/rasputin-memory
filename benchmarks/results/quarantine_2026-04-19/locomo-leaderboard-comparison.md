# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-18 20:36
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 1986 (1540 non-adversarial, 446 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 58.44%**
- Token F1: 15.03%

## Including adversarial
- All categories: 45.97%
- Adversarial only: 2.91%

## Per-Category Breakdown
- **adversarial**: 2.9% judge, 10.2% F1 (446 Qs)
- **multi-hop**: 33.3% judge, 8.1% F1 (96 Qs)
- **open-domain**: 73.5% judge, 17.6% F1 (841 Qs)
- **single-hop**: 30.5% judge, 11.6% F1 (282 Qs)
- **temporal**: 51.1% judge, 13.5% F1 (321 Qs)

## Per-Conversation
- **conv-26**: 55.9% (85/152 excl. adv)
- **conv-30**: 48.1% (39/81 excl. adv)
- **conv-41**: 56.6% (86/152 excl. adv)
- **conv-42**: 49.7% (99/199 excl. adv)
- **conv-43**: 57.3% (102/178 excl. adv)
- **conv-44**: 58.5% (72/123 excl. adv)
- **conv-47**: 67.3% (101/150 excl. adv)
- **conv-48**: 62.3% (119/191 excl. adv)
- **conv-49**: 60.9% (95/156 excl. adv)
- **conv-50**: 64.6% (102/158 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 433 | 54 (12%) | 166 (38%) | 213 (49%) |
| multi-hop | 64 | 42 (65%) | 15 (23%) | 7 (10%) |
| open-domain | 223 | 78 (34%) | 73 (32%) | 72 (32%) |
| single-hop | 196 | 97 (49%) | 58 (29%) | 41 (20%) |
| temporal | 157 | 26 (16%) | 26 (16%) | 105 (66%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **58.44%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |