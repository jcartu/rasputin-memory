# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-10 12:26
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 1986 (1540 non-adversarial, 446 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 68.18%**
- Token F1: 16.04%

## Including adversarial
- All categories: 55.29%
- Adversarial only: 10.76%

## Per-Category Breakdown
- **adversarial**: 10.8% judge, 11.5% F1 (446 Qs)
- **multi-hop**: 50.0% judge, 7.8% F1 (96 Qs)
- **open-domain**: 81.1% judge, 19.5% F1 (841 Qs)
- **single-hop**: 40.4% judge, 12.9% F1 (282 Qs)
- **temporal**: 64.2% judge, 12.2% F1 (321 Qs)

## Per-Conversation
- **conv-26**: 69.7% (106/152 excl. adv)
- **conv-30**: 70.4% (57/81 excl. adv)
- **conv-41**: 73.7% (112/152 excl. adv)
- **conv-42**: 66.8% (133/199 excl. adv)
- **conv-43**: 65.2% (116/178 excl. adv)
- **conv-44**: 65.0% (80/123 excl. adv)
- **conv-47**: 72.7% (109/150 excl. adv)
- **conv-48**: 72.8% (139/191 excl. adv)
- **conv-49**: 62.2% (97/156 excl. adv)
- **conv-50**: 63.9% (101/158 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 398 | 20 (5%) | 82 (20%) | 296 (74%) |
| multi-hop | 48 | 42 (87%) | 5 (10%) | 1 (2%) |
| open-domain | 159 | 29 (18%) | 24 (15%) | 106 (66%) |
| single-hop | 168 | 75 (44%) | 34 (20%) | 59 (35%) |
| temporal | 115 | 27 (23%) | 17 (14%) | 71 (61%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **68.18%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |