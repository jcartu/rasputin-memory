# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-10 21:26
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 639 (505 non-adversarial, 134 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 67.72%**
- Token F1: 16.00%

## Including adversarial
- All categories: 54.93%
- Adversarial only: 6.72%

## Per-Category Breakdown
- **adversarial**: 6.7% judge, 11.0% F1 (134 Qs)
- **multi-hop**: 60.0% judge, 9.6% F1 (30 Qs)
- **open-domain**: 79.1% judge, 18.8% F1 (278 Qs)
- **single-hop**: 37.8% judge, 13.4% F1 (90 Qs)
- **temporal**: 65.4% judge, 12.6% F1 (107 Qs)

## Per-Conversation
- **conv-48**: 72.8% (139/191 excl. adv)
- **conv-49**: 64.7% (101/156 excl. adv)
- **conv-50**: 64.6% (102/158 excl. adv)

## Retrieval Oracle (failure diagnosis)
| Category | Wrong | Not in top-60 | In 60 not 10 | In top-10 but wrong |
|----------|-------|---------------|--------------|---------------------|
| adversarial | 125 | 9 (7%) | 27 (21%) | 89 (71%) |
| multi-hop | 12 | 9 (75%) | 1 (8%) | 2 (16%) |
| open-domain | 58 | 12 (20%) | 6 (10%) | 40 (68%) |
| single-hop | 56 | 29 (51%) | 14 (25%) | 13 (23%) |
| temporal | 37 | 13 (35%) | 3 (8%) | 21 (56%) |

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **67.72%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |