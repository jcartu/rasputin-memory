# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-05 02:21
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → claude-haiku-4-5-20251001 → gpt-4o-mini-2024-07-18 judge
**Mode:** production | top-K 60, 60-chunk context
**Total questions:** 1986 (1540 non-adversarial, 446 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 65.32%**
- Token F1: 15.70%

## Including adversarial
- All categories: 51.61%
- Adversarial only: 4.26%

## Per-Category Breakdown
- **adversarial**: 4.3% judge, 11.2% F1 (446 Qs)
- **multi-hop**: 38.5% judge, 7.9% F1 (96 Qs)
- **open-domain**: 78.7% judge, 18.8% F1 (841 Qs)
- **single-hop**: 50.0% judge, 12.9% F1 (282 Qs)
- **temporal**: 51.7% judge, 12.3% F1 (321 Qs)

## Per-Conversation
- **conv-26**: 61.8% (94/152 excl. adv)
- **conv-30**: 71.6% (58/81 excl. adv)
- **conv-41**: 68.4% (104/152 excl. adv)
- **conv-42**: 64.8% (129/199 excl. adv)
- **conv-43**: 66.3% (118/178 excl. adv)
- **conv-44**: 69.9% (86/123 excl. adv)
- **conv-47**: 64.0% (96/150 excl. adv)
- **conv-48**: 64.4% (123/191 excl. adv)
- **conv-49**: 63.5% (99/156 excl. adv)
- **conv-50**: 62.7% (99/158 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **65.32%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |