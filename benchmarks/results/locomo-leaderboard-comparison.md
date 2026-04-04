# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-04 09:47
**Pipeline:** Window chunking → Multi-query search (top-60) → Dedup → Claude Opus 4 → GPT-4o-mini judge
**v2:** Adversarial prompt, conversation windows, multi-query, top-K 60, 50-chunk context
**Total questions:** 1986 (1540 non-adversarial, 446 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 92.79%**
- Token F1: 20.32%

## Including adversarial
- All categories: 84.94%
- Adversarial only: 57.85%

## Per-Category Breakdown
- **adversarial**: 57.8% judge, 16.0% F1 (446 Qs)
- **multi-hop**: 89.6% judge, 9.5% F1 (96 Qs)
- **open-domain**: 94.8% judge, 25.3% F1 (841 Qs)
- **single-hop**: 90.4% judge, 14.6% F1 (282 Qs)
- **temporal**: 90.7% judge, 15.4% F1 (321 Qs)

## Per-Conversation
- **conv-26**: 96.7% (147/152 excl. adv)
- **conv-30**: 97.5% (79/81 excl. adv)
- **conv-41**: 95.4% (145/152 excl. adv)
- **conv-42**: 91.5% (182/199 excl. adv)
- **conv-43**: 92.1% (164/178 excl. adv)
- **conv-44**: 93.5% (115/123 excl. adv)
- **conv-47**: 92.0% (138/150 excl. adv)
- **conv-48**: 91.1% (174/191 excl. adv)
- **conv-49**: 91.0% (142/156 excl. adv)
- **conv-50**: 90.5% (143/158 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **92.79%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |