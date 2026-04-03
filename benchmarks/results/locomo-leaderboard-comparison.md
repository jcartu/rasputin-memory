# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-03 05:15
**Pipeline:** Window chunking → Multi-query search (top-120) → Dedup → Claude Opus 4 → GPT-4o-mini judge
**v2:** Adversarial prompt, conversation windows, multi-query, top-K 120, 50-chunk context
**Total questions:** 1986 (1540 non-adversarial, 446 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 91.36%**
- Token F1: 21.52%

## Including adversarial
- All categories: 83.94%
- Adversarial only: 58.30%

## Per-Category Breakdown
- **adversarial**: 58.3% judge, 17.0% F1 (446 Qs)
- **multi-hop**: 86.5% judge, 10.0% F1 (96 Qs)
- **open-domain**: 93.7% judge, 27.5% F1 (841 Qs)
- **single-hop**: 87.2% judge, 14.9% F1 (282 Qs)
- **temporal**: 90.3% judge, 15.3% F1 (321 Qs)

## Per-Conversation
- **conv-26**: 94.7% (144/152 excl. adv)
- **conv-30**: 93.8% (76/81 excl. adv)
- **conv-41**: 95.4% (145/152 excl. adv)
- **conv-42**: 89.4% (178/199 excl. adv)
- **conv-43**: 89.3% (159/178 excl. adv)
- **conv-44**: 91.9% (113/123 excl. adv)
- **conv-47**: 90.0% (135/150 excl. adv)
- **conv-48**: 91.1% (174/191 excl. adv)
- **conv-49**: 91.0% (142/156 excl. adv)
- **conv-50**: 89.2% (141/158 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **91.36%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |