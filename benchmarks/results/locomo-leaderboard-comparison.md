# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-03 00:44
**Pipeline:** Vector search (nomic-embed-text) → Top-60 → Claude Opus 4 → GPT-4o-mini judge
**Improvements:** LLM judge, Opus answers, exclude adversarial, no reranker, nomic prefixes, top-K 60
**Total questions:** 1986 (1540 non-adversarial, 446 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 89.81%**
- Token F1: 16.21%

## Including adversarial
- All categories: 83.23%
- Adversarial only: 60.54%

## Per-Category Breakdown
- **adversarial**: 60.5% judge, 11.2% F1 (446 Qs)
- **multi-hop**: 86.5% judge, 9.6% F1 (96 Qs)
- **open-domain**: 91.9% judge, 19.6% F1 (841 Qs)
- **single-hop**: 82.6% judge, 11.4% F1 (282 Qs)
- **temporal**: 91.6% judge, 13.6% F1 (321 Qs)

## Per-Conversation
- **conv-26**: 96.1% (146/152 excl. adv)
- **conv-30**: 96.3% (78/81 excl. adv)
- **conv-41**: 86.2% (131/152 excl. adv)
- **conv-42**: 85.4% (170/199 excl. adv)
- **conv-43**: 90.4% (161/178 excl. adv)
- **conv-44**: 88.6% (109/123 excl. adv)
- **conv-47**: 92.0% (138/150 excl. adv)
- **conv-48**: 89.0% (170/191 excl. adv)
- **conv-49**: 87.8% (137/156 excl. adv)
- **conv-50**: 90.5% (143/158 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **89.81%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |