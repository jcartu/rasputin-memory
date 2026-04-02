# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-02 20:09
**Pipeline:** Vector search (nomic-embed-text) → Top-60 → Claude Opus 4 → GPT-4o-mini judge
**Improvements:** LLM judge, Opus answers, exclude adversarial, no reranker, nomic prefixes, top-K 60
**Total questions:** 1986 (1540 non-adversarial, 446 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 69.42%**
- Token F1: 12.88%

## Including adversarial
- All categories: 60.98%
- Adversarial only: 31.84%

## Per-Category Breakdown
- **adversarial**: 31.8% judge, 7.1% F1 (446 Qs)
- **multi-hop**: 76.0% judge, 9.4% F1 (96 Qs)
- **open-domain**: 65.8% judge, 14.8% F1 (841 Qs)
- **single-hop**: 69.9% judge, 9.7% F1 (282 Qs)
- **temporal**: 76.6% judge, 11.6% F1 (321 Qs)

## Per-Conversation
- **conv-26**: 84.9% (129/152 excl. adv)
- **conv-30**: 84.0% (68/81 excl. adv)
- **conv-41**: 69.1% (105/152 excl. adv)
- **conv-42**: 59.3% (118/199 excl. adv)
- **conv-43**: 47.8% (85/178 excl. adv)
- **conv-44**: 49.6% (61/123 excl. adv)
- **conv-47**: 88.7% (133/150 excl. adv)
- **conv-48**: 63.9% (122/191 excl. adv)
- **conv-49**: 69.9% (109/156 excl. adv)
- **conv-50**: 88.0% (139/158 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **69.42%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |