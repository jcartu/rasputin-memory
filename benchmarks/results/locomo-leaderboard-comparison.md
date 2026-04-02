# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1

**Date:** 2026-04-02 14:21
**Pipeline:** Vector search (nomic-embed-text) → Top-60 → Claude Opus 4 → GPT-4o-mini judge
**Improvements:** LLM judge, Opus answers, exclude adversarial, no reranker, nomic prefixes, top-K 60
**Total questions:** 199 (152 non-adversarial, 47 adversarial)

## Headline Score (excluding adversarial)
**LLM-Judge Accuracy: 91.45%**
- Token F1: 15.80%

## Including adversarial
- All categories: 76.38%
- Adversarial only: 27.66%

## Per-Category Breakdown
- **adversarial**: 27.7% judge, 7.6% F1 (47 Qs)
- **multi-hop**: 100.0% judge, 8.7% F1 (13 Qs)
- **open-domain**: 90.0% judge, 20.0% F1 (70 Qs)
- **single-hop**: 87.5% judge, 6.9% F1 (32 Qs)
- **temporal**: 94.6% judge, 18.0% F1 (37 Qs)

## Per-Conversation
- **conv-26**: 91.4% (139/152 excl. adv)

## Leaderboard Comparison
| System | LLM-Judge Accuracy |
|--------|-------------------|
| Backboard | 90.00% |
| MemMachine | 84.87% |
| **RASPUTIN** | **91.45%** |
| Memobase | 75.78% |
| Zep | 75.14% |
| mem0 | 66.88% |