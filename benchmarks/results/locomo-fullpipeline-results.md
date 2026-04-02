# RASPUTIN Memory — LoCoMo Full Pipeline Benchmark Results

**Date:** 2026-04-02 03:11
**Pipeline:** BM25 + vector + reranker + entity boost + keyword overlap
**Conversations:** 10
**Total questions:** 1986

## Overall F1: 37.84

### Improvement over raw vector
- Raw vector: 41.44
- Full pipeline: 37.84
- **Improvement: +-3.60 (-8.7%)**

## Leaderboard
1. **Backboard**: 90.00
2. **Memvid**: 85.70
3. **MemMachine**: 84.87
4. **Memobase**: 75.78
5. **Zep**: 75.14
6. **mem0**: 66.88
7. **RASPUTIN raw vector**: 41.44
8. **RASPUTIN full pipeline**: 37.84 ← **YOU ARE HERE**

## Per-Category Breakdown
- **single-hop** (cat 1): 33.68 (282 questions)
- **temporal** (cat 2): 40.71 (321 questions)
- **multi-hop** (cat 3): 19.19 (96 questions)
- **open-domain** (cat 4): 51.93 (841 questions)
- **adversarial** (cat 5): 15.85 (446 questions)

## Per-Conversation Results
- **conv-26**: F1=42.27 (199 Qs, 419 committed)
- **conv-30**: F1=45.36 (105 Qs, 369 committed)
- **conv-41**: F1=46.45 (193 Qs, 663 committed)
- **conv-42**: F1=41.09 (260 Qs, 629 committed)
- **conv-43**: F1=42.97 (242 Qs, 680 committed)
- **conv-44**: F1=45.95 (158 Qs, 675 committed)
- **conv-47**: F1=47.23 (190 Qs, 689 committed)
- **conv-48**: F1=26.69 (239 Qs, 681 committed)
- **conv-49**: F1=25.99 (196 Qs, 509 committed)
- **conv-50**: F1=20.72 (204 Qs, 568 committed)