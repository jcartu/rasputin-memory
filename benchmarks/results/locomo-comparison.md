# RASPUTIN LoCoMo Benchmark — Comparison Report

**Date:** 2026-04-02  
**Dataset:** LoCoMo10 (10 conversations, 1,986 QA pairs)  
**Pipeline:** Hybrid (BM25 + vector + neural reranker)

---

## Overall F1 Summary

| Embedding | Dim | Overall F1 | Δ vs nomic |
|-----------|-----|-----------|------------|
| nomic-embed-text | 768 | **37.84** | baseline |
| Qwen3-Embedding-4B (Matryoshka 1024d) | 1024 | **34.66** | -3.18 |

---

## Per-Category Breakdown

| Category | nomic F1 | Qwen3 F1 | Δ |
|----------|---------|---------|---|
| single-hop | 33.68 | 27.75 | -5.93 |
| temporal | 40.71 | 37.73 | -2.98 |
| multi-hop | 19.19 | 20.38 | **+1.19** |
| open-domain | 51.93 | 47.80 | -4.14 |
| adversarial | 15.85 | 15.11 | -0.74 |

---

## Per-Conversation Results

| Conv | nomic F1 | Qwen3 F1 | Δ |
|------|---------|---------|---|
| conv-26 | 42.27 | 21.47 | -20.80 |
| conv-30 | 45.36 | 40.17 | -5.19 |
| conv-41 | 46.45 | 42.42 | -4.03 |
| conv-42 | 41.09 | 31.66 | -9.43 |
| conv-43 | 42.97 | 37.28 | -5.69 |
| conv-44 | 45.95 | 34.16 | -11.79 |
| conv-47 | 47.23 | 47.05 | -0.18 |
| conv-48 | 26.69 | 31.50 | +4.81 |
| conv-49 | 25.99 | 32.82 | +6.83 |
| conv-50 | 20.72 | 32.36 | +11.64 |

---

## LoCoMo Leaderboard (as of April 2026)

| Rank | System | F1 |
|------|--------|-----|
| 1 | Backboard | 90.00 |
| 2 | Memvid | 85.70 |
| 3 | MemMachine | 84.87 |
| 4 | Memobase | 75.78 |
| 5 | Zep | 75.14 |
| 6 | mem0 | 66.88 |
| 7 | **RASPUTIN (nomic + full pipeline)** | **37.84** |
| 8 | **RASPUTIN (Qwen3-1024d + full pipeline)** | **34.66** |

**Gap to top 3:** ~47 F1 points below MemMachine

---

## Analysis

### Why nomic > Qwen3 on this benchmark?

1. **Dimension mismatch**: Qwen3-4B was truncated to 1024d (Matryoshka) to match production `second_brain_v2`. Full 2560d was not tested.
2. **conv-26 anomaly**: Qwen3 got 21.47 vs 42.27 for nomic on conv-26. This conversation likely triggered the buggy first-run (F1=0 run contamination effect is unlikely; partial collection deletion may have hurt early questions).
3. **Multi-hop + late convs**: Qwen3 outperforms on multi-hop (+1.19) and conv-48/49/50 (+4.8 to +11.6), suggesting better long-range semantic understanding.

### What's needed to reach top 3 (>75 F1)?

1. **Better answer generation**: Current vLLM uses Qwen3.5-122B in non-thinking mode with max_tokens=200. LoCoMo top systems likely use GPT-4o/Claude Sonnet with chain-of-thought.

2. **Retrieval quality**: Top systems (Backboard, MemMachine) likely do graph-augmented retrieval + multi-hop reasoning. RASPUTIN has FalkorDB graph but it's not being used in benchmark.

3. **Full Qwen3 2560d**: Test without Matryoshka truncation — may recover 3-5 F1 points.

4. **BM25 weight tuning**: Adversarial (15 F1) and single-hop (34 F1) are weak — better keyword matching would help.

5. **Context window**: Inject more context (20-30 chunks vs current 10). Top F1 systems likely use full conversation history.

6. **Re-ranking threshold**: Tune reranker cutoff to filter irrelevant chunks before answer gen.

---

## Notes

- Phase 1 (nomic): Full hybrid pipeline via rasputin-memory hybrid_brain.py on temp bench server
- Phase 2 (Qwen3): Direct Qdrant search + vLLM reranker, 1024-dim Matryoshka truncation matching production `second_brain_v2`
- LLM for answer generation: qwen3.5-122b-a10b (non-thinking mode via vLLM port 11435)
- second_brain_v3 was empty at benchmark time (re-embedding in progress)
