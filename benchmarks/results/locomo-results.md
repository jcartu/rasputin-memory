# LoCoMo Benchmark Results — RASPUTIN Memory

**Date:** 2026-04-02 02:03 UTC
**Total QA pairs:** 1986
**Embedding:** nomic-embed-text (768-dim) via Ollama
**LLM:** Qwen 3.5 122B-A10B via cartu-proxy
**Retrieval:** Vector search (cosine), top-20

## Overall Score

| Metric | Score |
|--------|-------|
| **F1** | **0.4144** |

## Per-Category Scores

| Category | Count | F1 |
|----------|-------|----|
| single-hop | 282 | 0.3956 |
| multi-hop | 321 | 0.4651 |
| temporal | 96 | 0.2031 |
| open-domain | 841 | 0.5841 |
| adversarial | 446 | 0.1153 |

## Leaderboard Comparison

| System | F1 Score |
|--------|----------|
| memmachine | 0.8487 |
| zep | 0.7514 |
| mem0 | 0.6688 |
| rasputin | 0.4144 ⬅️ |

## Methodology

- Each conversation ingested as individual turns into isolated Qdrant collection
- Turns formatted as `[date] Speaker: text` for temporal context
- Embeddings: Ollama nomic-embed-text (768-dim)
- Vector search with cosine similarity, top-10 retrieval
- Answer generation via Qwen 3.5 122B (local, via cartu-proxy)
- F1 computed as token-level overlap (same as LoCoMo paper)
