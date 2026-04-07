# Benchmarks

## Modes

The benchmark runner supports two modes that control the answer model, judge prompt, and context settings.

### `--mode production` (default)

Tracks retrieval quality over time. Uses a weak answer model so metric changes reflect retrieval improvements, not answer generation improvements.

- Answer model: `claude-haiku-4-5-20251001`
- Judge: `gpt-4o-mini-2024-07-18` (pinned) with neutral prompt
- Context: 60 chunks, search limit 60
- Results: `{commit}-{bench}-production.json`

### `--mode compare`

Competition-comparable numbers. Matches the methodology used by Memvid, Backboard, and other published systems: generous judge prompt, same context depth.

- Answer model: `gpt-4o-mini` (override with `--answer-model gpt-4o`)
- Judge: `gpt-4o-mini-2024-07-18` (pinned) with generous prompt (field standard)
- Context: 60 chunks, search limit 60
- Results: `{commit}-{bench}-compare.json`

The leaderboard comparison table uses compare-mode numbers because that is the only apples-to-apples comparison with competitors who all use generous judging. Production-mode numbers are lower because the neutral judge and weaker answer model are more demanding, which is the point for tracking retrieval improvements.

## Cost

| Setup | Embeddings | Reranker | Answers | Cost/LoCoMo run |
|---|---|---|---|---|
| Default | nomic via Ollama | Cross-encoder (local) | Claude Haiku | ~$2-5 |
| OpenAI-only | nomic via Ollama | Cross-encoder (local) | gpt-4o-mini | ~$3-6 |
| Fully local (free) | nomic via Ollama | Cross-encoder (local) | Ollama qwen | $0 |
| Batch mode | any | any | any | 50% of above |

Note: Cohere reranker and Gemini embeddings are available but ablation testing proved they add 0pp at 60-chunk context. nomic-embed-text + local cross-encoder is the proven default.

## Reproducing Published Numbers

### Required

- Qdrant running on port 6333
- Ollama running on port 11434 with `nomic-embed-text` pulled
- `ANTHROPIC_API_KEY` (answer generation for production mode)
- `OPENAI_API_KEY` (judge model for both modes)

### Optional

- `COHERE_API_KEY` — enables Cohere rerank-v3.5 (ablation showed 0pp at 60-chunk context)
- `GEMINI_API_KEY` — enables Gemini embedding (ablation showed identical to nomic at 768d)

### Compare mode (leaderboard numbers)

```bash
python3 benchmarks/bench_runner.py locomo --mode compare
python3 benchmarks/bench_runner.py longmemeval --mode compare
```

### Production mode (retrieval tracking)

```bash
python3 benchmarks/bench_runner.py locomo --mode production
```

### Batch mode (50% cheaper, same results)

```bash
python3 benchmarks/bench_runner.py locomo --submit --mode production
# Wait for Anthropic batch (~1 hour)
python3 benchmarks/bench_runner.py locomo --retrieve
# Wait for OpenAI batch (up to 24 hours)
python3 benchmarks/bench_runner.py locomo --finalize
```

### Fully local (free)

```bash
EMBED_PROVIDER=ollama RERANK_PROVIDER=none \
  BENCH_ANSWER_MODEL=qwen3.5:9b \
  python3 benchmarks/bench_runner.py locomo --mode production
```

## Available Benchmarks

| Benchmark | Dataset | Questions | What it Tests |
|---|---|---|---|
| `locomo` | LoCoMo (ACL 2024) | ~2000 | Conversational memory: single-hop, multi-hop, temporal, open-domain, adversarial |
| `longmemeval` | LongMemEval (ICLR 2025) | 500 | Long-term interactive memory across session types |
| `frames` | FRAMES (Google 2024) | 824 | Multi-step reasoning over Wikipedia |
| `locomo-plus` | LoCoMo-Plus | varies | Cognitive/implicit memory |

## Comparing Commits

```bash
python3 benchmarks/bench_runner.py locomo --compare-to <old-hash>
```

Prints per-metric deltas and offers `git revert HEAD` on regression.

## Result Files

```
benchmarks/results/
  {hash}-{bench}-production.json    # tracked over time
  {hash}-{bench}-compare.json       # for leaderboard claims
  {hash}-{bench}-state.json         # batch phase state (resumable)
  history.csv                       # all production-mode runs
```
