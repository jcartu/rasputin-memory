# Benchmarks

## Synthetic Benchmark (offline, no services needed)
Tests memory-specific behaviors with mock embeddings.

```bash
make benchmark
```

## LoCoMo Benchmark (requires Qdrant + embedding service + LLM)
Standard conversational memory benchmark (ACL 2024, Snap Research).
10 conversations, ~2000 QA pairs across 5 categories.

Reports both token-level F1 AND LLM-judge accuracy (the metric used by
mem0, Zep, MemMachine, Memvid on their leaderboards).

### Run with local LLM:
```bash
python3 benchmarks/full_pipeline_bench.py
```

### Run with Claude Opus (answer gen) + Anthropic judge:
```bash
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... python3 benchmarks/full_pipeline_bench.py
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| LLM_BACKEND | openai | `openai` or `anthropic` |
| ANTHROPIC_API_KEY | | Required for anthropic backend |
| ANTHROPIC_MODEL | claude-opus-4-20250514 | Answer generation model |
| JUDGE_BACKEND | (same as LLM_BACKEND) | Judge LLM backend |
| JUDGE_MODEL | gpt-4o-mini | Judge model |
| BENCH_EMBED_URL | http://localhost:11434/api/embed | Embedding endpoint |
| BENCH_EMBED_MODEL | nomic-embed-text | Embedding model |
| BENCH_EMBED_DIM | 768 | Embedding dimensions |
