# Cloud Provider Guide — No GPU Required

Every ML component in RASPUTIN can run via cloud APIs instead of local GPU. Mix and match — each component is independently swappable.

## What Needs ML

| Component | What It Does | Local Default | Cloud Alternative |
|-----------|-------------|---------------|-------------------|
| **Embeddings** | Converts text → 768-dim vectors | Ollama nomic-embed-text | OpenAI, Voyage, Cohere |
| **Reranker** | Re-scores search results for precision | BGE cross-encoder on GPU | Cohere, Jina, or skip it |
| **LLM (A-MAC)** | Quality-gates incoming memories | Local 35B model | Any OpenAI-compatible endpoint |
| **LLM (Facts/STORM)** | Extracts facts, generates research | Local 35B–122B model | OpenRouter, OpenAI, Anthropic |

---

## Embeddings

⚠️ **WARNING: Switching embedding providers means you must re-embed your entire Qdrant collection.** Vectors from different models are incompatible. Choose once, or plan for a migration.

### Option 1: OpenAI text-embedding-3-small (Recommended)

- **Price:** $0.02 per 1M tokens (~$0.50 to embed 100K memories)
- **Quality:** Excellent. Supports custom dimensions.
- **Setup:**

```bash
# .env
EMBED_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=768
```

The `dimensions=768` parameter tells OpenAI to output 768-dim vectors, matching Qdrant's collection config. This is native Matryoshka dimension reduction — no quality loss vs. the full 1536-dim output.

### Option 2: Voyage AI voyage-3

- **Price:** $0.06 per 1M tokens
- **Quality:** Top-tier on MTEB benchmarks, especially for code/technical content
- **Setup:**

```bash
# .env
EMBED_PROVIDER=voyage
VOYAGE_API_KEY=pa-...
EMBED_MODEL=voyage-3
EMBED_DIM=1024  # voyage-3 native dim — update Qdrant collection to match
```

Note: voyage-3 outputs 1024-dim vectors. You'll need to create your Qdrant collection with `size=1024` instead of 768.

### Option 3: Cohere embed-v3 (Free Tier)

- **Price:** Free tier: 100 requests/min, 1M tokens/month. Paid: $0.10 per 1M tokens.
- **Quality:** Strong multilingual support
- **Setup:**

```bash
# .env
EMBED_PROVIDER=cohere
COHERE_API_KEY=...
EMBED_MODEL=embed-english-v3.0
EMBED_DIM=768  # Cohere supports 768 natively
```

### Option 4: Ollama on CPU (Free, Slow)

No GPU needed — Ollama runs on CPU, just 10x slower (~500ms vs ~50ms per embed).

```bash
# .env (default — no changes needed)
EMBED_PROVIDER=ollama
EMBED_URL=http://localhost:11434/api/embed
EMBED_MODEL=nomic-embed-text
```

Good enough for small collections (<10K memories). Not practical for bulk re-embedding.

---

## Reranker

The neural reranker is **optional**. Disabling it causes ~10–15% quality loss on search precision — noticeable but not critical. Start without it and add later.

### Option 1: Skip It (Simplest)

```bash
# .env
RERANKER_ENABLED=false
```

The hybrid brain falls back to RRF score fusion without neural reranking.

### Option 2: Cohere Rerank v3

- **Price:** $1.00 per 1K searches. Free tier: 100 searches/month.
- **Quality:** Best cloud reranker available.
- **Setup:**

```bash
# .env
RERANKER_PROVIDER=cohere
COHERE_API_KEY=...  # same key as embeddings if using Cohere for both
RERANKER_MODEL=rerank-english-v3.0
```

### Option 3: Jina Reranker v2

- **Price:** $0.02 per 1K tokens (reranked). Free tier: 1M tokens/month.
- **Setup:**

```bash
# .env
RERANKER_PROVIDER=jina
JINA_API_KEY=jina_...
RERANKER_MODEL=jina-reranker-v2-base-multilingual
```

---

## LLM (A-MAC Scoring + Fact Extraction + STORM)

Any OpenAI-compatible chat completions endpoint works. The system sends standard `messages` format and parses structured output.

### Option 1: OpenRouter (Best Value — Multiple Models)

[OpenRouter](https://openrouter.ai) gives you access to 100+ models through a single API key and endpoint. Sign up, add $5 credits, and you're set.

**Setup:**

```bash
# .env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=google/gemini-2.0-flash-001    # Budget pick
```

**Budget models (A-MAC scoring, fact extraction):**

| Model | Price (input/output per 1M tok) | Notes |
|-------|------|-------|
| `google/gemini-2.0-flash-001` | $0.10 / $0.40 | Best bang for buck. Fast. |
| `meta-llama/llama-3.1-70b-instruct` | $0.50 / $0.70 | Strong open-source option |
| `mistralai/mistral-small-2501` | $0.10 / $0.30 | Very cheap, decent quality |

**Quality models (STORM research, complex extraction):**

| Model | Price (input/output per 1M tok) | Notes |
|-------|------|-------|
| `anthropic/claude-sonnet-4-6` | $3.00 / $15.00 | Best structured output quality |
| `google/gemini-2.5-pro-preview` | $2.50 / $15.00 | Long context, strong reasoning |
| `openai/gpt-4o` | $2.50 / $10.00 | Reliable all-rounder |

### Option 2: OpenAI Direct

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-4o-mini  # $0.15/$0.60 per 1M — cheapest OpenAI
```

### Option 3: Ollama on CPU (Free, Slow)

```bash
# .env
LLM_PROVIDER=ollama
LLM_API_URL=http://localhost:11434/v1/chat/completions
LLM_MODEL=llama3.1:8b
```

The 8B model runs on CPU with 8GB RAM. Quality is noticeably worse than cloud models for A-MAC scoring — expect more false positives/negatives. Usable for development and small-scale use.

---

## Recommended Stacks by Budget

### $0/month — All Local (CPU)

```bash
EMBED_PROVIDER=ollama
EMBED_URL=http://localhost:11434/api/embed
EMBED_MODEL=nomic-embed-text
RERANKER_ENABLED=false
LLM_PROVIDER=ollama
LLM_API_URL=http://localhost:11434/v1/chat/completions
LLM_MODEL=llama3.1:8b
```

- ✅ Free forever, fully private
- ❌ Slow embeddings (~500ms), weak A-MAC scoring, no reranker
- **Good for:** Development, testing, small personal use (<5K memories)

### ~$5/month — Cloud Starter

```bash
EMBED_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=768
RERANKER_PROVIDER=cohere
COHERE_API_KEY=...
RERANKER_MODEL=rerank-english-v3.0
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=google/gemini-2.0-flash-001
```

- ✅ Fast, good quality, Cohere free tier covers light reranking
- ❌ Data leaves your machine
- **Good for:** Personal use, 10K–50K memories, daily agent usage

### ~$20/month — Cloud Pro

```bash
EMBED_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=768
RERANKER_PROVIDER=cohere
COHERE_API_KEY=...
RERANKER_MODEL=rerank-english-v3.0
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=anthropic/claude-sonnet-4-6
```

- ✅ Best cloud quality, excellent A-MAC scoring, precise reranking
- ❌ Higher cost, data leaves your machine
- **Good for:** Production agents, 50K+ memories, business use

### Local GPU (Fastest, Most Private)

```bash
EMBED_PROVIDER=ollama
EMBED_URL=http://localhost:11434/api/embed
EMBED_MODEL=nomic-embed-text
RERANKER_ENABLED=true
RERANKER_URL=http://localhost:8006/rerank
LLM_PROVIDER=ollama
LLM_API_URL=http://localhost:11434/v1/chat/completions
LLM_MODEL=qwen2.5:32b
```

- ✅ Zero cost per query, full privacy, fastest inference
- ❌ Requires NVIDIA GPU with 8GB+ VRAM (24GB+ for 32B LLM)
- **Good for:** Power users, high-volume agents, privacy-critical deployments

---

## OpenRouter Quick Setup

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Add $5 credits (lasts weeks for typical agent use)
3. Create an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
4. Add to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=google/gemini-2.0-flash-001
```

OpenRouter is OpenAI-compatible — no code changes needed. Just swap the URL and key.

---

## Migration: Switching Embedding Providers

If you switch embedding providers on an existing collection, **all stored vectors become useless** — different models produce incompatible vector spaces.

### Migration Steps

1. Export all memory texts from Qdrant (the `text` field in each payload)
2. Delete and recreate the collection with the new vector dimensions
3. Re-embed and re-insert all memories using the new provider

```bash
# Export existing memories
python3 -c "
from qdrant_client import QdrantClient
import json
client = QdrantClient('localhost', port=6333)
points = client.scroll('second_brain', limit=100000, with_payload=True)[0]
with open('memory_export.jsonl', 'w') as f:
    for p in points:
        f.write(json.dumps(p.payload) + '\n')
print(f'Exported {len(points)} memories')
"

# Recreate collection with new dimensions (e.g., 768 for OpenAI)
# Then re-commit each memory through the /commit endpoint
```

This is why **choosing your embedding provider early matters**. Avoid switching unless necessary.
