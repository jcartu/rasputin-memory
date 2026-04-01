# RASPUTIN Memory System — Setup Guide

Step-by-step from zero to a working hybrid memory system.

## Prerequisites

- Linux (Ubuntu 22.04+ or Arch recommended)
- Docker + Docker Compose
- Python 3.10+
- NVIDIA GPU (recommended for reranker; CPU fallback works but is slower). No GPU? See [CLOUD_PROVIDERS.md](CLOUD_PROVIDERS.md)
- 16GB+ RAM, 50GB+ disk

---

## Quick Start (5 minutes)

```bash
# 1. Clone the repo
git clone https://github.com/jcartu/rasputin-memory.git
cd rasputin-memory

# 2. Install dependencies
pip install -r requirements-core.txt
# Optional: pip install -r requirements-ml.txt  # for reranker/embedding GPU server

# 3. Configure
# Edit config/rasputin.toml if you need non-default ports or models

# 4. Start databases (Qdrant + FalkorDB)
docker-compose up -d

# 5. Create the Qdrant collection
curl -X PUT http://localhost:6333/collections/second_brain \
  -H 'Content-Type: application/json' \
  -d '{"vectors":{"size":768,"distance":"Cosine"},"optimizers_config":{"memmap_threshold":20000},"hnsw_config":{"m":16,"ef_construct":100}}'

# 6. Install Ollama + embedding model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text

# 7. Start the Hybrid Brain API
python3 tools/hybrid_brain.py
# Starts on http://localhost:7777
```

Or use the Makefile:

```bash
make setup    # steps 2-5
make start    # step 7
```

---

## Verify It Works

```bash
# Health check
curl http://localhost:7777/health
# Expected: {"status":"ok","components":{"qdrant":"up (0 pts)","falkordb":"up",...}}

# Commit a test memory
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text":"Test memory: the system is working correctly","source":"manual","force":true}'

# Search for it
curl "http://localhost:7777/search?q=system+working&limit=3"

# Check stats
curl http://localhost:7777/stats
```

---

## Step-by-Step Details

### Databases (Qdrant + FalkorDB)

The `docker-compose.yml` starts both:

```bash
docker-compose up -d
```

- **Qdrant** on port 6333 — vector database for 768-dim embeddings
- **FalkorDB** on port 6380 — Redis-based knowledge graph

Data is persisted in Docker volumes (`qdrant_storage`, `falkordb_data`).

### Embedding Service (Ollama + nomic-embed-text)

```bash
ollama pull nomic-embed-text
ollama serve   # if not already running as a systemd service
```

Verify:
```bash
curl -s http://localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":["test"]}' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Dim: {len(d[\"embeddings\"][0])}')"
# Expected: Dim: 768
```

> ⚠️ **All embeddings MUST use the same model.** Mixing models produces invisible vectors.

### Neural Reranker (Optional)

The system degrades gracefully without a reranker. To enable it:

```bash
python3 tools/brain/reranker.py
# Starts BGE reranker on port 8006
```

Requires GPU for reasonable speed. See `tools/brain/reranker.py` for details.

### LLM for A-MAC Quality Gate (Optional)

A-MAC scores incoming memories for quality. It needs an OpenAI-compatible LLM endpoint. Without it, the system fail-opens (accepts all commits).

Set `LLM_API_URL` in your `.env` to point to any OpenAI-compatible server (Ollama, vLLM, llama-swap, etc.).

---

## Production Deployment

Use any process manager to keep the server running:

```bash
# systemd (recommended for Linux)
# Create /etc/systemd/system/rasputin.service, then:
systemctl enable --now rasputin

# Docker (see docker-compose.yml)
docker compose up -d

# Or any process manager (supervisord, PM2, etc.)
python3 tools/hybrid_brain.py --port 7777
```

---

## Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| Qdrant | 6333 | Vector database |
| FalkorDB | 6380 | Knowledge graph |
| Ollama | 11434 | Embeddings (nomic-embed-text) |
| Reranker | 8006 | Neural reranker (optional) |
| Hybrid Brain API | 7777 | Main search/commit API |

See [CONFIGURATION.md](CONFIGURATION.md) for all options.
