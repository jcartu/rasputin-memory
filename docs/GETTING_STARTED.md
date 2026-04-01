# Getting Started — Clone to Working Brain

The definitive guide to setting up RASPUTIN Memory from scratch.

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Linux (Ubuntu 22.04+, Debian 12+, Arch) | Ubuntu 24.04 |
| Docker | 24.0+ with Compose v2 | Latest |
| Python | 3.10+ | 3.12 |
| RAM | 8 GB | 16 GB+ |
| Disk | 10 GB | 50 GB+ |
| GPU | None (CPU or cloud APIs work) | NVIDIA GPU with 4GB+ VRAM |

> 🖥️ **No GPU?** All ML components (embeddings, reranker, LLM) can run via cloud APIs. See [`CLOUD_PROVIDERS.md`](CLOUD_PROVIDERS.md) for setup — from $0 (CPU-only) to ~$5/month (cloud starter).

## 1. Clone & Bootstrap

```bash
git clone https://github.com/jcartu/rasputin-memory.git
cd rasputin-memory
chmod +x quickstart.sh
./quickstart.sh
```

The quickstart script handles everything: Docker services, Python deps, Ollama, embedding model, and Qdrant collection creation. Takes about 5–15 minutes depending on download speeds.

## 2. Verify Setup

```bash
# Health check
curl http://localhost:7777/health
```

Expected: all components show `"up"`.

## 3. Your First Memory

```bash
# Commit a memory
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text": "The project deadline is March 15, 2026. Budget is $50K.", "source": "test"}'
```

Expected: `{"status": "committed", "id": "..."}` (or similar success response).

## 4. Your First Search

```bash
curl 'http://localhost:7777/search?q=project+deadline&limit=5'
```

Expected: your committed memory appears in the results with a high relevance score.

## 5. Optional Components

### Neural Reranker (improves search precision)

The reranker rescores search results using a cross-encoder model. Significantly improves result quality but requires a GPU.

```bash
# Start the reranker server
python3 tools/brain/reranker.py
# Runs on port 8006 by default
```

Set `RERANKER_URL=http://localhost:8006/rerank` in `.env`.

### Knowledge Graph (FalkorDB)

Already started by `docker compose`. Enables entity-relationship queries alongside vector search. The hybrid brain automatically uses it if available.

### BM25 Keyword Search

Built into `hybrid_brain.py` — no extra setup needed. Complements vector search with exact keyword matching.

### STORM Research Pipeline

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full STORM (Synthesis Through Organized Research & Memory) pipeline documentation.

## 6. Setting Up Crons

For a self-maintaining memory system, set up automated jobs. See [CRON_JOBS.md](CRON_JOBS.md) for the full schedule.

Quick start — add these to your crontab (`crontab -e`):

```crontab
# Fact extraction every 4 hours
0 */4 * * * cd /path/to/rasputin-memory && python3 tools/fact_extractor.py >> logs/fact_extractor.log 2>&1

# Weekly dedup
0 5 * * 0 cd /path/to/rasputin-memory && python3 tools/memory_dedup.py --execute >> logs/dedup.log 2>&1
```

## 7. Integrating with Your AI Agent

### Raw HTTP API

The Hybrid Brain exposes a simple REST API:

```bash
# Search
GET  /search?q=<query>&limit=<n>

# Commit memory
POST /commit  {"text": "...", "source": "..."}

# Health check
GET  /health
```

### OpenClaw Integration

See [OPENCLAW-INTEGRATION.md](OPENCLAW-INTEGRATION.md) for hooking RASPUTIN into OpenClaw's memory pipeline.

### LangChain / Custom Agents

Use the HTTP API directly from any framework:

```python
import requests

# Search
results = requests.get("http://localhost:7777/search", params={"q": "project deadline", "limit": 5}).json()

# Commit
requests.post("http://localhost:7777/commit", json={"text": "Important fact here", "source": "my-agent"})
```

### MCP Server

For MCP-compatible agents:

```bash
# Example: map MCP tools to HTTP API endpoints in your MCP runtime
curl -s "http://localhost:7777/stats"
```

## 8. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Search returns no results | Embedding model mismatch | Verify `nomic-embed-text` v1 (768-dim). See [EMBEDDINGS.md](EMBEDDINGS.md) |
| `Connection refused` on 7777 | Hybrid brain not running | `python3 tools/hybrid_brain.py` |
| `Connection refused` on 6333 | Qdrant not running | `docker compose up -d` |
| Slow embeddings (>1s) | Running on CPU | Install NVIDIA drivers + CUDA, restart Ollama |
| A-MAC rejects everything | Threshold too high or LLM endpoint down | Check `LLM_API_URL` env var, or set `AMAC_THRESHOLD=0` to disable |
| FalkorDB errors | Redis container not healthy | `docker compose restart redis` |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more detailed diagnostics.
