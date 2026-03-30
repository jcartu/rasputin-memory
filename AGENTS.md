# AGENTS.md — Rasputin Memory Project (Complete Reference)

## Project Overview
**Rasputin Memory** — Production-grade 4-stage hybrid retrieval memory system for AI agents.
- Vector search (Qdrant) + BM25 + Knowledge Graph (FalkorDB) + Neural Reranker
- Sub-150ms retrieval, self-hosted, $0/query
- 134K+ vectors, 107K graph nodes, 125K graph edges in production

## GitHub & Git
- **Repo:** https://github.com/jcartu/rasputin-memory.git (public, MIT license)
- **Default branch:** main
- **Current commit:** `0d0c812` — "Initial release: RASPUTIN Memory System v2.0"
- **Auth:** `gh` CLI authenticated as `jcartu` (PAT stored in `~/.config/gh/hosts.yml`)
- **Git config:** user=`j`, email=`josh@cartu.com`
- **Push:** `git add -A && git commit -m "msg" && git push origin main`
- **CI:** GitHub Actions workflow at `.github/workflows/ci.yml`

## Repo Structure
```
rasputin-memory/
├── tools/                    # Core Python modules
│   ├── hybrid_brain.py       # Main server (PM2 runs the live copy)
│   ├── memory_engine.py      # Memory engine abstraction layer
│   ├── hybrid_brain_v2_tenant.py  # Multi-tenant version
│   ├── reranker_server.py    # BGE reranker FastAPI server
│   ├── embed_server_gpu1.py  # GPU embedding server
│   ├── memory_consolidator_v4.py  # Consolidation pipeline
│   ├── fact_extractor.py     # Fact extraction from transcripts
│   ├── memory_dedup.py       # Deduplication tool
│   ├── memory_decay.py       # Memory decay/TTL management
│   ├── memory_health_check.py # Health diagnostics
│   ├── memory_autogen.py     # Auto-generate MEMORY.md
│   ├── memory_mcp_server.py  # MCP server for LLM tool use
│   ├── smart_memory_query.py # Smart query routing
│   ├── bm25_search.py        # BM25 keyword search module
│   ├── enrich_second_brain.py # Enrichment pipeline
│   ├── memory_consolidate.py # Older consolidation script
│   ├── consolidate_second_brain.py # Consolidation helper
│   ├── verify_memory_system.sh # System verification script
│   └── memory-audit.sh       # Audit script
├── brainbox/                 # BrainBox SQLite-based memory
│   ├── brainbox.py
│   └── README.md
├── storm-wiki/               # STORM wiki generator integration
│   ├── generate.py
│   ├── qdrant_rm.py
│   └── README.md
├── honcho/                   # Honcho context sync
│   ├── honcho-query.sh
│   ├── sync-honcho-context.sh
│   └── test-honcho-integration.py
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md
│   ├── HYBRID-BRAIN.md
│   ├── OPENCLAW-INTEGRATION.md
│   ├── SETUP.md
│   ├── GETTING_STARTED.md
│   ├── CONFIGURATION.md
│   ├── TROUBLESHOOTING.md
│   ├── EMBEDDINGS.md
│   ├── CRON_JOBS.md
│   ├── OPERATIONS.md
│   ├── CLOUD_PROVIDERS.md
│   ├── AGENT-INTEGRATION.md
│   ├── MEMORY-LIFECYCLE.md
│   ├── QUALITY-GATES.md
│   └── ENRICHMENT.md
├── .env.example              # Environment template
├── requirements.txt          # Python dependencies
├── Makefile                  # Build/run targets
├── quickstart.sh             # One-command setup
├── LICENSE                   # MIT
└── README.md
```

## Live Production System (THIS MACHINE — RASPUTIN)
The memory system is **running in production right now**. All services are on localhost.

### Services & Ports
| Service | Port | Manager | Process/Image | Status |
|---------|------|---------|---------------|--------|
| **hybrid-brain** (API) | 7777 | PM2 | `/home/josh/.openclaw/workspace/tools/hybrid_brain.py` (python3) | online |
| **Qdrant** (vector DB) | 6333, 6334 | Docker | `qdrant/qdrant:v1.17.0` container `rasputin-qdrant` | Up |
| **FalkorDB** (graph DB) | 6380→6379 | Docker | `falkordb/falkordb:latest` container `falkordb` | Up |
| **Ollama** (embeddings) | 11434 | systemd | nomic-embed-text v1 | active |
| **Reranker** (BGE) | 8006 | PM2 | `/home/josh/.openclaw/workspace/tools/reranker_server.py` (python3) | online |
| **PostgreSQL** | 5433 | systemd | ⚠️ NOT 5432 | active |
| **llama-swap** | 11436 | PM2 | Routes to vLLM + Ollama | online |
| **Qwen 35B** (A-MAC LLM) | via 11436 | llama-swap | GPU1/5090 | online |

### Qdrant Collections
| Collection | Vectors | Dimensions | Distance |
|-----------|---------|------------|----------|
| `second_brain` | 134,754 | 768 | Cosine |
| `memories_v2` | (active) | 768 | Cosine |
| `memories_archive` | (archive) | 768 | Cosine |
| `episodes` | (episodic) | 768 | Cosine |

### FalkorDB Graph
- Graph name: `brain`
- Nodes: 107,320
- Edges: 124,792
- Access: `redis-cli -p 6380 GRAPH.QUERY brain "<cypher>"`

### API Endpoints
```bash
# Health check
curl http://localhost:7777/health

# Search memories
curl "http://localhost:7777/search?q=<query>&limit=5"

# Commit new memory
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text":"your memory text", "source":"conversation"}'
```

## Environment & Credentials

### Git / GitHub
- `gh` CLI: authenticated as `jcartu` via PAT in keyring
- Git: user `j`, email `josh@cartu.com`
- Protocol: HTTPS

### API Keys (available in shell via /etc/environment + ~/.bashrc)
- `OPENAI_API_KEY` — set in ~/.bashrc
- All other keys in `/etc/environment`

### Live Service Configuration (.env equivalent)
```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=second_brain
FALKORDB_HOST=localhost
FALKORDB_PORT=6380
FALKORDB_GRAPH=brain
EMBED_PROVIDER=ollama
EMBED_URL=http://localhost:11434/api/embed
EMBED_MODEL=nomic-embed-text
EMBED_DIM=768
RERANKER_ENABLED=true
RERANKER_PROVIDER=local
RERANKER_URL=http://localhost:8006/rerank
LLM_PROVIDER=ollama
LLM_API_URL=http://localhost:11436/v1/chat/completions
LLM_MODEL=qwen3.5:35b
```

## Hardware
- **GPU0:** RTX PRO 6000 Blackwell 96GB (`GPU-ba6334bc-6fec-5f2c-df75-a887bbca476e`)
- **GPU1:** RTX 5090 32GB (`GPU-047f194b-7fbf-4a3d-0868-9a2df4da573b`) — embeddings + reranker + Qwen 35B
- **GPU2:** RTX PRO 6000 Blackwell 96GB (`GPU-538bf008-7ff2-0d1d-69e9-20db81a00459`)
- Total: 224GB VRAM

## Architecture — 4-Stage Retrieval Pipeline
1. **Vector Search** — Qdrant cosine similarity on nomic-embed-text v1 (768-dim) embeddings
2. **BM25 Keyword Search** — Full-text matching for exact terms (implemented in bm25_search.py)
3. **Knowledge Graph** — FalkorDB Cypher queries for entity/relationship traversal
4. **Neural Reranker** — BGE cross-encoder (bge-reranker-v2-m3) rescores top candidates

Results from all 4 stages are fused (reciprocal rank fusion) and reranked for final output.

### Quality Gate (A-MAC)
- Qwen 35B (via llama-swap on 11436) scores incoming commits
- Dimensions: Relevance, Novelty, Specificity
- Composite score < 4.0 = rejected
- Reject log: `/tmp/amac_rejected.log`

## Production vs Repo Files
The **live production** files are at `/home/josh/.openclaw/workspace/tools/` — PM2 runs those.
The **repo** files are at `/home/josh/.openclaw/workspace/rasputin-memory/tools/` — these are what GitHub has.

When making changes:
1. Edit the repo copy first
2. Test against the live services (all localhost)
3. If the change is good, sync to the production copy if needed
4. Push to GitHub: `git add -A && git commit -m "description" && git push origin main`

## ⚠️ Critical Rules
1. **Embeddings: Ollama nomic-embed-text v1 on port 11434 ONLY** — Port 8003 uses v1.5 (INCOMPATIBLE). Mixing = invisible/unfindable memories.
2. **PostgreSQL is on port 5433** (not 5432)
3. **NEVER restart hybrid-brain, reranker-gpu1, Qdrant, or FalkorDB without asking Josh**
4. **Test thoroughly before pushing** — this is a live system serving an AI agent 24/7
5. **Don't modify inference server configs** (`llama-serve-*.sh`, `llama-swap-*.yaml`)

## Useful Commands
```bash
# Check all services
pm2 status
docker ps

# Watch hybrid-brain logs
pm2 logs hybrid-brain --lines 50

# Test search
curl -s "http://localhost:7777/search?q=test query&limit=3" | python3 -m json.tool

# Qdrant info
curl -s http://localhost:6333/collections/second_brain | python3 -m json.tool

# FalkorDB query
redis-cli -p 6380 GRAPH.QUERY brain "MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC LIMIT 10"

# Run health check
make health

# Run tests
make test
```
