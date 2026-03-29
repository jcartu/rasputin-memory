<!-- Social preview: 1280x640, set via GitHub Settings → Social preview -->

<div align="center">

<p align="center">
  <img src="assets/rasputin-logo-transparent.png" alt="RASPUTIN Memory" width="320">
</p>

# RASPUTIN Memory System

**Production-grade long-term memory architecture for AI agents.**

> **Start here:** [`tools/hybrid_brain.py`](tools/hybrid_brain.py) is the main server. Everything else feeds into it.

[![CI](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-8b5cf6?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge)](https://python.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-dc382c?style=for-the-badge)](https://qdrant.tech)
[![FalkorDB](https://img.shields.io/badge/FalkorDB-Graph_DB-6bbe4a?style=for-the-badge)](https://falkordb.com)


`134K+ vectors` · `107K graph nodes` · `125K graph edges` · `<150ms search` · `$0/query`

*Production deployment stats. A fresh install starts at 0 and scales with your data.*

*A production-grade open-source AI agent memory system. Achieves comparable or better retrieval quality through 4-stage hybrid pipeline — see Architecture for details.*

---

</div>

## Quick Start (One Command)

```bash
git clone https://github.com/jcartu/rasputin-memory.git
cd rasputin-memory
bash quickstart.sh
```

That's it. The script handles Docker services, Python deps, Ollama, embedding model, and Qdrant collection creation. Takes 5–15 minutes.

### Docker (one command)

```bash
docker run -d --name rasputin-memory \
  -p 7777:7777 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  ghcr.io/jcartu/rasputin-memory
```

> Requires Qdrant and Ollama running on the host. See [Quick Start](#quick-start-one-command) for full infrastructure setup.

> 🔒 **Optional auth:** Set `MEMORY_API_TOKEN` in your `.env` to require a bearer token on all API endpoints. See `.env.example`.

> 🖥️ **No GPU?** Everything works on CPU or with cloud APIs. See [`docs/CLOUD_PROVIDERS.md`](docs/CLOUD_PROVIDERS.md) for OpenAI, Cohere, and OpenRouter alternatives — starting at ~$5/month.

📖 **Full guide:** [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) — from clone to working brain with all optional components.

You can also use the Makefile: `make quickstart` or `make demo` (demo only).

<details>
<summary>Manual setup (if you prefer)</summary>

```bash
# 1. Clone and install
git clone https://github.com/jcartu/rasputin-memory.git
cd rasputin-memory
pip install -r requirements.txt
cp .env.example .env

# 2. Start databases
docker-compose up -d

# 3. Create the Qdrant collection
curl -X PUT http://localhost:6333/collections/second_brain \
  -H 'Content-Type: application/json' \
  -d '{"vectors":{"size":768,"distance":"Cosine"},"hnsw_config":{"m":16,"ef_construct":100}}'

# 4. Start embedding model
ollama pull nomic-embed-text
ollama serve

# 5. Start the Hybrid Brain API
python3 tools/hybrid_brain.py       # Standard mode (port 7777)
# or: python3 tools/hybrid_brain_v2_tenant.py  # Multi-tenant mode

# 6. Search and commit
curl "http://localhost:7777/search?q=your+query&limit=5"
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text": "Something worth remembering", "source": "manual"}'
```

</details>

> See [`docs/SETUP.md`](docs/SETUP.md) for the full deployment guide.

### Try It

```bash
# Store a memory
curl -X POST http://localhost:7777/commit \
  -H "Content-Type: application/json" \
  -d '{"text": "The project deadline is March 15th. We decided to use PostgreSQL.", "source": "meeting-notes"}'

# Search memories
curl "http://localhost:7777/search?q=database+decision&limit=3"
```

That's it. Two endpoints, instant results.

---

## What Is This?

RASPUTIN Memory is a self-hosted AI memory system that gives an autonomous agent **perfect recall** across every conversation, decision, and piece of context it has ever encountered. No cloud APIs. No token limits. No forgetting.

It combines **vector search** (Qdrant), **keyword search** (BM25), a **knowledge graph** (FalkorDB), and **neural reranking** (BGE cross-encoder) into a single hybrid retrieval pipeline — all running on local hardware at zero per-query cost.

**Standalone by design.** This memory system works with any AI agent, framework, or direct HTTP calls. OpenClaw integration is included but entirely optional.

This repo contains the full architecture documentation, implementation guides, and source code for every component.

### See It In Action

<p align="center">
  <img src="assets/demo.gif" alt="RASPUTIN Memory Demo — health check, commit, and hybrid search" width="800">
</p>

*Live terminal session: health check → commit a memory → hybrid search retrieval in <160ms*

---

## Why This Exists (And Why It's Different)

Every AI memory system picks one or two retrieval strategies and calls it a day. Here's what that looks like in practice:

- **OpenClaw's built-in `memorySearch`** — Markdown files + embeddings. No graph, no reranker, no quality gates. Fine for hobby use.
- **Mem0** (48K+ stars) — Vector + graph, but the knowledge graph is locked behind the $249/month Pro tier. No reranker, no quality gate, no procedural memory.
- **Zep** — Best-in-class temporal knowledge graph, but consumes 600K+ tokens per conversation (vs ~7K here) and takes hours to process new memories before they're searchable.
- **Hindsight** — Strong on benchmarks but single retrieval strategy. No graph, no BM25, no reranker.
- **LangMem** — 59-second p95 latency. Unusable for real-time agents.

RASPUTIN is the only system that combines **all of these** in a single pipeline:

| Capability | RASPUTIN | Mem0 | Zep | Hindsight | OpenClaw Native |
|---|---|---|---|---|---|
| 4-stage hybrid retrieval | ✅ | ➖ | ➖ | ➖ | ➖ |
| Knowledge graph (240K+ nodes) | ✅ | 🟡 Pro only ($249/mo) | ✅ | ➖ | ➖ |
| Neural cross-encoder reranker | ✅ | ➖ | ➖ | ➖ | ➖ |
| LLM quality gate (A-MAC) | ✅ | ➖ | ➖ | ➖ | ➖ |
| Procedural memory (Hebbian) | ✅ | ➖ | ➖ | ➖ | ➖ |
| Wiki generation from memory | ✅ | ➖ | ➖ | ➖ | ➖ |
| Multi-tenant agent isolation | ✅ | ➖ | ➖ | ➖ | ➖ |
| Ebbinghaus temporal decay | ✅ | ➖ | ✅ | ➖ | 🟡 Simple |
| Self-hosted, $0/query | ✅ | ➖ | ➖ | ✅ | ✅ |
| <150ms p95 latency | ✅ | 🟡 50-200ms | ➖ Hours for new | ✅ <50ms | ✅ <100ms |

**The key architectural insight:** Every other system picks 1-2 retrieval strategies. RASPUTIN runs 4 in parallel and fuses them. Vector search finds semantically similar memories. BM25 catches exact terms that embedding models miss. The knowledge graph traverses entity relationships. Then a neural cross-encoder reranks everything for precision. This is why it doesn't miss things.

---

## Architecture

```
                         ┌──────────────┐
                         │  User Query  │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │   Embedding  │
                         │ nomic-embed  │
                         │ 768-dim ~5ms │
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
       ┌──────▼───────┐ ┌──────▼───────┐ ┌────────▼────────┐
       │    Qdrant    │ │     BM25     │ │    FalkorDB     │
       │  Vector DB   │ │   Keyword    │ │   Knowledge     │
       │  134K docs   │ │    Search    │ │     Graph       │
       │    ~15ms     │ │     ~5ms     │ │   107K nodes    │
       └──────┬───────┘ └──────┬───────┘ └────────┬────────┘
              │                │                   │
              └────────────────┼───────────────────┘
                               │
                        ┌──────▼───────┐
                        │ Score Fusion │
                        │  RRF merge   │
                        │    ~2ms      │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │ BGE Reranker │
                        │Cross-encoder │
                        │  GPU ~40ms   │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │Final Results │
                        │ <150ms total │
                        └──────────────┘
```

### How Search Works

1. **Embed** — Query is converted to a 768-dimensional vector via `nomic-embed-text` on GPU (~5ms)
2. **Parallel Retrieval** — Three systems search simultaneously:
   - **Qdrant** — HNSW nearest-neighbor across 134K+ embedded documents (~15ms)
   - **BM25** — Keyword/term frequency matching for exact phrases (~5ms)
   - **FalkorDB** — Graph traversal across entity relationships (107K nodes, 125K edges, ~8ms)
3. **Score Fusion** — Reciprocal Rank Fusion merges results from all three sources (~2ms)
4. **Neural Reranking** — BGE cross-encoder rescores the top candidates for precision (~40ms)

> **End-to-end: <150ms p95** — runs on every single message without the user noticing.

---

## Memory Lifecycle

Memories flow through 7 stages: Create → Score → Store → Search → Decay → Consolidate → Deduplicate. A-MAC quality gates filter noise at ingest, Ebbinghaus decay manages relevance over time.

→ [Full details](docs/MEMORY-LIFECYCLE.md)

---

## Components

| Component | File | Description |
|-----------|------|-------------|
| Hybrid Brain API | `tools/hybrid_brain.py` | Core search + commit API server (port 7777) |
| Hybrid Brain (Tenant) | `tools/hybrid_brain_v2_tenant.py` | Multi-tenant variant of hybrid_brain.py — use hybrid_brain.py unless you need agent isolation |
| Memory Engine | `tools/memory_engine.py` | CLI + library for recall, commit, briefing, deep search |
| BM25 Search | `tools/bm25_search.py` | Keyword scoring + RRF fusion layer |
| Reranker Server | `tools/reranker_server.py` | BGE cross-encoder neural reranker HTTP server |
| Memory Consolidator | `tools/memory_consolidate.py` | 5-pass fact extraction from daily logs |
| Deduplication | `tools/memory_dedup.py` | Cosine similarity dedup scanner + remover |
| Memory Autogen | `tools/memory_autogen.py` | Nightly MEMORY.md regeneration from live data |
| STORM Wiki | `storm-wiki/generate.py` | Wiki article generation from memory (Stanford STORM) |
| Qdrant Retrieval Module | `storm-wiki/qdrant_rm.py` | Custom STORM retrieval adapter for Qdrant |
| BrainBox | `brainbox/brainbox.py` | Hebbian procedural memory system |
| Memory Decay | `tools/memory_decay.py` | Ebbinghaus-inspired memory lifecycle management |
| Enrichment Pipeline | `tools/enrich_second_brain.py` | A-MAC importance scoring + auto-tagging |
| Collection Consolidator | `tools/consolidate_second_brain.py` | Merges Qdrant collections into `second_brain` |
| Parallel Consolidator | `tools/memory_consolidator_v4.py` | Parallel fact extraction from session transcripts |
| Fact Extractor | `tools/fact_extractor.py` | Structured fact extraction from conversations (runs every 4h via cron) |
| Health Check | `tools/memory_health_check.py` | End-to-end memory pipeline health check |
| Smart Query | `tools/smart_memory_query.py` | Multi-query decomposition + enhanced search |
| Embedding Server | `tools/embed_server_gpu1.py` | GPU embedding server (nomic-embed-text) |
| Memory Audit | `tools/memory-audit.sh` | Qdrant collection audit script |
| Verify System | `tools/verify_memory_system.sh` | Full system verification |
| MCP Server | `tools/memory_mcp_server.py` | MCP protocol adapter for memory API |
| Graph Brain | [`graph-brain/`](graph-brain/) | FalkorDB knowledge graph layer |
| Honcho Integration | `honcho/` | Peer-derived context from Honcho dialectic engine |
| Predictive Memory | `predictive-memory/` | Access pattern tracking, anticipatory prefetch |
| OpenClaw-Mem Hook | `hooks/openclaw-mem/` | Auto-recall hook (optional — for OpenClaw users only) |

---

## Agent Integration

Integrate via HTTP API (`/search`, `/commit`) from any agent framework. Includes auto-recall hooks, hot context injection, and optional OpenClaw integration.

→ [Full details](docs/AGENT-INTEGRATION.md)

---

## Quality Gates

Every memory passes A-MAC scoring (Relevance, Novelty, Specificity — composite ≥ 4.0 required) before storage. Ebbinghaus-inspired decay keeps search results fresh over time.

→ [Full details](docs/QUALITY-GATES.md)

---

## Stats

| Metric | Value |
|--------|------:|
| Qdrant vectors (second_brain) | 134,000+ |
| Qdrant collections | 4 |
| FalkorDB nodes | 107,000+ |
| FalkorDB edges | 125,000+ |
| Embedding model | nomic-embed-text v1 (768-dim) |
| Embedding inference | Ollama, local GPU |
| Reranker | BGE Reranker v2 (cross-encoder) |
| Search latency (p95) | <150ms |
| Inference cost per query | $0 |

*Stats as of March 2026. Run `python3 tools/memory_health_check.py` for live numbers.*

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Setup Guide](docs/SETUP.md) | Step-by-step from zero to working memory system |
| [Architecture Deep Dive](docs/ARCHITECTURE.md) | How the 4-stage pipeline works internally |
| [Hybrid Brain API](docs/HYBRID-BRAIN.md) | Core API endpoints, search, and commit |
| [Memory Lifecycle](docs/MEMORY-LIFECYCLE.md) | 7-stage pipeline from creation to deduplication |
| [Quality Gates](docs/QUALITY-GATES.md) | A-MAC scoring and Ebbinghaus decay |
| [Agent Integration](docs/AGENT-INTEGRATION.md) | Auto-recall, hot context, framework integration |
| [Enrichment Pipeline](docs/ENRICHMENT.md) | A-MAC scoring, entity extraction, auto-tagging |
| [OpenClaw Integration](docs/OPENCLAW-INTEGRATION.md) | Hooks, auto-recall, hot context, session memory |
| [Configuration](docs/CONFIGURATION.md) | All options, env vars, ports, model choices |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and performance tuning |

See [CHANGELOG.md](CHANGELOG.md) for version history.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jcartu/rasputin-memory&type=Date)](https://star-history.com/#jcartu/rasputin-memory&Date)

## Built With

| Technology | Role |
|-----------|------|
| [Qdrant](https://qdrant.tech/) | Vector database — HNSW index, scalar quantization |
| [FalkorDB](https://www.falkordb.com/) | Graph database — in-memory Cypher traversals |
| [Ollama](https://ollama.ai/) | Local embedding inference (nomic-embed-text) |
| [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Cross-encoder neural reranking |
| [Stanford STORM](https://github.com/stanford-oval/storm) | Wiki article generation framework |
| [OpenClaw](https://github.com/openclaw/openclaw) | Agent framework with memory hooks |

## FAQ

**How is this different from RAG?**

RAG retrieves from static document corpora. RASPUTIN is a *living memory* that ingests, scores, decays, and consolidates memories over time — closer to how human memory works than to a document search engine. Memories strengthen with repeated access, weaken with age, and get consolidated into higher-level knowledge. RAG answers "what's in these documents?" RASPUTIN answers "what do you remember?"

**Do I need a GPU?**

No. CPU inference works for embeddings (nomic-embed-text via Ollama). A GPU speeds up the neural reranker and A-MAC quality scoring but neither is required — the system gracefully degrades to vector + BM25 search without them.

**Do I need OpenClaw?**

No. The memory system is standalone — it works with any AI agent, framework, or direct HTTP calls. The OpenClaw hook is an optional integration for automatic memory recall.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR process.

## License

MIT — See [LICENSE](LICENSE).

<div align="center">

*"Hard to kill, impossible to ignore." — RASPUTIN*

</div>

