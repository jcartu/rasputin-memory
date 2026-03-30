<!-- Social preview: 1280x640, set via GitHub Settings → Social preview -->

<div align="center">

<p align="center">
  <img src="assets/rasputin-logo-transparent.png" alt="RASPUTIN Memory" width="320">
</p>

# RASPUTIN Memory System

**Self-hosted long-term memory architecture for AI agents — published as a personal reference system.**

> **Start here:** [`tools/hybrid_brain.py`](tools/hybrid_brain.py) is the main server. Everything else feeds into it.

[![CI](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-8b5cf6?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge)](https://python.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-dc382c?style=for-the-badge)](https://qdrant.tech)
[![FalkorDB](https://img.shields.io/badge/FalkorDB-Graph_DB-6bbe4a?style=for-the-badge)](https://falkordb.com)


`<150ms search` · `$0/query` · `4-stage hybrid pipeline`

*This is a personal system published for reference. The architecture is genuinely novel; no formal benchmarks have been run to validate claims against other systems. A fresh install starts at 0 and scales with your data.*

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
# or: ENABLE_TENANTS=1 python3 tools/hybrid_brain.py  # Multi-tenant mode

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
| Knowledge graph | ✅ | 🟡 Pro only ($249/mo) | ✅ | ➖ | ➖ |
| Neural cross-encoder reranker | ✅ | ➖ | ➖ | ➖ | ➖ |
| LLM quality gate (A-MAC) | ✅ | ➖ | ➖ | ➖ | ➖ |
| Procedural memory (Hebbian) | ✅ | ➖ | ➖ | ➖ | ➖ |
| Wiki generation from memory | ✅ | ➖ | ➖ | ➖ | ➖ |
| Multi-tenant agent isolation | ✅ | ➖ | ➖ | ➖ | ➖ |
| Ebbinghaus temporal decay | ✅ | ➖ | ✅ | ➖ | 🟡 Simple |
| Self-hosted, $0/query | ✅ | ➖ | ➖ | ✅ | ✅ |
| <150ms p95 latency | ✅ | 🟡 50-200ms | ➖ Hours for new | ✅ <50ms | ✅ <100ms |

*Comparison based on publicly available documentation as of March 2026. This table reflects capabilities, not validated benchmark results — formal BEIR evaluations have not been run. See each project's docs for current capabilities.*

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
       │   Vector DB  │ │    Search    │ │     Graph       │
       │    ~15ms     │ │     ~5ms     │ │     Graph DB    │
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
   - **Qdrant** — HNSW nearest-neighbor search across embedded documents (~15ms)
   - **BM25** — Keyword/term frequency matching for exact phrases (~5ms)
   - **FalkorDB** — Graph traversal across entity relationships (~8ms)
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
| Hybrid Brain API | `tools/hybrid_brain.py` | Core search + commit API server (port 7777). Supports multi-tenant mode via `--tenant-mode` or `ENABLE_TENANTS=1`. |
| Memory Engine | `tools/memory_engine.py` | CLI + library for recall, commit, briefing, deep search |
| BM25 Search | `tools/bm25_search.py` | Keyword scoring + RRF fusion layer |
| Reranker Server | `tools/reranker_server.py` | BGE cross-encoder neural reranker HTTP server |
| Consolidator | `tools/consolidator.py` | Unified pipeline: daily logs (`memory`), session transcripts (`sessions`), collection migration (`migrate`) |
| Deduplication | `tools/memory_dedup.py` | Cosine similarity dedup scanner + remover |
| Memory Autogen | `tools/memory_autogen.py` | Nightly MEMORY.md regeneration from live data |
| STORM Wiki | `storm-wiki/generate.py` | Wiki article generation from memory (Stanford STORM) |
| Qdrant Retrieval Module | `storm-wiki/qdrant_rm.py` | Custom STORM retrieval adapter for Qdrant |
| BrainBox | `brainbox/brainbox.py` | Hebbian procedural memory system |
| Memory Decay | `tools/memory_decay.py` | Ebbinghaus-inspired memory lifecycle management |
| Enrichment Pipeline | `tools/enrich_second_brain.py` | A-MAC importance scoring + auto-tagging |
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
| Embedding model | nomic-embed-text (768-dim) |
| Embedding inference | Ollama, local GPU |
| Reranker | BGE Reranker v2 (cross-encoder) |
| Search latency (p95) | <150ms |
| Inference cost per query | $0 |

*Run `python3 tools/memory_health_check.py` for live stats from your deployment.*

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

## Benchmarks (v0.3.0)

Evaluated on [BEIR](https://github.com/beir-cellar/beir) benchmark datasets using the full local infrastructure
(nomic-embed-text-v2-moe embeddings, Qdrant, BGE reranker). All numbers are on 50-query test splits.

### BEIR Results

| Dataset | Pipeline | NDCG@10 | Recall@10 | Recall@100 |
|---------|----------|---------|-----------|------------|
| SciFact | Vector Only | 0.8230 | 0.8860 | 0.9800 |
| SciFact | **Hybrid Full** | **0.8336** | 0.8660 | 0.9800 |
| NFCorpus | Vector Only | 0.3710 | 0.1779 | 0.3136 |
| NFCorpus | **Hybrid Full** | 0.3323 | 0.1581 | 0.3136 |

Hybrid Full = Vector + BM25 (RRF) + Neural Reranker.

### Ablation Study (SciFact)

| Configuration | NDCG@10 | Recall@10 | MRR@10 | ΔNDCG |
|---------------|---------|-----------|--------|-------|
| 1. Vector Only | 0.8230 | 0.8860 | 0.8061 | — |
| 2. BM25 Only | 0.7866 | 0.8880 | 0.7744 | -0.036 |
| 3. Vector + BM25 (RRF) | 0.8111 | **0.9010** | 0.7912 | -0.012 |
| 4. Vector + BM25 + Reranker | **0.8322** | 0.8860 | **0.8287** | +0.009 |

The MoE embedding model (nomic-embed-text-v2-moe) is a strong baseline. RRF fusion boosts Recall@10 (+1.5%).
The neural reranker recovers NDCG (+0.9%) at the cost of some recall. Full results: [`benchmarks/ABLATION.md`](benchmarks/ABLATION.md).

Reproduce:
```bash
python3 benchmarks/run_beir.py --datasets scifact nfcorpus
python3 benchmarks/run_ablation.py --datasets scifact nfcorpus
```

## Library API (v0.3.0)

The core pipeline is now available as an importable Python package:

```python
from hybrid_brain import HybridSearch, RRFFusion, TemporalDecay, QualityGate

# Hybrid search over your Qdrant collection
search = HybridSearch(collection="second_brain")
results = search.query("Josh meeting about Q3 targets", limit=5)
for r in results:
    print(f"{r['score']:.3f}  {r['text'][:80]}")

# RRF fusion standalone
fuser = RRFFusion(k=60)
merged = fuser.fuse([vector_ranked_ids, bm25_ranked_ids])

# Temporal decay
decay = TemporalDecay()
results = decay.apply(results)

# AMAC quality gate (LLM-scored admission control)
gate = QualityGate(threshold=4.0)
if gate.evaluate(text).admitted:
    commit_memory(text)
```

The server entry point (`tools/hybrid_brain.py`) is unchanged — `python hybrid_brain.py` still starts the full server.
The library lives in `hybrid_brain/` and is independently importable without starting any server.

## Roadmap

- **Config file support** — Replace hardcoded env vars with a single `config.yaml` per-deployment
- **Async ingestion pipeline** — Non-blocking write path so query latency is not affected during batch imports
- **More BEIR datasets** — FiQA, TREC-COVID, HotpotQA evaluation

PRs toward any of these are welcome.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR process.

## License

MIT — See [LICENSE](LICENSE).

<div align="center">

*Self-hosted memory system — architecture published for reference.*

</div>

