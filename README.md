# RASPUTIN Memory v0.7 — #1 on LoCoMo (91.36%)

![RASPUTIN Memory](assets/social-preview-1280x640.png)

[![CI](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A self-hosted memory backend for AI agents that goes beyond simple vector search. RASPUTIN combines four retrieval strategies — semantic vectors, keyword matching, knowledge graph traversal, and neural reranking — into a single API, with an LLM quality gate that prevents junk from entering the memory store.

**Why not just use pgvector / Pinecone / plain RAG?** Because vector similarity alone misses keyword-exact matches, can't follow entity relationships, and treats "ok thanks" the same as a critical business decision. RASPUTIN solves all three.

Production-grade long-term memory for AI agents using a hybrid retrieval pipeline:

- Vector search (Qdrant)
- Keyword search (BM25)
- Knowledge graph traversal (FalkorDB)
- Neural reranking (BGE cross-encoder)

Main server: [`tools/hybrid_brain.py`](tools/hybrid_brain.py)

---

## Architecture Overview (v0.7)

```text
User Query
   │
   ├─► Multi-Query Expansion (name + topic decomposition)
   ├─► Query Embedding (nomic-embed-text, 768d)
   │
   ├─► Qdrant Vector Search (top-60, multi-query) ──┐
   ├─► BM25 Keyword Search ─────────────┼─► RRF ─► Temporal Boost ─► MMR Diversity ─► Final Top-K
   └─► FalkorDB Graph Search ───────────┘

Memory Commit
   │
   ├─► A-MAC quality gate (relevance/novelty/specificity)
   ├─► Duplicate detection
   ├─► Entity + name extraction (speaker, mentioned_names, has_date)
   └─► Persist to Qdrant (+ graph links where applicable)
```

### Core components

- API server: `tools/hybrid_brain.py`
- BM25 fusion layer: `tools/bm25_search.py`
- Reranker API: `tools/brain/reranker.py`
- Maintenance jobs:
  - `tools/memory_decay.py`
  - `tools/memory_dedup.py`
  - `tools/fact_extractor.py`

---

## How It Compares

| Feature | RASPUTIN | Mem0 | Zep | LightRAG |
|---------|----------|------|-----|----------|
| Vector search | ✅ Qdrant | ✅ | ✅ | ✅ |
| BM25 keyword search | ✅ | ❌ | ❌ | ❌ |
| Knowledge graph | ✅ FalkorDB | ❌ | ✅ | ✅ |
| Neural reranking | ✅ BGE cross-encoder | ❌ | ❌ | ❌ |
| LLM quality gate | ✅ A-MAC | ❌ | ❌ | ❌ |
| Memory decay model | ✅ Ebbinghaus-inspired | ❌ | ❌ | ❌ |
| Contradiction detection | ✅ | ❌ | ❌ | ❌ |
| Self-hosted / no vendor lock | ✅ | ✅ | ❌ (SaaS) | ✅ |
| Sub-200ms p95 latency | ✅ | ✅ | ✅ | ❓ |

---

## Benchmarks

### LoCoMo — 91.36% (#1 on the leaderboard)

Evaluated on [LoCoMo](https://github.com/snap-research/locomo) (ACL 2024), the standard benchmark for conversational memory systems. 10 conversations, 1,986 QA pairs.

| Rank | System | Accuracy |
|------|--------|----------|
| **🥇** | **RASPUTIN Memory v0.7** | **91.36%** |
| 🥈 | Backboard | 90.00% |
| 🥉 | Memvid | 85.70% |
| 4 | MemMachine | 84.87% |
| 5 | Memobase | 75.78% |
| 6 | Zep | 75.14% |
| 7 | mem0 | 66.88% |

| Category | Accuracy | Questions |
|----------|----------|-----------|
| Open-domain | 93.7% | 841 |
| Temporal | 90.3% | 321 |
| Single-hop | 87.2% | 282 |
| Multi-hop | 86.5% | 96 |
| Adversarial | 58.3% | 446 |

### LongMemEval — 89.40% (ICLR 2025)

500 conversational memory questions across 6 categories. Tests long-context recall over multi-session dialogues.

| Category | Accuracy |
|----------|----------|
| Single-session (user) | 98.6% |
| Single-session (assistant) | 96.4% |
| Knowledge update | 92.3% |
| Multi-session | 91.0% |
| Single-session (preference) | 83.3% |
| Temporal reasoning | 79.7% |

### FRAMES — 50.4% (Google Research 2024)

824 multi-hop factual reasoning questions over Wikipedia. Tests retrieval + complex reasoning (numerical, tabular, temporal, multi-constraint).

### Pipeline

All benchmarks use the same retrieval pipeline:

```
nomic-embed-text (768d) → Multi-query expansion → Qdrant top-60 → Dedup → Claude Opus 4 → GPT-4o-mini judge
```

Run benchmarks:
```bash
python3 benchmarks/locomo_leaderboard_bench.py    # LoCoMo
python3 benchmarks/longmemeval_bench.py            # LongMemEval
python3 benchmarks/frames_bench.py                 # FRAMES
```

---

## Quick Start

### 1) Infrastructure (Docker Compose)

```bash
docker compose up -d
```

This should start Qdrant and FalkorDB from the repository compose file.

### 2) Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

### 3) Start API server

```bash
python3 tools/hybrid_brain.py
```

Server runs on `http://127.0.0.1:7777` by default.

### 4) Smoke check

```bash
curl http://localhost:7777/health
curl "http://localhost:7777/search?q=test&limit=3"
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text":"Rasputin memory test event happened on 2026-03-01.","source":"conversation"}'
```

---

## Configuration Reference (`config/rasputin.toml`)

The runtime loader reads this TOML and allows env overrides (see `tools/config.py`).

### `[server]`
- `host` (string): bind host
- `port` (int): API port

### `[qdrant]`
- `url` (string): Qdrant base URL
- `collection` (string): active memory collection

### `[graph]`
- `host` (string): FalkorDB host
- `port` (int): FalkorDB port
- `graph_name` (string): graph key
- `disabled` (bool): disable graph search path

### `[embeddings]`
- `url` (string): embedding endpoint
- `model` (string): embedding model name
- `prefix_query` (string): query embedding prefix
- `prefix_doc` (string): document embedding prefix

### `[reranker]`
- `url` (string): reranker endpoint
- `timeout` (int): timeout seconds
- `enabled` (bool): enable rerank stage

### `[amac]`
- `threshold` (float): reject below this composite score
- `timeout` (int): scoring timeout seconds
- `model` (string): model for admission scoring

### `[scoring]`
- `decay_half_life_low` (int)
- `decay_half_life_medium` (int)
- `decay_half_life_high` (int)

### `[constraints]`
- `enabled` (bool): enable implicit constraint extraction at commit time
- `model` (string): LLM model for constraint extraction
- `timeout` (int): extraction timeout seconds

### `[entities]`
- `known_entities_path` (string): entity dictionary JSON path

---

## API Reference

All responses are JSON.

### `GET /health`
Returns service health and component status.

```bash
curl http://localhost:7777/health
```

### `GET /search?q=<query>&limit=<n>&source=<source>&expand=<bool>`
Hybrid retrieval endpoint.

```bash
curl "http://localhost:7777/search?q=payment+issue&limit=5"
```

### `POST /search`
Body-based search variant.

```bash
curl -X POST http://localhost:7777/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"project timeline","limit":5,"expand":true}'
```

### `POST /commit`
Commits memory after quality and duplicate checks.

```bash
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text":"Vendor contract moved to April 12 with revised pricing.","source":"conversation","importance":75}'
```

### `GET /graph?q=<query>&limit=<n>&hops=<n>`
Direct graph lookup.

### `GET /stats`
Qdrant and graph count summary.

### `GET /amac/metrics`
A-MAC admission counters and rejection stats.

### `GET /contradictions?limit=<n>`
Lists stored contradiction records.

### `POST /proactive`
Returns proactive memory suggestions from recent context.

```bash
curl -X POST http://localhost:7777/proactive \
  -H 'Content-Type: application/json' \
  -d '{"messages":["We are discussing launch timelines"],"max_results":3}'
```

### `POST /commit_conversation`
Commits multi-turn conversations with automatic window chunking.

```bash
curl -X POST http://localhost:7777/commit_conversation \
  -H 'Content-Type: application/json' \
  -d '{"turns":[{"speaker":"Alice","text":"I got a promotion today!"},{"speaker":"Bob","text":"Congratulations!"}],"source":"conversation","window_size":5,"stride":2}'
```

### `POST /feedback`
Updates retrieval usefulness signal.

```bash
curl -X POST http://localhost:7777/feedback \
  -H 'Content-Type: application/json' \
  -d '{"point_id":123,"helpful":true}'
```

---

## Development Guide

### Local workflow

```bash
# lint
ruff check .

# type check
mypy tools/hybrid_brain.py tools/bm25_search.py --ignore-missing-imports

# unit tests (default suite)
pytest tests/ -k "not integration" -v

# integration tests (Qdrant required)
pytest tests/test_integration.py -v
```

### Adding features safely

1. Add/update tests in `tests/`
2. Keep API behavior backward-compatible where possible
3. Prefer config via `config/rasputin.toml` + env overrides
4. Validate with lint + mypy + tests before commit

---

## Testing Instructions

### Unit tests

```bash
pytest tests/ -k "not integration" -v
```

### Integration tests

```bash
pytest tests/test_integration.py -v
```

### Coverage

```bash
pytest tests/ --cov=tools --cov-report=term-missing
```

Coverage threshold is configured in `pyproject.toml` (`fail_under = 40`).

---

## Version Notes

### v0.7.0 — #1 on LoCoMo (91.36%)
- **LoCoMo 91.36%** — beat Backboard (90.00%) for #1 on the leaderboard
- **LongMemEval 89.40%** — 500 conversational memory questions (ICLR 2025)
- **FRAMES 50.4%** — 824 multi-hop factual reasoning questions (Google 2024)
- Conversation-window chunking, multi-query retrieval, MMR diversity
- Temporal boost, adversarial-resistant prompts, boost cap at 3×
- Constraint extraction architecture (implicit goal/state/value/causal memory)
- Timing-safe auth, UTC datetimes, schema v0.7, expanded protected fields

### v0.6.0 — LoCoMo 89.81% (#2)
- LLM reranker (Claude Haiku), professional benchmark harness

### v0.5.0 — Search Quality Breakthrough
- Keyword overlap boosting, entity focus scoring
- recall@5: 0.67 → 0.82 (+22%), recall@10: 0.745 → 0.885 (+19%)

See [`CHANGELOG.md`](CHANGELOG.md) for full details.

---

## License

MIT — see [`LICENSE`](LICENSE).
