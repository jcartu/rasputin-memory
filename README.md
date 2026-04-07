# RASPUTIN Memory v0.7

![RASPUTIN Memory](assets/social-preview-1280x640.png)

[![CI](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A self-hosted memory backend for AI agents. RASPUTIN stores conversations as overlapping windows and LLM-extracted facts in Qdrant, with an LLM quality gate that prevents junk from entering the memory store.

Production-grade long-term memory for AI agents:

- Vector search (Qdrant) with two-lane retrieval (windows + facts)
- LLM-based fact extraction at ingest time
- Cross-encoder reranking (local, CPU)
- A-MAC quality gate on commits

Main server: [`tools/hybrid_brain.py`](tools/hybrid_brain.py)

---

## Architecture Overview (v0.7)

```text
Memory Commit
   │
   ├─► A-MAC quality gate (relevance/novelty/specificity)
   ├─► 5-turn overlapping windows (stride 2)
   ├─► LLM fact extraction (optional, Haiku)
   ├─► Embedding (nomic-embed-text, 768d)
   └─► Persist to Qdrant

Search (two-lane)
   │
   ├─► Multi-Query Expansion
   ├─► Query Embedding (nomic-embed-text, 768d)
   │
   ├─► Lane 1: Window search (45 slots) ──┐
   ├─► Lane 2: Fact search (15 slots)   ──┼─► Merge ─► Cross-encoder rerank ─► Top-60 to LLM
   └─► (Optional: BM25 keyword lane)    ──┘
```

### Core components

- API server: `tools/hybrid_brain.py`
- Fact extraction: `tools/brain/fact_extractor.py`
- Cross-encoder reranker: `tools/brain/cross_encoder.py`
- Maintenance jobs: `tools/memory_decay.py`, `tools/memory_dedup.py`

---

## How It Compares

| Feature | RASPUTIN | Mem0 | Zep | LightRAG |
|---------|----------|------|-----|----------|
| Vector search | ✅ Qdrant | ✅ | ✅ | ✅ |
| LLM fact extraction | ✅ | ❌ | ❌ | ❌ |
| Two-lane retrieval | ✅ windows + facts | ❌ | ❌ | ❌ |
| Cross-encoder reranking | ✅ local CPU | ❌ | ❌ | ❌ |
| LLM quality gate | ✅ A-MAC | ❌ | ❌ | ❌ |
| Contradiction detection | ✅ | ❌ | ❌ | ❌ |
| Self-hosted / no vendor lock | ✅ | ✅ | ❌ (SaaS) | ✅ |

---

## Benchmarks

Evaluated on [LoCoMo](https://github.com/snap-research/locomo) (ACL 2024), conv-0 (199 QA pairs). Two benchmark modes: production (Haiku answers, neutral judge — measures retrieval quality) and compare (gpt-4o-mini answers, generous judge — field-comparable). See [benchmarks/README.md](benchmarks/README.md) for methodology details.

### LoCoMo conv-0 (current best: two-lane retrieval)

| Mode | Non-adversarial | Overall |
|------|----------------|---------|
| Production (retrieval signal) | **69.7%** | 53.3% |
| Compare (field-comparable) | 72.4% | — |

| Category | Production | Questions |
|----------|-----------|-----------|
| Open-domain | 82.9% | 70 |
| Temporal | 73.0% | 37 |
| Multi-hop | 53.8% | 13 |
| Single-hop | 43.8% | 32 |
| Adversarial | 6.4% | 47 |

### Retrieval Quality (the actual signal)

| Metric | Value |
|--------|-------|
| Gold-in-ANY-chunk | 88.4% |
| Gold-in-Top-5 | 63.8% |
| Gold-in-Top-10 | 71.4% |

### Leaderboard Context

Other published systems use different methodologies (strong answer models, generous judges). Direct score comparison is not valid. RASPUTIN's compare-mode (72.4% non-adv, conv-0 only) uses gpt-4o-mini for answers.

| System | Reported Score | Methodology |
|--------|---------------|-------------|
| Backboard | 90.00% | GPT-4.1, generous judge |
| Memvid | 85.70% | GPT-4o, generous judge |
| MemMachine | 84.87% | Unknown |
| Memobase | 75.78% | Unknown |
| RASPUTIN (compare) | 72.4% | gpt-4o-mini, generous judge, conv-0 only |
| Zep | 75.14% | Unknown |
| mem0 | 66.88% | Unknown |

### Pipeline

```
nomic-embed-text (768d) → Two-lane search (windows + facts) → Cross-encoder rerank → Haiku/gpt-4o-mini → gpt-4o-mini judge
```

See `benchmarks/README.md` for how to run benchmarks and reproduce numbers.

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

### v0.7.0
- Two-lane retrieval: windows (45 slots) + LLM-extracted facts (15 slots)
- Cross-encoder reranker (ms-marco-MiniLM-L-6-v2, CPU)
- Structured fact extraction via Claude Haiku at ingest
- Windows-only chunking (individual turns proven to add 0pp)
- Ablation-tested: BM25, keyword/entity/temporal boosts, MMR, Cohere reranker all proven 0pp
- Benchmark infrastructure: production/compare modes, batch API (50% savings), failure analysis
- LoCoMo conv-0: 69.7% production, 72.4% compare (non-adversarial)
- Timing-safe auth, UTC datetimes, schema v0.7

### v0.6.0 — LoCoMo 89.81% (#2)
- LLM reranker (Claude Haiku), professional benchmark harness

### v0.5.0 — Search Quality Breakthrough
- Keyword overlap boosting, entity focus scoring
- recall@5: 0.67 → 0.82 (+22%), recall@10: 0.745 → 0.885 (+19%)

See [`CHANGELOG.md`](CHANGELOG.md) for full details.

---

## License

MIT — see [`LICENSE`](LICENSE).
