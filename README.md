# RASPUTIN Memory v0.9

![RASPUTIN Memory](assets/social-preview-1280x640.png)

[![CI](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jcartu/rasputin-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A self-hosted long-term memory backend for AI agents. RASPUTIN stores conversations as
overlapping windows and LLM-extracted facts in Qdrant, with Qwen3-Reranker reranking,
BM25 keyword search via SQLite FTS5, per-question prompt routing, and native MCP support
for Claude Code, Cursor, and any MCP-compatible client.

Production-grade long-term memory for AI agents:

- **MCP server** for Claude Code, Cursor, Codex, and any MCP client (FastMCP 3.2, streamable-http)
- **LLM memory synthesis** (`/reflect`) — retrieves memories and synthesizes coherent answers
- Vector search (Qdrant) with two-lane retrieval (windows + facts)
- BM25 keyword search (SQLite FTS5) with Reciprocal Rank Fusion
- LLM-based fact extraction at ingest time
- Qwen3-Reranker-0.6B foundation-model reranking (GPU, 0.99/0.0001 score separation)
- Per-question prompt routing (inference/factual/temporal)
- A-MAC quality gate on commits
- Knowledge graph (FalkorDB) with entity extraction
- 142 tests, 30+ ablation experiments with scientific methodology

API server: [`tools/hybrid_brain.py`](tools/hybrid_brain.py) — MCP server: [`tools/mcp/server.py`](tools/mcp/server.py)

---

## Architecture Overview

```text
MCP Client (Claude Code / Cursor / any MCP client)
   │
   └─► tools/mcp/server.py (port 8808, FastMCP 3.2)
       6 tools: store, search, reflect, stats, feedback, commit_conversation
       │
       └─► HTTP proxy ─► tools/hybrid_brain.py (port 7777)

Memory Commit
   │
   ├─► A-MAC quality gate (relevance/novelty/specificity)
   ├─► 5-turn overlapping windows (stride 2)
   ├─► LLM fact extraction (Haiku or local model)
   ├─► Embedding (nomic-embed-text, 768d)
   └─► Persist to Qdrant

Search
   │
   ├─► Multi-Query Expansion
   ├─► Query Embedding (nomic-embed-text, 768d)
   │
   ├─► Lane 1: Window search (45 slots) ──┐
   ├─► Lane 2: Fact search (15 slots)   ──┤
   ├─► Lane 3: BM25 keyword (FTS5, 10)  ──┼─► RRF Fusion ─► Qwen3-Reranker-0.6B ─► Top-60 to LLM
   │                                       │
   └─► Answer Prompt Routing ──────────────┘
       (inference / factual / temporal)

Reflect (LLM Synthesis)
   │
   ├─► hybrid_search(query, limit=20)
   ├─► Format top memories as context
   ├─► LLM call (Anthropic or Ollama)
   └─► Coherent synthesized answer + source citations
```

### Core components

- API server: `tools/hybrid_brain.py`
- MCP server: `tools/mcp/server.py` (thin HTTP proxy, FastMCP 3.2)
- LLM synthesis: `tools/brain/reflect.py`
- Fact extraction: `tools/brain/fact_extractor.py`
- Cross-encoder reranker: `tools/brain/cross_encoder.py`
- Maintenance jobs: `tools/memory_decay.py`, `tools/memory_dedup.py`

---

## How It Compares

| Feature | RASPUTIN | Mem0 | Zep | LightRAG |
|---------|----------|------|-----|----------|
| MCP protocol support | ✅ FastMCP 3.2 | ❌ | ❌ | ❌ |
| LLM memory synthesis | ✅ `/reflect` | ❌ | ❌ | ❌ |
| Vector search | ✅ Qdrant | ✅ | ✅ | ✅ |
| BM25 keyword search | ✅ SQLite FTS5 + RRF | ❌ | ❌ | ❌ |
| LLM fact extraction | ✅ | ❌ | ❌ | ❌ |
| Two-lane retrieval | ✅ windows + facts | ❌ | ❌ | ❌ |
| Foundation-model reranking | ✅ Qwen3-Reranker-0.6B (GPU) | ❌ | ❌ | ❌ |
| LLM quality gate | ✅ A-MAC | ❌ | ❌ | ❌ |
| Contradiction detection | ✅ | ❌ | ❌ | ❌ |
| Self-hosted / no vendor lock | ✅ | ✅ | ❌ (SaaS) | ✅ |

---

## Benchmarks

> **Full-dataset, fully-disclosed evaluation.** All numbers below are from the complete 10-conversation LoCoMo dataset (1986 questions), not a cherry-picked subset. Methodology, judge prompts, and all 30+ experiment records are public in this repository.

Evaluated on [LoCoMo](https://github.com/snap-research/locomo) (ACL 2024). Full 10-conversation 
dataset (1986 QA pairs). Two benchmark modes: production (Haiku answers, strict judge — measures 
retrieval quality) and compare (Haiku answers, generous judge — field-comparable).

### LoCoMo Full 10-Conv (v0.9 — current)

| Mode | Non-adversarial | Questions |
|------|----------------|-----------|
| **Compare (field-comparable)** | **77.7%** | 1540 |
| Production (retrieval signal) | **74.2%** | 1540 |

| Category | Production | Compare | Questions | Notes |
|----------|-----------|---------|-----------|-------|
| Open-domain | 84.8% | 83.2% | 841 | +2.9pp from v0.8 |
| Temporal | 71.3% | 75.4% | 321 | +6.5pp from v0.8 |
| Single-hop | 54.3% | 68.1% | 282 | +17.1pp from v0.8 — largest gain from Qwen3 CE |
| Multi-hop | 49.0% | 64.6% | 96 | −6.2pp production from v0.8 (see note) |
| Adversarial | 13.0% | 23.1% | 446 | Not an optimization target |

**Wide retrieval pool option:** Set `BENCH_LANE_WINDOWS=75 BENCH_LANE_FACTS=25` for single-hop-heavy workloads. Trades open-domain (−1.2pp) for single-hop (+4.2pp, 58.5%). Overall stays at 74.1%. See [Retrieval Pool Tuning](#retrieval-pool-tuning) below.

**Multi-hop note:** Multi-hop decreased in production mode (55.2% → 49.0%) because the Qwen3 reranker is better at ranking, which helps ranking-bound categories (single-hop +12.8pp, open-domain +3.0pp) but slightly hurts retrieval-bound categories where the gold content was never in the candidate pool. The compare mode number (64.6%) is higher because the generous judge credits partial answers. Our retrieval analysis shows 86% of multi-hop failures are retrieval misses (gold not in any chunk), not ranking failures — the reranker can't fix what retrieval didn't find.

### What's Been Tested (30+ Experiments)

| Experiment | Result | Status |
|-----------|--------|--------|
| **Qwen3-Reranker-0.6B** | **+4.5pp production, +8.6pp compare** | ✅ Shipped (v0.9) |
| **BM25 FTS5 + RRF fusion** | **+0.6pp with Qwen3 CE** | ✅ Shipped (v0.9) |
| **Wide retrieval pool (75w+25f)** | **Single-hop +4.2pp, open-dom −1.2pp** | ✅ Option (v0.9) |
| Prompt routing (inference/factual/temporal) | +1.6pp full 10-conv | ✅ Shipped (v0.8) |
| Two-lane search (windows + facts) | +6.5pp overall | ✅ Shipped (v0.7) |
| Cross-encoder reranking | Essential at two-lane | ✅ Shipped (v0.7) |
| Windows-only chunking (w5s2) | +5.2pp | ✅ Shipped (v0.7) |
| Pipeline strip (700→427 lines) | 0pp change, cleaner code | ✅ Shipped (v0.8) |
| BM25 with L-6 CE (3 variants) | −14pp to −28pp | ❌ CE too weak to filter |
| Consolidation (6 variants) | Net negative, all configs | ❌ Parked |
| Graph expansion (kNN links) | −4.4pp | ❌ Parked |
| Entity search (3 variants) | −10pp to −14pp | ❌ Parked |
| CE L-12 cross-encoder | −12.6pp single-hop | ❌ Reverted |
| Compare mode with gpt-4o-mini answers | −10.8pp vs Haiku | ❌ Haiku is better |
| Embedding upgrades (Qwen3 768d, 4096d) | 0pp or worse | ❌ No improvement |

Full experiment records in `experiments/`.

### On Benchmark Methodology

Published LoCoMo scores across memory systems are not directly comparable. Each system measures something different, uses different models, and reports under different conditions.

**What varies across systems:**

| Variable | Effect on Score | Example |
|----------|----------------|---------|
| Answer generation model | GPT-4o vs Haiku: ~20pp difference | A strong model rescues poor retrieval |
| Judge prompt leniency | "Be generous" vs neutral: ~5-10pp | Generous judges forgive vague answers |
| Context window size | 60 chunks vs 10: ~15pp | More context means ranking doesn't matter |
| Metric type | Retrieval recall vs answer accuracy | Fundamentally different measurements |

**What each system actually measures:**

| System | Metric | What It Tests |
|--------|--------|---------------|
| MemPalace | Retrieval recall | Whether the right evidence was found (no answer generated, no LLM) |
| LoCoMo original | Token F1 | Answer quality against gold standard (algorithmic, no LLM judge) |
| AMB/Hindsight | LLM judge accuracy | End-to-end: retrieval + answer + LLM evaluation |
| RASPUTIN | LLM judge accuracy | End-to-end with fixed, disclosed methodology |
| Memvid | LLM judge (claimed) | Methodology not published |

MemPalace's 96.6% LongMemEval score, for instance, is a retrieval recall metric — it measures whether the system found the right passage, not whether it generated a correct answer. This is a valid and useful metric, but it is not comparable to answer-accuracy scores reported by other systems.

Similarly, systems that use GPT-4o or Claude Opus for answer generation are primarily measuring LLM capability, not retrieval quality. A strong model can extract the correct answer from a large, poorly-ranked context window — which is exactly what our ablation program proved: at 60-chunk context, the entire ranking pipeline (BM25, keyword boosts, entity boosts, Cohere reranking, cross-encoder reranking) contributes 0pp because the answer model compensates.

**RASPUTIN's methodology is fully disclosed:**
- Production mode: Claude Haiku answers + strict judge (isolates retrieval quality)
- Compare mode: Claude Haiku answers + generous judge (field-comparable baseline)
- Judge model pinned to `gpt-4o-mini-2024-07-18` (prevents version drift)
- All benchmark code, judge prompts, and experiment results are in this repository

We report production-mode numbers as primary because they reflect actual retrieval quality. Compare-mode numbers are provided for rough context against other systems, with the caveat that methodology differences make direct comparison approximate at best.

For a standardized comparison, we recommend the [Agent Memory Benchmark](https://github.com/vectorize-io/agent-memory-benchmark) (AMB), which evaluates all systems under identical conditions with a published judge prompt.

| System | Reported Score | Benchmark | Methodology |
|--------|---------------|-----------|-------------|
| Hindsight | 92.0% | LoCoMo | AMB harness, published methodology |
| Backboard | 90.00% | LoCoMo | GPT-4.1, generous judge |
| MemMachine | 84.87% | LoCoMo | Not published |
| **RASPUTIN (compare)** | **77.7%** | **LoCoMo full 10-conv** | **Haiku answers, generous judge** |
| Memobase | 75.78% | LoCoMo | Not published |
| Zep | 75.14% | LoCoMo | Not published |
| **RASPUTIN (production)** | **74.2%** | **LoCoMo full 10-conv** | **Haiku answers, strict judge** |
| mem0 | 66.88% | LoCoMo | Not published |

† Only RASPUTIN and Hindsight publish their full evaluation methodology, judge prompts, and experiment data. Other scores are self-reported under undisclosed conditions. See [On Benchmark Methodology](#on-benchmark-methodology) below for why these numbers are not directly comparable.

### Pipeline

```
nomic-embed-text (768d) → Two-lane search (windows + facts) + BM25 FTS5 → RRF fusion → Qwen3-Reranker-0.6B → Haiku → gpt-4o-mini judge
```

See `benchmarks/README.md` for how to run benchmarks and reproduce numbers. See `experiments/` for the full ablation program and scientific record.

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

### 4) Start MCP server (optional — for Claude Code, Cursor, etc.)

```bash
pip install "fastmcp>=3.2.0"
python3 tools/mcp/server.py
# MCP server on http://127.0.0.1:8808/mcp

# Connect Claude Code:
claude mcp add --transport http rasputin http://localhost:8808/mcp
```

### 5) Smoke check

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

### `[reflect]`
- `provider` (string): LLM provider for synthesis (`anthropic` or `ollama`)
- `model` (string): model name (default `claude-haiku-4-5-20251001`)
- `max_tokens` (int): max tokens for synthesized answer (default `1000`)

### `[entities]`
- `known_entities_path` (string): entity dictionary JSON path

### Retrieval Pool Tuning

The default retrieval pool fetches 45 windows + 15 facts + 10 BM25 keywords = 70 candidates per query variant. The Qwen3-Reranker selects the best 60 for the answer model.

For workloads dominated by single-hop factual questions (e.g. "What is Alice's job?", "Where does Bob live?"), a wider pool improves accuracy by giving the reranker more candidates to choose from:

```bash
export BENCH_LANE_WINDOWS=75   # default: 45
export BENCH_LANE_FACTS=25     # default: 15
```

| Config | Overall | Single-hop | Open-domain | Temporal |
|--------|---------|------------|-------------|----------|
| Default (45w+15f) | **74.2%** | 54.3% | **84.8%** | **71.3%** |
| Wide (75w+25f) | 74.1% | **58.5%** | 83.6% | 70.4% |

The wider pool trades ~1pp open-domain for ~4pp single-hop. Use it when your users ask specific factual questions more than broad open-domain questions. The overall accuracy is unchanged.

Both configs use the same Qwen3-Reranker and BM25 FTS5 pipeline — only the initial candidate count differs.

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

### `POST /reflect`
Retrieves relevant memories and synthesizes a coherent answer via LLM.

```bash
curl -X POST http://localhost:7777/reflect \
  -H 'Content-Type: application/json' \
  -d '{"q":"What do we know about the auth service?","limit":20}'
```

Returns `{"answer": "...", "sources": [...], "search_elapsed_ms": ..., "reflect_model": "..."}`.

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

Coverage threshold is configured in `pyproject.toml` (`fail_under = 55`).

Test breakdown: 106 core pipeline + 22 MCP server proxy + 14 reflect module = **142 tests**.

---

## Version Notes

### v0.9.0
- **Qwen3-Reranker-0.6B**: Foundation-model reranker replacing ms-marco-MiniLM-L-6-v2 — 0.99 vs 0.0001 score separation (+4.5pp production, +8.6pp compare)
- **BM25 keyword search**: SQLite FTS5 in-memory sidecar with Reciprocal Rank Fusion (+0.6pp) — first positive BM25 result, enabled by the stronger reranker
- **Production: 74.2% non-adv** (+6.7pp from baseline), **Compare: 77.7% non-adv** (+10.2pp)
- Single-hop: 37.2% → 54.3% (+17.1pp) — largest category improvement from reranker upgrade
- 30+ experiments with scientific methodology (consolidation ×6, graph expansion, entity search ×3, BM25 ×4, CE A/B, compare mode variants)
- Multi-hop retrieval analysis: 60% extraction misses, 40% embedding misses — no ranking failures
- Reranker server supports both classic CrossEncoder and Qwen3 chat-template inference

### v0.8.0
- **MCP server** (`tools/mcp/server.py`): 6 tools via FastMCP 3.2 streamable-http transport — Claude Code, Cursor, Codex support
- **LLM memory synthesis** (`/reflect` endpoint): search → format → LLM → coherent answer with source citations
- **`tools/brain/reflect.py`**: Anthropic + Ollama LLM providers with automatic fallback
- Docker service for MCP server + deployment docs
- 36 new tests (22 MCP + 14 reflect), total 142 tests
- Full 10-conv validation: 69.1% non-adv (1986 questions, production mode)
- Prompt routing: +16.7pp multi-hop, +3.9pp single-hop
- Pipeline stripped from 700→427 lines (ablation-proven dead weight removed)
- Cross-encoder GPU server for remote inference
- Fact extraction module, consolidation engine, kNN link computation
- 21 experiments with scientific methodology
- Consolidation tested (6 variants) and parked — net negative with dense-only retrieval

### v0.7.0
- Two-lane retrieval: windows (45 slots) + LLM-extracted facts (15 slots)
- Cross-encoder reranker (ms-marco-MiniLM-L-6-v2, CPU)
- Structured fact extraction via Claude Haiku at ingest
- Windows-only chunking (individual turns proven to add 0pp)
- Ablation-tested: BM25, keyword/entity/temporal boosts, MMR, Cohere reranker all proven 0pp
- LoCoMo conv-0: 69.7% production, 72.4% compare (non-adversarial)

### v0.5.0 — Search Quality Breakthrough
- Keyword overlap boosting, entity focus scoring
- recall@5: 0.67 → 0.82 (+22%), recall@10: 0.745 → 0.885 (+19%)

See [`CHANGELOG.md`](CHANGELOG.md) for full details.

---

## License

MIT — see [`LICENSE`](LICENSE).
