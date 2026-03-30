# 🏗️ RASPUTIN Memory System — Master Architecture Review & v3.0 Redesign Plan

**Reviewer:** CTO-level architecture audit  
**Date:** 2026-03-30  
**Codebase:** `rasputin-memory/` — ~15K lines across Python (11K) + JavaScript (4.5K)  
**Production state:** 127K+ vectors in Qdrant, FalkorDB graph (240K nodes, 535K edges), running 24/7

---

## PHASE 1: System Map

### Components Identified

| Component | Location | Lines | Role | Status |
|-----------|----------|-------|------|--------|
| **hybrid_brain.py** | tools/ | 1573 | Core API server (search/commit) on port 7777 | Production, monolith |
| **hybrid_brain_v2_tenant.py** | tools/ | 1573 | Multi-tenant fork of above | Dead code (identical minus tenant filter) |
| **memory_engine.py** | tools/ | 867 | CLI + multi-angle recall pipeline | Production, DUPLICATES hybrid_brain |
| **handler.js** | hooks/openclaw-mem/ | 1521 | OpenClaw hook (session capture, auto-recall) | Production |
| **database.js** | hooks/openclaw-mem/ | 626 | SQLite storage for observations/sessions | Production |
| **context-builder.js** | hooks/openclaw-mem/ | 703 | Session context injection | Production |
| **gateway-llm.js** | hooks/openclaw-mem/ | 250 | LLM calls for session summarization | Production |
| **fact_extractor.py** | tools/ | 602 | Cron: mines sessions for facts → Qdrant | Production |
| **memory_consolidator_v4.py** | tools/ | 479 | Parallel session consolidation | Production |
| **bm25_search.py** | tools/ | 143 | BM25 keyword scoring | Production |
| **reranker_server.py** | tools/ | 123 | BGE-reranker-v2-m3 HTTP wrapper | Production |
| **embed_server_gpu1.py** | tools/ | 67 | Embedding server wrapper | Production |
| **graph_api.py** | graph-brain/ | 268 | FalkorDB REST API on port 7778 | Production |
| **graph_query.py** | graph-brain/ | 263 | Graph query utilities | Likely unused |
| **migrate_to_graph.py** | graph-brain/ | 413 | One-time migration script | Dead after migration |
| **schema.py** | graph-brain/ | 83 | FalkorDB schema setup | Utility |
| **BrainBox** | brainbox/ | 348 | Hebbian procedural memory (SQLite) | Never integrated |
| **predictive-memory/** | predictive-memory/ | 1032 | Predictive caching layer | Never integrated |
| **storm-wiki/** | storm-wiki/ | 327 | Knowledge synthesis via STORM | Never integrated |
| **memory_dedup.py** | tools/ | 354 | Standalone dedup tool | Maintenance script |
| **memory_decay.py** | tools/ | 452 | Standalone decay tool | Maintenance script |
| **memory_consolidate.py** | tools/ | 456 | Earlier consolidation version | Superseded by v4 |
| **memory_autogen.py** | tools/ | 152 | Auto-generate test memories | Development utility |
| **smart_memory_query.py** | tools/ | 278 | Alternative query interface | Likely unused |
| **memory_health_check.py** | tools/ | 183 | Health check script | Maintenance |
| **memory_mcp_server.py** | tools/ | 148 | MCP protocol wrapper | Minimal/stub |
| **enrich_second_brain.py** | tools/ | 268 | Overnight enrichment cron | Production |
| **honcho/** | honcho/ | ~100 | Honcho integration (shell scripts) | Experimental, partially integrated |

### Data Stores
1. **Qdrant** (port 6333) — 127K+ vectors, 768d, nomic-embed-text. THE source of truth.
2. **FalkorDB** (port 6380) — Graph layer, 240K nodes, 535K edges. Entity relationships.
3. **SQLite** (~/.openclaw-mem/memory.db) — openclaw-mem observations, sessions, summaries.
4. **SQLite** (brainbox/brainbox.db) — Hebbian procedural memory. Unused.
5. **JSONL files** — facts.jsonl, access_log.jsonl, various state files.
6. **Markdown files** — hot-context/, last-recall.md, honcho-context.md, SESSION-MEMORY.md.
7. **JSON files** — entity_graph.json, known_entities.json, cache.json (predictive).

### Configuration Sources (The Chaos)
1. Environment variables (QDRANT_URL, EMBED_URL, RERANKER_URL, etc.)
2. `config/known_entities.json`
3. `memory/entity_graph.json`
4. Hardcoded constants in hybrid_brain.py (AMAC_THRESHOLD, SOURCE_WEIGHTS, STOP_WORDS, half-life values)
5. Hardcoded constants in memory_engine.py (different trigger lists, source tiers)
6. Hardcoded URLs in handler.js (template literals with env vars that don't actually resolve)
7. PM2 process config (external)
8. docker-compose.yml (ports, service names)
9. pyproject.toml (dependencies only)

**Verdict: 9+ configuration sources with no single source of truth. Critical.** 

---

## PHASE 2: Architectural Audit

### 1. Overall Architecture: Research Prototype with Production Traffic

**Rating: 4/10**

This is a classic "grow by accretion" system. Each component was added to solve a specific problem, but nobody ever stepped back to design the whole thing. The result is:

- **Two complete search implementations** doing almost the same thing (hybrid_brain.py and memory_engine.py) with different ranking algorithms, different dedup logic, different entity extraction, and different source tiering
- **Three storage layers** (Qdrant, FalkorDB, SQLite) with no unified abstraction
- **Four unfinished experiments** cluttering the repo (BrainBox, predictive-memory, storm-wiki, honcho)
- **A monolith (hybrid_brain.py)** that is simultaneously: HTTP server, embedding client, graph client, reranker client, A-MAC quality gate, entity extractor, dedup engine, temporal decay calculator, multi-factor scorer, access tracker, and proactive surfacing engine. 1573 lines in one file.

The system WORKS — and that's commendable for a personal project. But it's not architected, it's accumulated.

### 2. Data Model: Adequate But Fragile

**Rating: 5/10**

**Good:**
- Qdrant payload schema is reasonable: text, source, date, importance, retrieval_count
- Graph schema with typed entity nodes (Person, Organization, Project, Topic, Location) is solid
- A-MAC quality gating prevents garbage from entering the system

**Bad:**
- **No schema versioning.** 127K vectors with no way to know which embedding model version created each one. If you switch from nomic-embed-text v1 to v2-moe, old vectors are silently incompatible.
- **Point IDs are MD5 hash collisions waiting to happen:** `abs(int(hashlib.md5((text + str(time.time())).encode()).hexdigest()[:15], 16))` — this truncates to 15 hex chars and takes abs(), creating a VERY collision-prone ID space.
- **No importance recalculation.** Importance is set at commit time and never updated. A memory that was important 6 months ago may be irrelevant now.
- **No contradiction detection.** If you commit "User moved to Moscow" then later "User moved to St. Petersburg", both coexist forever.
- **`connected_to` as a flat list in Qdrant** duplicates what's already in FalkorDB. Data inconsistency risk.

### 3. The "Perfect Recall" Problem

Josh wants every new session to have instant, complete, relevant context without asking. Current architecture:

**What exists:**
- `handler.js` does auto-recall on every user message → writes to `last-recall.md` → agent reads it
- `SESSION-MEMORY.md` is written at bootstrap with pre-loaded memories
- Hot-context directory for recent cron outputs
- Honcho context injection (experimental)

**What's broken:**
- Auto-recall in handler.js uses **template literals that don't resolve** (`${MEMORY_API_URL:-http://...}` is JavaScript, not bash — these are literal strings, the fetch calls go to `${MEMORY_API_URL:-http://...}/search` which is... not a URL)
- The search query for auto-recall is just raw keyword extraction — no entity awareness, no context from current conversation
- Bootstrap pre-load uses a single hardcoded query: `"active+tasks+recent+decisions+important+context"` — same for every session regardless of who/what is being discussed
- Proactive surfacing exists in hybrid_brain.py (`/proactive` endpoint) but **nobody calls it**
- The predictive-memory module was designed exactly for this... but was never integrated

**What's needed for perfect recall:**
1. Session-aware context building (know what the user is likely to talk about based on time of day, recent topics, calendar)
2. Entity-triggered pre-fetch (mention "dad" → automatically pull all medical/health context)
3. Streaming context enrichment during conversation (not just at bootstrap)
4. Proactive contradiction surfacing ("You mentioned X last week, but now you're saying Y")

### 4. Dead Weight Analysis

| Component | Verdict | Justification |
|-----------|---------|---------------|
| **hybrid_brain_v2_tenant.py** | 🔴 KILL | Exact copy of hybrid_brain.py with 10 lines of tenant filter added. Dead code. |
| **memory_consolidate.py** | 🔴 KILL | Superseded by memory_consolidator_v4.py |
| **BrainBox** | 🔴 KILL | Interesting concept, never integrated, adds complexity with zero value. SQLite procedural memory that nobody reads. |
| **storm-wiki/** | 🔴 KILL | Research prototype, never integrated. STORM knowledge synthesis requires DSPy + expensive API calls for marginal value. |
| **predictive-memory/** | 🟡 ABSORB | Good IDEAS (temporal patterns, entity associations, heat maps) but the implementation was never connected. Extract the design, kill the code, rebuild as part of the core retrieval pipeline. |
| **honcho/** | 🟡 EVALUATE | Partially integrated into handler.js but the Honcho fetch URLs are broken template literals. If Honcho adds real value (dialectic reasoning), fix the integration. If not, kill it. |
| **smart_memory_query.py** | 🔴 KILL | Alternative query tool that duplicates memory_engine.py functionality |
| **memory_autogen.py** | 🟡 KEEP (dev) | Test data generator, useful for development |
| **graph_query.py** | 🔴 KILL | Standalone graph query tool duplicating graph_api.py functionality |
| **migrate_to_graph.py** | 🔴 KILL | One-time migration, already run |

### 5. What's Missing Entirely

1. **Contradiction detection** — No mechanism to detect when new memories contradict existing ones. Critical for a system meant to be the "single source of truth" about a user.

2. **Embedding version tracking** — 127K vectors with no `embedding_model` or `embedding_version` field. If you migrate models, you need to re-embed everything or live with mixed-model cosine distances (which are meaningless).

3. **Importance recalculation** — Importance is static. A memory about "looking for a car" loses relevance after you buy the car. No mechanism to decay/update importance based on newer information.

4. **Memory lifecycle management** — No archival, no soft-delete, no "this is outdated" flag. Everything lives forever at equal standing.

5. **Multi-modal memory** — Images, voice notes, diagrams, screenshots — all reduced to text. No image embedding, no CLIP-style search.

6. **Temporal reasoning** — "What was I working on last Tuesday?" requires scanning dates. No temporal index, no calendar awareness.

7. **Feedback loop** — No way for the agent to signal "this memory was helpful" or "this was irrelevant to what I needed." Retrieval count exists but isn't meaningfully used for learning.

8. **Test infrastructure** — Tests only verify files compile. Zero functional tests, zero integration tests, zero benchmarks. For a system processing 127K+ vectors in production, this is dangerous.

### 6. Code Duplication: The Twin Brains Problem

**hybrid_brain.py** and **memory_engine.py** are two complete search systems that evolved independently:

| Feature | hybrid_brain.py | memory_engine.py |
|---------|----------------|-----------------|
| Search | Qdrant client library | Raw HTTP to Qdrant API |
| Embedding | Single text, retries | Batch embed, no retries |
| Entity extraction | Known entities + regex | Entity graph JSON + regex |
| BM25 | Via imported module | Via imported module |
| Reranking | Custom neural_rerank() | Different rerank() function |
| Temporal decay | Ebbinghaus power-law | Not implemented |
| Multi-factor scoring | Yes (5 factors) | Not implemented |
| Deduplication | Cosine + text overlap | Hash-based by source type |
| Graph search | FalkorDB Cypher direct | entity_graph.json file lookup |
| A-MAC gate | Yes | No |
| Proactive surfacing | Yes (unused) | No |
| Query expansion | Basic | Sophisticated (12 angles) |
| Source tiering | No | Yes (Gold/Silver/Bronze) |

**This is the #1 architectural problem.** Two divergent implementations of the same pipeline, each with features the other lacks. memory_engine.py has better query expansion; hybrid_brain.py has better ranking. Neither is complete.

### 7. Configuration Chaos

Counted **9+ distinct configuration mechanisms** with no single source of truth (see list above). The handler.js file has JavaScript template literals that look like bash parameter expansion (`${MEMORY_API_URL:-http://...}`) — these are LITERAL STRINGS in JavaScript, meaning the Qdrant/Honcho fetch calls are going to invalid URLs. This is a production bug.

### 8. OpenClaw Integration

**Rating: 3/10**

The hook system (`hooks/openclaw-mem/`) is the bridge between OpenClaw and the memory system. Issues:

- **handler.js is 1521 lines** handling 7+ event types with inline business logic, Honcho integration, auto-recall, session management, and MCP server spawning. No separation of concerns.
- **Broken URL templates** as noted — `${MEMORY_API_URL:-http://...}` doesn't work in JavaScript
- **Two separate auto-recall implementations** — one in handleMessage() and one in handleUserPromptSubmit(), both doing the same Qdrant search with slightly different logic
- **Session summary uses "Kimi"** (comment says DeepSeek, log says Kimi) — unclear which model actually runs
- **No graceful degradation** — if hybrid_brain is down, the hook silently fails and the agent gets no context
- **The SQLite database (database.js)** stores observations/sessions separately from Qdrant, creating a second source of truth that's never queried meaningfully

---

## PHASE 3: The Redesign Plan

### What We KEEP

| Component | Justification |
|-----------|---------------|
| **Qdrant as primary vector store** | 127K+ vectors, battle-tested, great API. No reason to switch. |
| **FalkorDB as graph layer** | Entity relationships are genuinely valuable for 2-hop traversal. Keep but simplify the integration. |
| **BM25 hybrid search** | Keyword matching catches what embeddings miss. Proven value. |
| **Neural reranker (BGE-reranker-v2-m3)** | Dramatically improves result quality. Keep. |
| **A-MAC quality gate** | Prevents garbage memories. One of the best features. Keep and improve. |
| **Ebbinghaus temporal decay** | Elegant memory decay model. Keep. |
| **Multi-factor scoring** | Importance × recency × source × retrieval. Sound approach. Keep. |
| **fact_extractor.py** | Autonomous knowledge mining is high-value. Refactor but keep concept. |
| **OpenClaw hook architecture** | The event model (bootstrap, message, stop, tool:post) is correct. Refactor implementation. |
| **Entity extraction** | Both regex and known-entity approaches have value. Unify them. |

### What We KILL

| Component | Justification |
|-----------|---------------|
| **hybrid_brain_v2_tenant.py** | Dead fork. Tenant isolation should be a config flag, not a file copy. |
| **memory_engine.py (as standalone)** | Merge its best features (query expansion, source tiering) INTO hybrid_brain, then delete. |
| **memory_consolidate.py** | Superseded by v4. |
| **BrainBox** | Never integrated. If Hebbian patterns matter, build them into the core later. |
| **storm-wiki/** | Research toy. No production value. |
| **smart_memory_query.py** | Duplicate of memory_engine. |
| **graph_query.py** | Duplicate of graph_api. |
| **migrate_to_graph.py** | One-time script, already run. |
| **Honcho integration** (unless proven valuable) | Broken URL templates, unclear value vs. native recall. Evaluate in 2 weeks, kill if no measurable improvement. |
| **The SQLite observations DB** (database.js) | If tool observations matter, commit them to Qdrant with a `source: "tool_observation"` tag. Don't maintain a parallel database. |
| **predictive-memory/ code** | Extract design principles, kill implementation. Rebuild as part of core. |

### What We BUILD NEW

| Component | Priority | Effort | Justification |
|-----------|----------|--------|---------------|
| **Unified Search Pipeline** | P0 | 2 weeks | Merge hybrid_brain.py + memory_engine.py into a single, modular pipeline with pluggable stages. |
| **Schema versioning** | P0 | 2 days | Add `embedding_model`, `embedding_version`, `schema_version` to every Qdrant point. Backfill existing 127K vectors. |
| **Config consolidation** | P0 | 3 days | Single `config.yaml` or `config.toml` for all components. Environment variables override file config. |
| **Contradiction detector** | P1 | 1 week | On commit: search for semantically similar memories, flag potential contradictions, ask user or auto-resolve. |
| **Importance recalculator** | P1 | 3 days | Cron job: re-evaluate importance based on recency of related memories, user engagement, topic currency. |
| **Proactive context engine** | P1 | 1 week | Replace the broken auto-recall with a proper context engine: entity extraction → predictive fetch → session-aware ranking. |
| **Memory lifecycle manager** | P1 | 3 days | Soft-delete, archive, mark-as-outdated. Not everything should live forever. |
| **Real test suite** | P0 | 1 week | Unit tests for each pipeline stage. Integration tests against test Qdrant collection. Benchmark suite for recall quality. |
| **Handler.js refactor** | P1 | 3 days | Split into small event handlers. Fix URL templates. Remove duplicate auto-recall. |
| **Embedding migration tool** | P2 | 1 week | When switching embedding models, batch re-embed all vectors while keeping the system live. |
| **Feedback loop** | P2 | 3 days | Agent signals "helpful" / "not helpful" on recalled memories. Adjusts future retrieval weight. |
| **Temporal index** | P2 | 3 days | Qdrant payload index on date field + calendar-aware queries ("last Tuesday"). |

### Target Architecture (v3.0)

```
┌──────────────────────────────────────────────────────────────────┐
│                        OpenClaw Gateway                          │
│  Events: bootstrap | message | tool:post | stop | command:new    │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Memory Hook (handler.js)                       │
│  Thin event router → delegates to modules:                       │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────┐            │
│  │ AutoRecall   │ │ SessionMgr   │ │ CommitBridge   │            │
│  │ (context     │ │ (lifecycle,  │ │ (extract facts, │            │
│  │  injection)  │ │  summaries)  │ │  commit to API) │            │
│  └──────┬──────┘ └──────────────┘ └───────┬────────┘            │
└─────────┼─────────────────────────────────┼──────────────────────┘
          │                                 │
          ▼                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Memory API (rasputin-server)                     │
│                  Single Python process, port 7777                 │
│                                                                   │
│  ┌─── Endpoints ────────────────────────────────────────────┐    │
│  │ GET  /search?q=...     → Unified Search Pipeline         │    │
│  │ POST /commit           → Admission → Store → Index       │    │
│  │ POST /proactive        → Context-Aware Surfacing         │    │
│  │ GET  /health           → Component Health                │    │
│  │ GET  /stats            → System Metrics                  │    │
│  │ POST /feedback         → NEW: Relevance Feedback         │    │
│  │ GET  /contradictions   → NEW: Contradiction Check        │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─── Unified Search Pipeline ──────────────────────────────┐    │
│  │ 1. Query Expansion (from memory_engine.py)                │    │
│  │ 2. Batch Embed (Ollama nomic-embed-text)                  │    │
│  │ 3. Qdrant ANN Search (multi-angle)                        │    │
│  │ 4. FalkorDB Graph Traversal (entity → 2-hop)             │    │
│  │ 5. BM25 Keyword Rerank                                    │    │
│  │ 6. Temporal Decay (Ebbinghaus)                            │    │
│  │ 7. Multi-Factor Scoring                                    │    │
│  │ 8. Neural Rerank (BGE-reranker-v2-m3)                     │    │
│  │ 9. Dedup (cosine + text + source-aware)                   │    │
│  │ 10. Source Tiering (Gold/Silver/Bronze)                    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─── Admission Pipeline ───────────────────────────────────┐    │
│  │ 1. Garbage Vector Detection (magnitude check)             │    │
│  │ 2. A-MAC Quality Gate (R/N/S scoring)                     │    │
│  │ 3. Dedup Check (cosine > 0.92)                            │    │
│  │ 4. Contradiction Check (NEW)                              │    │
│  │ 5. Schema Versioning (embed model tag)                    │    │
│  │ 6. Entity Extraction → FalkorDB                           │    │
│  │ 7. Qdrant Upsert                                          │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
    ┌──────────┐     ┌──────────────┐     ┌────────────┐
    │  Qdrant  │     │  FalkorDB    │     │   Ollama   │
    │  127K+   │     │  240K nodes  │     │  Embeddings│
    │  vectors │     │  535K edges  │     │  port 11434│
    └──────────┘     └──────────────┘     └────────────┘

  ┌─── Background Jobs (cron) ────────────────────────────────┐
  │ • fact_extractor.py — every 4h, mine sessions for facts   │
  │ • enrich_second_brain.py — overnight, importance scoring  │
  │ • importance_recalculator.py — NEW, daily                 │
  │ • embedding_health.py — NEW, weekly, detect model drift   │
  └───────────────────────────────────────────────────────────┘

  ┌─── Config (single source of truth) ───────────────────────┐
  │ config/rasputin.toml                                       │
  │ ├── [server] port, host, auth_token                        │
  │ ├── [qdrant] url, collection, threshold                    │
  │ ├── [graph] host, port, graph_name, disabled               │
  │ ├── [embeddings] url, model, prefix_query, prefix_doc      │
  │ ├── [reranker] url, timeout, enabled                       │
  │ ├── [amac] threshold, model, url, timeout                  │
  │ ├── [scoring] source_weights, decay_half_lives             │
  │ ├── [entities] known_entities_path, entity_graph_path      │
  │ └── [hook] auto_recall, proactive, honcho_enabled          │
  └───────────────────────────────────────────────────────────┘
```

### Module-by-Module Spec (v3.0)

#### 1. `rasputin_server.py` (replaces hybrid_brain.py + memory_engine.py)
- **Responsibility:** HTTP API server, orchestrates all search/commit pipelines
- **API:** /search, /commit, /proactive, /feedback, /contradictions, /health, /stats, /amac/metrics
- **Internal:** Imports pipeline stages as modules, not inline functions
- **Tests:** Unit test each endpoint. Integration test full pipeline with test Qdrant collection.

#### 2. `pipeline/search.py`
- **Responsibility:** Unified search pipeline (10 stages listed above)
- **API:** `search(query, limit, config) → SearchResult`
- **Tests:** Test each stage independently. Benchmark p50/p95 latency.

#### 3. `pipeline/commit.py`
- **Responsibility:** Admission pipeline (7 stages listed above)
- **API:** `commit(text, source, importance, metadata, config) → CommitResult`
- **Tests:** Test A-MAC gate, dedup, contradiction detection independently.

#### 4. `pipeline/embedding.py`
- **Responsibility:** Embedding generation with retries, batch support, version tracking
- **API:** `embed(texts, prefix, config) → list[Vector]`, `embed_one(text, prefix) → Vector`
- **Tests:** Mock Ollama, test retry logic, test batch vs single.

#### 5. `pipeline/ranking.py`
- **Responsibility:** BM25, temporal decay, multi-factor scoring, neural rerank
- **API:** `rank(query, results, config) → list[RankedResult]`
- **Tests:** Unit test each scoring function with known inputs.

#### 6. `pipeline/entities.py`
- **Responsibility:** Unified entity extraction (merges both current implementations)
- **API:** `extract(text, config) → list[Entity]`
- **Tests:** Test against known entity corpus.

#### 7. `pipeline/graph.py`
- **Responsibility:** FalkorDB read/write, entity traversal
- **API:** `graph_search(entities, hops) → list[GraphResult]`, `graph_write(point_id, text, entities)`
- **Tests:** Integration test with test graph.

#### 8. `config.py`
- **Responsibility:** Load config from `config/rasputin.toml`, env var overrides
- **API:** `load_config(path) → Config`
- **Tests:** Test file loading, env override, defaults.

#### 9. `hooks/openclaw-mem/handler.js` (refactored)
- **Responsibility:** Thin event router only. Delegates to focused modules.
- **Sub-modules:** auto-recall.js, session-manager.js, commit-bridge.js
- **Fix:** Replace template literal URLs with proper env var reading.

### Migration Plan (Current → v3.0)

**Principle: Never break the live system. 127K vectors must remain queryable throughout.**

1. **v2.1 (Week 1-2): Schema + Config**
   - Add `embedding_model`, `embedding_version`, `schema_version` fields to new commits
   - Create `config/rasputin.toml` — hybrid_brain reads from it but falls back to env vars
   - Backfill existing vectors with `{"embedding_model": "nomic-embed-text", "schema_version": "2.0"}`
   - Fix handler.js URL templates (env var reads instead of bash-style expansion)
   - Delete dead code: hybrid_brain_v2_tenant.py, memory_consolidate.py, smart_memory_query.py, graph_query.py, migrate_to_graph.py

2. **v2.2 (Week 3-4): Unified Pipeline**
   - Extract search pipeline stages from hybrid_brain.py into `pipeline/` modules
   - Merge memory_engine.py's query expansion and source tiering into the pipeline
   - hybrid_brain.py becomes a thin HTTP server importing pipeline modules
   - Delete memory_engine.py (replace with CLI that calls /search API)
   - Write unit tests for each pipeline stage

3. **v2.3 (Week 5-6): Hook Refactor + Proactive Context**
   - Refactor handler.js into focused modules
   - Implement proper auto-recall using the /proactive endpoint
   - Remove duplicate auto-recall code
   - Kill or fix Honcho integration based on 2-week evaluation

4. **v2.4 (Week 7-8): New Features**
   - Contradiction detection on commit
   - Importance recalculation cron
   - Relevance feedback endpoint
   - Memory lifecycle management (archive, soft-delete)

5. **v3.0 (Week 9-10): Polish + Benchmarks**
   - Full test suite (unit + integration + e2e)
   - Performance benchmarks against v2.0 baseline
   - Kill remaining dead code (BrainBox, storm-wiki, predictive-memory)
   - Documentation overhaul
   - Public release candidate

### Test Strategy

| Level | What | How | When |
|-------|------|-----|------|
| **Unit** | Each pipeline stage independently | pytest, mocked dependencies | Every PR |
| **Integration** | Full search/commit against test Qdrant collection | Docker-based test env, separate collection | Every PR |
| **E2E** | OpenClaw hook → Memory API → Qdrant → recall | Simulated session with known facts, verify recall | Weekly |
| **Benchmark** | p50/p95 latency, recall@5, precision@5 | Fixed query set (50 queries), gold-standard answers | Per release |
| **Regression** | Known bugs don't recur | Specific test cases per bug fix | Every PR |
| **Load** | Concurrent search/commit under load | locust or similar, 50 concurrent users | Per release |

**Benchmark Suite (recall quality):**
- 50 hand-crafted queries with known correct answers from the 127K vector corpus
- Metrics: Recall@5, Recall@10, MRR (Mean Reciprocal Rank), latency p50/p95
- Compare: v2.0 baseline vs each v2.x milestone vs Mem0/Zep equivalent

### Version Roadmap

| Version | Timeline | Key Deliverable | Success Criteria |
|---------|----------|----------------|------------------|
| **v2.1** | Week 1-2 | Schema versioning + config consolidation + dead code removal | All new commits tagged with embedding model. Single config file. 30% fewer files. |
| **v2.2** | Week 3-4 | Unified search pipeline + tests | One search path (not two). 80% unit test coverage on pipeline. Latency ≤ v2.0. |
| **v2.3** | Week 5-6 | Hook refactor + proactive context | Auto-recall works reliably. handler.js < 300 lines. |
| **v2.4** | Week 7-8 | Contradiction detection + importance recalc + feedback | Contradictions flagged on commit. Importance updated daily. |
| **v3.0** | Week 9-10 | Full test suite + benchmarks + polish | Recall@5 ≥ 15% improvement over v2.0. Latency ≤ 200ms p95. Zero dead code. |

### Success Metrics

1. **Recall@5 improvement** — v3.0 should return the correct memory in top-5 results ≥15% more often than v2.0 (measured against 50-query benchmark set)
2. **Latency** — Search p95 ≤ 200ms (current ~150-200ms, maintain or improve)
3. **Code reduction** — Remove ≥3000 lines of dead/duplicate code (from ~15K to ~12K or less)
4. **Test coverage** — ≥80% on pipeline modules, ≥50% overall
5. **Configuration sources** — From 9+ to 1 (config file + env overrides)
6. **Context injection reliability** — Auto-recall succeeds ≥99% of sessions (currently broken due to URL templates)
7. **Contradiction detection** — Catches ≥80% of directly contradicting memories on commit
8. **vs Mem0** — Feature parity on: search, commit, contradiction detection, importance scoring. Superior on: graph traversal, A-MAC quality gating, temporal decay, multi-factor scoring.
9. **vs Zep** — Feature parity on: session memory, entity extraction. Superior on: proactive surfacing, Ebbinghaus decay, BM25 hybrid search.

---

## PHASE 4: Audit Report Integration

No specialist audit reports found in `audits/` at time of writing. This review is the primary deliverable. When the 6 specialist reports arrive, their findings should be cross-referenced against the keep/kill/build lists above and any new issues incorporated into the v2.1 sprint.

---

## Top 5 Most Critical Findings

1. **🔴 Two divergent search implementations** (hybrid_brain.py vs memory_engine.py) — each has features the other lacks. This is the #1 source of bugs and wasted effort. Unify immediately.

2. **🔴 Broken URL templates in handler.js** — JavaScript template literals with bash-style `${VAR:-default}` syntax are literal strings, not variable expansion. Auto-recall and Honcho integration are silently failing in production.

3. **🔴 No schema versioning on 127K vectors** — Switching embedding models will silently corrupt search quality. Every vector needs an `embedding_model` tag.

4. **🟠 9+ configuration sources** — Constants scattered across Python files, JSON configs, env vars, and broken JS templates. One wrong env var and the system degrades silently.

5. **🟠 Zero functional tests** — Smoke tests only verify files compile. No test verifies that search returns correct results. For a 127K-vector production system, this is high-risk.

---

*Generated by RASPUTIN architecture audit, 2026-03-30*
