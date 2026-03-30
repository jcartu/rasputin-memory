# RASPUTIN Memory Architecture

Deep dive on the 7-layer hybrid retrieval pipeline, data flow, and design decisions.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      RASPUTIN Memory System                     │
└─────────────────────────────────────────────────────────────────┘

LAYER 1: HOT CONTEXT (always in prompt)
├── MEMORY.md — identity, tools, architecture
├── AGENTS.md — behavioral rules
└── SOUL.md — personality, voice, style

LAYER 2: SEMANTIC SEARCH (on-demand retrieval)
├── Qdrant (port 6333) — 134K+ vectors, 768d nomic-embed-text v1
├── BM25 sparse keyword search (bm25_search.py)
└── Reranker (port 8006) — BAAI/bge-reranker-v2-m3 on GPU

LAYER 3: KNOWLEDGE GRAPH (relationship reasoning)
├── FalkorDB (port 6380, Docker) — 240K nodes, 535K edges
├── Graph: "brain" — entities, relationships, temporal
└── Entity extraction: fast NER on every commit

LAYER 4: HYBRID SEARCH ENGINE (orchestration)
└── hybrid_brain.py (PM2, port 7777)
    ├── /search — vector + graph + BM25 + neural rerank
    ├── /commit — embed + A-MAC + store + entity extract + graph
    ├── /graph — direct Cypher queries
    ├── /stats — counts + health
    └── /health — component health check

LAYER 5: OBSERVATIONAL MEMORY (fast local context)
├── om_observations.md — compressed recent context (24h TTL)
└── entity_graph.json — lightweight in-process entity cache

LAYER 6: LLM ENRICHMENT (quality gate + extraction)
├── A-MAC quality gate — Qwen 35B scores every commit
├── Entity extraction — fast NER + graph writes
└── fact_extractor.py — 4h cron, mines sessions for facts

LAYER 7: CONTINUOUS MAINTENANCE
├── fact_extractor.py — every 4h, personal knowledge extraction
├── Second brain enrichment — 6x overnight, importance scoring
└── Graph deepening — daily, FalkorDB relationship extraction
```

---

## Data Flow: Search Request

When you call `GET /search?q=query&limit=10`:

```
1. Query arrives at hybrid_brain.py HTTP server
2. Query → Ollama nomic-embed-text → 768d vector (port 11434)
3. Vector → Qdrant ANN search (HNSW, top 20 candidates, threshold 0.50)
4. Query → FalkorDB Cypher match (entity lookup, 2-hop traversal, port 6380)
5. Qdrant + FalkorDB results merged via RRF (Reciprocal Rank Fusion)
6. BM25 sparse keyword scoring applied (bm25_search.py)
7. Ebbinghaus temporal decay applied (power-law, importance-scaled)
8. Neural reranker re-scores top candidates (bge-reranker-v2-m3, port 8006)
9. Multi-factor composite score: similarity × importance × recency × source_reliability
10. Top K results returned with graph context + entity enrichment
```

Total latency: ~150-200ms p95 (mostly embed + rerank)

---

## Data Flow: Commit Request

When you call `POST /commit {"text": "...", "source": "conversation"}`:

```
1. A-MAC Quality Gate: text → Qwen 35B (port 11436)
   - Scores Relevance, Novelty, Specificity (0–10 each)
   - Composite = mean of all three
   - Composite < 4.0 → REJECTED (logged to /tmp/amac_rejected.log)
   - Timeout (30s) → FAIL-OPEN (accepted anyway)

2. Deduplication check:
   - Embed the text (Ollama 11434)
   - Search Qdrant for cosine > 0.92
   - If duplicate found AND text overlap > 50% → UPDATE existing point
   - Otherwise → create new point

3. Store in Qdrant:
   - point_id = abs(hash(text + timestamp)) % 2^63
   - payload: {text, source, date, importance, auto_committed: true}
   - vector: 768d nomic-embed-text embedding

4. Entity extraction (fast NER):
   - Regex-based pattern matching for people, orgs, money amounts, dates
   - Named entity list from text

5. FalkorDB graph write:
   - MERGE Person/Entity nodes
   - CREATE MENTIONED_IN relationships
   - Link entities to memory point

6. Return: {point_id, entities, graph_written, amac_scores}
```

---

## Multi-Angle Query Expansion

`memory_engine.py` expands every query into up to 12 search angles before embedding:

1. **Original query** — baseline
2. **Proper noun extraction** — each capitalized entity searched separately
3. **Entity graph lookup** — augment entity queries with graph context
4. **Topic rephrasing** — strip question words, search the clean topic
5. **Source-specific** — if query mentions "email" or "searched", add source filters
6. **Temporal expansion** — "last week" queries get a recency boost
7. **Semantic expansions** — known domain mappings (project planning→planning, product→deposits)

All 12 queries are batch-embedded in a single Ollama API call (one HTTP request, all embeddings in parallel). Typical cost: ~55ms for 8 embeddings.

---

## Scoring Architecture

Results are scored by a multi-factor composite:

```python
composite_score = (
    semantic_similarity      # cosine distance from Qdrant
    × importance_weight      # metadata importance field (0–100)
    × temporal_decay         # Ebbinghaus power-law decay
    × source_reliability     # ChatGPT > Perplexity > email > other
    × retrieval_frequency    # how often this memory has been recalled
)
```

After initial scoring, the **neural reranker** (`bge-reranker-v2-m3`) applies cross-encoder scoring to re-rank the top-50 candidates. Cross-encoder models read query + passage together, producing far more accurate relevance scores than bi-encoder cosine similarity alone.

---

## Source Tier System

Not all memories are equal. The system prioritizes:

**Tier 1 (GOLD):** ChatGPT conversations, Perplexity searches, direct conversations, social intel
**Tier 2 (SILVER):** Email (multiplied by 0.85 to reduce noise)
**Tier 3 (BRONZE):** All other sources (telegram, whatsapp, auto-commits)

The two-tier search strategy first fills high-value sources, only backfilling with email if fewer than 50 candidates are found.

---

## Deduplication

The deduplication system prevents showing the same email thread 5 times:

- **Email/Gmail**: dedup by `thread_id` or `gmail_id`
- **ChatGPT**: dedup by title + MD5 of first 200 chars
- **Perplexity**: dedup by filename or question text
- **Other**: dedup by Qdrant point ID

At commit time, if cosine similarity > 0.92 AND text overlap > 50%, the existing point is updated rather than a duplicate created.

---

## Observational Memory (OM)

A fast local cache layer that doesn't require a network call:

- **File:** `memory/om_observations.md`
- **Format:** Date-grouped observation blocks
- **Lookup:** Keyword overlap scoring (pure Python, <1ms)
- **TTL:** 24 hours (stale entries still used but flagged)
- **Use case:** Recent events that happened this session or today

OM results are prepended to the main MEMORY RECALL block in the context output.

---

## Entity Graph (JSON)

A lightweight in-memory entity graph at `memory/entity_graph.json`:

```json
{
  "people": {
    "Bob": {"role": "business contact", "context": "product affiliate"},
    "Alice": {"role": "wife", "context": "project planning, locally"}
  },
  "companies": {
    "Acme Corp": {"type": "SaaS brand", "context": "Rival platform"},
    "WidgetCo": {"type": "SaaS brand", "context": "EMEA market"}
  },
  "topics": {
    "project planning": {"context": "project planning, Alice"},
    "CHRONOS": {"context": "hardware wallet project"}
  }
}
```

This JSON is read in-process during query expansion to enrich entity-based queries before they hit Qdrant.

---

## FalkorDB Knowledge Graph

FalkorDB is a Redis-compatible graph database using the Cypher query language.

**Schema:**
- `(:Entity {name: "...", type: "person|org|concept|location|money|date"})` — nodes
- `[:MENTIONED_IN {memory_id: "...", timestamp: "..."}]` — memory links
- `[:RELATED_TO {strength: 0.8}]` — entity-to-entity relationships

**Graph search logic:**
1. Extract candidate entity names from query (NER)
2. `MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($term)` 
3. Traverse 2 hops: `MATCH (e)-[*1..2]-(related)`
4. Collect memory IDs from `MENTIONED_IN` edges
5. Fetch those points from Qdrant

---

## Design Decisions

**Why Qdrant?** Fast HNSW ANN search, excellent Docker deployment, strong filtering API, good Python client.

**Why FalkorDB?** Redis-protocol compatible (any Redis client works), persistent graph with Cypher, lightweight Docker footprint.

**Why nomic-embed-text v1?** 768 dimensions is the sweet spot — large enough for nuanced similarity, small enough for fast cosine search. v1 was the model used when the collection was first built; all 134K+ vectors use it. Switching models would require re-embedding everything.

**Why neural reranking?** Bi-encoder cosine similarity (ANN search) is fast but imprecise — it embeds query and passage independently. Cross-encoder reranking reads query+passage together, which is dramatically more accurate. Running it only on top-50 keeps latency acceptable.

**Why A-MAC?** Without a quality gate, trivial content ("ok", "thanks", "yes") would flood the vector store and dilute search quality. A-MAC ensures only information-dense, novel, specific memories get stored.

**Why local LLMs?** Zero marginal cost. A-MAC runs 24/7 scoring every commit. At cloud API prices this would be $100s/month. With local Qwen 35B it's $0.
