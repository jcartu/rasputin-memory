# RASPUTIN Memory Architecture (v0.9)

Deep dive on the hybrid retrieval pipeline, data flow, and design decisions.

## Overview

```
+-------------------------------------------------------------------+
|                      RASPUTIN Memory System                       |
+-------------------------------------------------------------------+

LAYER 0: MCP INTERFACE
+-- tools/mcp/server.py (port 8808, FastMCP 3.2, streamable-http)
    +-- 6 tools: store, search, reflect, stats, feedback, commit_conversation
    +-- Thin HTTP proxy -> API server (port 7777)

LAYER 1: HYBRID SEARCH (two-lane retrieval + reranking)
+-- tools/brain/search.py
    +-- Lane 1: Qdrant windows (45 slots, 768d nomic-embed-text)
    +-- Lane 2: Qdrant facts (15 slots, 768d nomic-embed-text)
    +-- Lane 3: BM25 keywords (SQLite FTS5 in-memory, 10 slots)
    +-- RRF fusion -> Qwen3-Reranker-0.6B (GPU, 0.99/0.0001 separation)
    +-- Prompt routing (inference / factual / temporal)

LAYER 2: KNOWLEDGE GRAPH
+-- FalkorDB (port 6380, Cypher) -- graph "brain"
    +-- Entity extraction (fast NER) + entity resolution
    +-- Canonical names + alias tracking + mention counts

LAYER 3: HTTP API (tools/brain/server.py, port 7777)
+-- /search, /commit, /reflect, /graph, /stats, /health

LAYER 4: LLM ENRICHMENT
+-- A-MAC quality gate (Ollama qwen2.5:14b, local)
+-- Structured fact extraction (Pydantic-validated, Haiku/Cerebras)
+-- Reflect -- LLM synthesis (Anthropic / Ollama)
+-- Contradiction detection + supersede tracking
```

---

## Data Flow: Search Request

```
 1. Query arrives at HTTP server
 2. Multi-query expansion (known_entities.json lookup)
 3. Embed: query -> nomic-embed-text -> 768d vector
 4. Two-lane Qdrant search:
    +-- Lane 1: window chunks (45 slots, HNSW ANN)
    +-- Lane 2: fact chunks (15 slots, HNSW ANN)
 5. BM25 keyword search (FTS5 in-memory, 10 slots)
 6. Reciprocal Rank Fusion (k=60) merges dense + BM25
 7. FalkorDB graph search (entity lookup, 1-2 hop traversal)
 8. Qwen3-Reranker-0.6B re-scores candidates (cross-encoder, GPU)
    +-- final_score = ce_score * recency_boost (alpha=0.2)
 9. Contradiction resolution (superseded memories demoted 0.3x)
10. Top-60 results returned with graph context
```

Latency: ~150-200ms p95 (embed + rerank dominant)

---

## Data Flow: Commit Request

```
1. A-MAC Quality Gate: text -> LLM (qwen2.5:14b, Ollama)
   - Scores Relevance, Novelty, Specificity (0-10)
   - Composite < 4.0 -> REJECTED; timeout 30s -> FAIL-OPEN

2. Deduplication: embed text, search Qdrant cosine > 0.92
   - Duplicate found -> UPDATE existing point
   - New content -> create point (UUID-based ID)

3. Contradiction detection: top-5 similar, LLM-verified
   - New memory supersedes contradicted ones

4. Qdrant upsert: 768d vector + payload (text, source, importance,
   mentioned_names, extracted_dates, contradicts, supersedes, ...)

5. Entity extraction (fast NER):
   - Known-entity dictionary (persons, orgs, projects)
   - Capitalized name regex (English + Cyrillic)
   - Optional entity resolver (canonical names + aliases)

6. FalkorDB: MERGE entity nodes, CREATE MENTIONS edges

7. Return: {id, source, dedup, contradictions, graph}
```

---

## Data Flow: Reflect (LLM Synthesis)

```
1. POST /reflect -> hybrid_search(query, limit=20)
2. Format top 15 memories as numbered context blocks
3. LLM call: Anthropic (claude-haiku-4-5-20251001) or Ollama fallback
4. Return: {answer, sources[], search_elapsed_ms, reflect_model}
```

Produces coherent synthesized answers from retrieved memories.  Connects
dots across multiple memories, notes contradictions, favors most recent.

---

## Structured Fact Extraction

`tools/brain/fact_extractor.py` extracts facts at ingest time via LLM.
Each fact is Pydantic-validated (`ExtractedFact` model):

- `what` -- core fact, self-contained (1-2 sentences)
- `who` -- people involved, pronouns resolved to names
- `when` / `where` -- temporal and spatial context
- `fact_type` -- `"world"` | `"experience"` | `"inference"`
- `occurred_start` / `occurred_end` -- ISO dates when determinable
- `entities` -- list of `{name, type}` pairs
- `confidence` -- 0.9 explicit, 0.7 inference, 0.5 weak signal

Facts stored as Qdrant points (`chunk_type="fact"`) and retrieved via
Lane 2 in the two-lane search pipeline.

---

## BM25 + Reciprocal Rank Fusion

`tools/bm25_search.py` provides sparse keyword search complementing dense
retrieval.  In-process BM25Scorer (k1=1.5, b=0.75) with RRF fusion (k=60):

```
rrf(i) = 1/(k + rank_dense(i) + 1) + 1/(k + rank_bm25(i) + 1)
hybrid  = (1 - 0.3) * dense_score + 0.3 * bm25_normalized
```

BM25 was net negative with the L-6 cross-encoder (-14pp to -28pp) but
became +0.6pp with Qwen3-Reranker -- the stronger reranker filters BM25's
false positives while keeping true keyword matches.

---

## Scoring Architecture

Multi-stage scoring pipeline:

```
Stage 1: Dense cosine similarity (Qdrant ANN)
Stage 2: BM25 keyword scoring + RRF fusion
Stage 3: Qwen3-Reranker-0.6B cross-encoder (ce_score)
Stage 4: Recency boost: final = ce * (1 + 0.2 * (recency - 0.5))
Stage 5: Contradiction demotion (superseded * 0.3)
```

Pre-reranker multifactor (when source_weight present):
```
multiplier = 0.45 + 0.30*importance_norm + 0.15*source_weight + 0.10*retrieval_boost
```

The Qwen3 cross-encoder reads query + passage together, producing far more
accurate relevance scores than bi-encoder cosine similarity alone.

---

## FalkorDB Knowledge Graph

Redis-compatible graph database using Cypher.

**Schema:**
- `(:Memory {id, text, created_at})` -- memory nodes
- `(:Person {name, type, canonical, mention_count})` -- entity nodes
- `(:Organization ...)`, `(:Project ...)`, `(:Topic ...)`, `(:Location ...)`
- `[:MENTIONS]` -- memory-to-entity edges

**Search:** Extract entities from query -> match nodes by name ->
1-hop (direct MENTIONS) -> 2-hop (co-mentioned entities, up to 5) ->
keyword fallback on Memory.text -> entity-to-entity context edges.

---

## Deduplication

At commit time: embed text, query Qdrant for cosine > 0.92.  If a
near-duplicate is found, update the existing point in-place.  Otherwise
create a new point.  All content types use the same cosine threshold --
no source-specific dedup logic.

---

## MCP Server

`tools/mcp/server.py` -- thin HTTP proxy, never imports brain modules.

```
FastMCP 3.2, streamable-http transport, port 8808

memory_store               -> POST /commit
memory_search              -> GET  /search
memory_reflect             -> POST /reflect
memory_stats               -> GET  /stats
memory_feedback            -> POST /feedback
memory_commit_conversation -> POST /commit_conversation
```

---

## Design Decisions

**Why Qdrant?** Fast HNSW ANN, strong filtering API, good Python client.
Two-lane search uses `chunk_type` field conditions in a single collection.

**Why FalkorDB?** Redis-protocol compatible, persistent Cypher graph,
lightweight Docker.  Entity nodes track canonical names and mention counts.

**Why nomic-embed-text v1?** 768d is the sweet spot.  Qwen3 embeddings
(768d and 4096d) tested, showed 0pp or worse.  Switching requires
re-embedding.

**Why Qwen3-Reranker-0.6B?** Foundation model vs lightweight cross-encoder.
ms-marco-MiniLM-L-6-v2 (v0.7) had score distributions too narrow for
effective ranking.  Qwen3 separates relevant/irrelevant at 0.99 vs 0.0001
-- clear binary signal.  +4.5pp production, +8.6pp compare.  L-12 CE also
tested and reverted (-12.6pp single-hop).

**Why BM25 + RRF?** Dense retrieval misses exact keyword matches.  BM25
was net negative with L-6 CE (-14pp to -28pp) but +0.6pp with Qwen3 --
the stronger reranker keeps true matches, discards noise.

**Why A-MAC?** Without a quality gate, trivial content floods the store.
Uses Ollama locally for zero marginal cost.

**Why two-lane retrieval?** Windows capture narrative context, facts
capture discrete knowledge.  Merging via reranker: +6.5pp over single-lane.

---

## Benchmarks

LoCoMo full 10-conv (1986 questions, non-adversarial):

| Mode | Score |
|------|-------|
| Production (Haiku answers, strict judge) | 74.2% |
| Compare (Haiku answers, generous judge) | 77.7% |

```
nomic-embed-text (768d) -> Two-lane (45w+15f) + BM25 FTS5 (10)
  -> RRF -> Qwen3-Reranker-0.6B -> top-60 -> Haiku -> gpt-4o-mini judge
```
