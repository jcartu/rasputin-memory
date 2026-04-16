# Hybrid Brain API

The core service: `hybrid_brain.py` running on port 7777. This is the single entry point for all memory operations.

## Endpoints

### GET /search

Hybrid semantic + graph + BM25 search with neural reranking.

```bash
curl "http://localhost:7777/search?q=LATAM+campaigns&limit=5"
curl "http://localhost:7777/search?q=Alice+project+planning&limit=10&source=email"
```

**Parameters:**
- `q` (required) — search query string
- `limit` (optional, default 10) — max results to return
- `source` (optional) — filter by source type: `email`, `chatgpt`, `perplexity`, `conversation`

**Response:**
```json
{
  "results": [
    {
      "id": "abc123",
      "score": 0.87,
      "rerank_score": 0.94,
      "payload": {
        "text": "the user decided to use Acme Corp for LATAM campaigns, regional only",
        "source": "conversation",
        "date": "2026-03-15T14:22:00",
        "importance": 70
      }
    }
  ],
  "graph_context": "Acme Corp (company): SaaS brand, Rival platform",
  "total": 1,
  "query_time_ms": 187
}
```

---

### POST /commit

Store a memory with full enrichment pipeline (A-MAC → embed → dedup → store → entity extract → graph write).

```bash
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "the user decided to go with Acme Corp for LATAM campaigns, regional only",
    "source": "conversation",
    "importance": 70,
    "metadata": {
      "session": "2026-03-15",
      "tags": ["business", "latam", "acme-corp"]
    }
  }'
```

**Request body:**
- `text` (required) — the memory text (max 4000 chars stored, first 800 chars used for A-MAC)
- `source` (optional, default `"conversation"`) — source type for filtering/display
- `importance` (optional, default 60) — importance score 0–100, affects retrieval ranking
- `metadata` (optional) — additional key/value pairs merged into the Qdrant payload
- `force` (optional, default false) — bypass A-MAC quality gate

**Response (accepted):**
```json
{
  "status": "accepted",
  "point_id": 1234567890,
  "entities": ["Acme Corp", "LATAM", "regional"],
  "graph_written": true,
  "amac": {
    "relevance": 8.0,
    "novelty": 7.0,
    "specificity": 9.0,
    "composite": 8.0
  },
  "duplicate": false
}
```

**Response (rejected by A-MAC):**
```json
{
  "status": "rejected",
  "reason": "amac_quality_gate",
  "amac": {
    "relevance": 2.0,
    "novelty": 1.0,
    "specificity": 3.0,
    "composite": 2.0
  }
}
```

---

### POST /reflect

Retrieve relevant memories and synthesize a coherent answer via LLM.

```bash
curl -X POST http://localhost:7777/reflect \
  -H 'Content-Type: application/json' \
  -d '{"q": "What do we know about the auth service architecture?", "limit": 20}'
```

**Request body:**
- `q` or `query` (required) — the question to answer from memory
- `limit` (optional, default 20) — how many memories to retrieve for synthesis (1-30)
- `source` (optional) — filter memories by source type
- `collection` (optional) — override Qdrant collection

**Response:**
```json
{
  "answer": "Based on stored memories, the auth service uses PostgreSQL...",
  "sources": [
    {"point_id": 123, "text": "We chose PostgreSQL for auth...", "score": 0.92},
    {"point_id": 456, "text": "MySQL was rejected due to...", "score": 0.71}
  ],
  "search_elapsed_ms": 55.2,
  "total_elapsed_ms": 1230.5,
  "reflect_model": "claude-haiku-4-5-20251001"
}
```

The reflect endpoint uses the LLM configured in `config/rasputin.toml` `[reflect]` section (default: Anthropic Claude Haiku). Falls back to Ollama if no Anthropic API key is set.

---

### GET /health

Check health of all components.

```bash
curl http://localhost:7777/health
```

```json
{
  "status": "ok",
  "qdrant": "ok",
  "falkordb": "ok",
  "reranker": "available",
  "embeddings": "ok",
  "amac_llm": "ok"
}
```

---

### GET /stats

Get counts and metrics.

```bash
curl http://localhost:7777/stats
```

```json
{
  "qdrant_count": 134821,
  "falkordb_nodes": 240152,
  "falkordb_edges": 535891,
  "amac_stats": {
    "accepted": 12450,
    "rejected": 3201,
    "bypassed": 44,
    "timeout_accepts": 12,
    "avg_score": 6.84
  }
}
```

---

### POST /graph

Direct Cypher query against FalkorDB.

```bash
curl -X POST http://localhost:7777/graph \
  -H 'Content-Type: application/json' \
  -d '{"query": "MATCH (p:Entity {type: \"person\"}) RETURN p.name LIMIT 10"}'
```

---

## Internal Search Pipeline (hybrid_search function)

```python
def hybrid_search(query, limit=10, graph_hops=2, source_filter=None):
    # 1. Embed query
    vector = get_embedding(query)  # → Ollama 11434, nomic-embed-text

    # 2. Qdrant ANN search
    qdrant_results = qdrant_search(query, limit=20, source_filter=source_filter)
    # threshold=0.50, with_payload=True

    # 3. FalkorDB graph search
    entities = extract_entities(query)           # fast NER
    graph_results = graph_search(query, hops=2)  # Cypher MATCH + traversal
    
    # 4. Merge via RRF (Reciprocal Rank Fusion)
    merged = rrf_merge(qdrant_results, graph_results)
    
    # 5. BM25 sparse scoring
    merged = bm25_hybrid_rerank(query, merged)
    
    # 6. Temporal decay (Ebbinghaus power law)
    merged = apply_temporal_decay(merged, half_life_days=30)
    
    # 7. Multi-factor scoring
    merged = apply_multifactor_scoring(merged)
    
    # 8. Neural rerank (bge-reranker-v2-m3)
    final = neural_rerank(query, merged[:50], top_k=limit)
    
    # 9. Enrich with graph context
    final = enrich_with_graph(final)
    
    return final[:limit]
```

---

## Internal Commit Pipeline (commit_memory function)

```python
def commit_memory(text, source="conversation", importance=60, metadata=None):
    # 1. A-MAC quality gate
    allowed, reason, amac_scores = amac_gate(text, source)
    if not allowed:
        return {"status": "rejected", "reason": reason, "amac": amac_scores}
    
    # 2. Embed text
    vector = get_embedding(text)  # → Ollama 11434
    
    # 3. Deduplication check
    existing_id = check_duplicate(vector, text, threshold=0.92)
    
    if existing_id:
        # Update existing point (PATCH payload)
        update_qdrant_point(existing_id, text, metadata)
        return {"status": "updated", "point_id": existing_id}
    
    # 4. Generate point ID
    point_id = abs(hash(text + str(datetime.now()))) % (2**63)
    
    # 5. Build payload
    payload = {
        "text": text[:4000],
        "source": source,
        "date": datetime.now().isoformat(),
        "importance": importance,
        "auto_committed": True,
    }
    if metadata:
        payload.update(metadata)
    
    # 6. Store in Qdrant
    qdrant_upsert(point_id, vector, payload)
    
    # 7. Extract entities (fast NER)
    entities = extract_entities_fast(text)
    
    # 8. Write to FalkorDB graph
    graph_written = write_to_graph(point_id, text, entities, timestamp=payload["date"])
    
    return {
        "status": "accepted",
        "point_id": point_id,
        "entities": entities,
        "graph_written": graph_written,
        "amac": amac_scores,
        "duplicate": False
    }
```

---

## Configuration

All runtime configuration lives in `config/rasputin.toml`.  Environment variables override TOML values — see `tools/config.py` for the mapping.

Key sections:

```toml
[server]
port = 7777

[qdrant]
url = "http://localhost:6333"
collection = "second_brain"

[embeddings]
model = "nomic-embed-text"         # 768-dim
url = "http://localhost:11434/api/embed"

[reranker]
provider = "qwen3"                 # Qwen3-Reranker-0.6B
url = "http://192.168.1.41:9091/rerank"
timeout = 30
enabled = true

[graph]
host = "localhost"
port = 6380                        # FalkorDB
graph_name = "brain"

[amac]
threshold = 4.0                    # Minimum composite score to accept
model = "qwen2.5:14b"             # Via Ollama
timeout = 30                       # Seconds — fail-open on timeout
```

See `config/rasputin.toml` for the full reference including `[reflect]`, `[scoring]`, `[constraints]`, and `[benchmark]` sections.

---

## Running as HTTP Server

```python
# Start the server (default port 7777)
python3 hybrid_brain.py serve

# Custom port
python3 hybrid_brain.py serve 8888

# Or with any process manager (systemd, supervisord, etc.)
# See docs/SETUP.md for production deployment options
```

The server is a simple Python `http.server.BaseHTTPRequestHandler` subclass — no external web framework required.

---

## CLI Usage (hybrid_brain.py)

Beyond the HTTP server, `hybrid_brain.py` also has CLI modes:

```bash
# Direct search from command line
python3 hybrid_brain.py search "LATAM campaigns"

# Commit directly
python3 hybrid_brain.py commit "Important decision about X"

# Run self-tests
python3 hybrid_brain.py test

# Show stats
python3 hybrid_brain.py stats
```

---

## API-first usage

The project now uses `hybrid_brain.py` as the single entrypoint. Use the HTTP API directly for both search and commit workflows.

```bash
# Recall/search
curl -s "http://localhost:7777/search?q=release+timeline&limit=5"

# Commit
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text":"Decision: move launch to April","source":"conversation"}'
```
