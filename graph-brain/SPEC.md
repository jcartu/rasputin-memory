# Graph Brain — FalkorDB Layer for Second Brain

## Overview
Add a graph layer (FalkorDB on localhost:6380) alongside existing Qdrant vector search (${MEMORY_API_HOST:-localhost:7777}).
This enables multi-hop traversal, explicit relationships, and structured entity queries.

## Schema

### Node Types
- **Memory** — A chunk from the second brain. Props: `id` (str, Qdrant point ID), `text` (str, truncated to 500 chars), `source` (str), `date` (str, YYYY-MM-DD), `category` (str)
- **Person** — A named person. Props: `name` (str, canonical), `aliases` (list of str)
- **Organization** — A company/org. Props: `name` (str), `type` (str: company/regulator/competitor/partner)
- **Project** — A project or initiative. Props: `name` (str), `status` (str: active/completed/blocked)
- **Topic** — A recurring theme. Props: `name` (str, normalized lowercase)
- **Location** — A place. Props: `name` (str), `country` (str)

### Relationship Types
- `MENTIONS` — Memory → Person/Organization/Project/Location (entity appears in text)
- `ABOUT` — Memory → Topic (thematic classification)
- `RELATED_TO` — any → any (general association, with `context` prop)
- `WORKS_AT` — Person → Organization
- `LOCATED_IN` — Organization/Person → Location
- `PART_OF` — Project → Organization
- `OCCURRED_ON` — Memory → Date node (for temporal traversal)

### Indexes
- Memory: index on `id`, `date`
- Person: index on `name`
- Organization: index on `name`
- Topic: index on `name`
- Full-text index on Memory.text

## Architecture

### Entity Extraction (NER)
Use LOCAL Ollama Qwen 72B (${OLLAMA_URL:-localhost:11434}, model `qwen2.5:72b`) for entity extraction.
Send each memory chunk with a structured prompt asking for JSON output:
```json
{
  "persons": ["Alice Smith", "Bob"],
  "organizations": ["ExampleOrg", "ExampleCo B.V."],
  "projects": ["Pipeline UI", "Grok Social Intel"],
  "topics": ["igaming", "proxy", "infrastructure"],
  "locations": ["city_A", "city_B"]
}
```

### Migration Script: `migrate_to_graph.py`
1. Connect to Qdrant (localhost:6333, collection "memories") to scroll through all points
2. For each batch of chunks (batch_size=50):
   a. Send to Qwen 72B for entity extraction
   b. Create/merge nodes in FalkorDB
   c. Create relationships
   d. Log progress every 100 chunks
3. Resume support: track last processed point ID in a state file
4. Rate limit: respect Ollama's throughput (~2-3 req/s for 72B)
5. Estimated runtime: 761K chunks / ~2 chunks per second = ~105 hours
   - With batching (send 5 chunks per prompt): ~21 hours

### Query Tool: `graph_query.py`
CLI tool for querying the graph:
```bash
python3 graph_query.py "who is connected to ExampleOrg?"
python3 graph_query.py --cypher "MATCH (p:Person)-[:WORKS_AT]->(o:Organization) RETURN p.name, o.name LIMIT 10"
python3 graph_query.py --entity "Alice Smith" --hops 2
python3 graph_query.py --topic "igaming" --since "2026-02-01"
```

### HTTP API: `graph_api.py`  
FastAPI server on port 7778:
- `GET /search?q=<query>&limit=5` — hybrid: vector search (Qdrant) + graph expansion (FalkorDB)
- `GET /entity/<name>` — get entity node + all relationships
- `GET /path?from=<entity>&to=<entity>` — shortest path between entities
- `GET /related?entity=<name>&hops=2` — multi-hop neighborhood
- `POST /cypher` — raw Cypher query

## Files to Create
All in `${WORKSPACE_DIR:-./}/tools/graph-brain/`:
1. `schema.py` — Schema creation (indexes, constraints)
2. `migrate_to_graph.py` — Migration script (Qdrant → FalkorDB via Qwen NER)
3. `graph_query.py` — CLI query tool
4. `graph_api.py` — FastAPI HTTP server
5. `requirements.txt` — Dependencies

## Technical Details
- FalkorDB: localhost:6380 (Docker, data at ${WORKSPACE_DIR:-./}/falkordb-data)
- Qdrant: localhost:6333, collection "memories" 
- Ollama: ${OLLAMA_URL:-localhost:11434}, model "qwen2.5:72b"
- Second Brain API: ${MEMORY_API_HOST:-localhost:7777} (existing, don't modify)
- Python 3.x, use existing system packages where possible

## Important
- This is ADDITIVE — don't modify the existing Qdrant/second-brain setup
- The graph layer runs alongside, not replacing, vector search
- Migration can be interrupted and resumed safely
- Use MERGE (not CREATE) for entities to avoid duplicates
