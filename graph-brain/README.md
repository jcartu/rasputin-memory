# Graph Brain

Knowledge graph layer for the memory system. Stores entities, relationships, and enables graph-based queries alongside vector search.

## Architecture

Uses Neo4j as the graph database. Memories are stored as both vector embeddings (Qdrant) and graph nodes (Neo4j), enabling hybrid retrieval.

## Files

| File | Description |
|------|-------------|
| `schema.py` | Graph schema definitions — node types, relationship types, constraints |
| `graph_api.py` | REST API for graph operations — CRUD on entities and relationships |
| `graph_query.py` | Graph query engine — traversal, path finding, subgraph extraction |
| `migrate_to_graph.py` | Migration tool — imports existing Qdrant memories into Neo4j |
| `SPEC.md` | Full specification document |

## Setup

```bash
# Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5

# Set environment
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# Initialize schema
python3 graph-brain/schema.py

# Migrate existing memories
python3 graph-brain/migrate_to_graph.py
```

## Integration

The graph brain complements vector search by enabling:
- **Entity resolution** — "What do I know about X?"
- **Relationship traversal** — "How are X and Y connected?"
- **Temporal queries** — "What changed about X over time?"
- **Subgraph extraction** — Pull relevant context clusters for LLM grounding
