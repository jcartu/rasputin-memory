#!/usr/bin/env python3
"""
GraphRAG API — FalkorDB graph query service for RASPUTIN's Second Brain.

Endpoints:
  GET /health              — Health check
  GET /search?q=&limit=5   — Entity search with multi-hop expansion
  GET /expand?entity=&hops=2 — Expand from a named entity
  GET /related?memory_id=  — Find entities related to a memory

Connects to FalkorDB (configurable via FALKOR_HOST/FALKOR_PORT env vars).
"""

import json as _json
import os
import time
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Query, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import uvicorn

FALKOR_HOST = os.environ.get("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.environ.get("FALKOR_PORT", 6380))
GRAPH_NAME = "brain"
MEMORY_API_TOKEN = os.environ.get("MEMORY_API_TOKEN", "")

_redis = None


def get_redis():
    global _redis
    if _redis is None:
        _redis = redis.Redis(host=FALKOR_HOST, port=FALKOR_PORT, decode_responses=False)
    return _redis


def gq(cypher: str, params: dict | None = None):
    """Execute a graph query with optional parameters, return list of rows (decoded)."""
    r = get_redis()
    args = ["GRAPH.QUERY", GRAPH_NAME, cypher]
    if params:
        args += ["--params", _json.dumps(params)]
    raw = r.execute_command(*args)
    rows = raw[1] if len(raw) > 1 else []
    decoded = []
    for row in rows:
        decoded.append([
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in row
        ])
    return decoded


ENTITY_LABELS = ["Person", "Organization", "Project", "Topic", "Location"]


def fuzzy_find_entities(query: str, limit: int = 10):
    """Find entities whose name contains the query (case-insensitive)."""
    entities = []
    seen = set()
    for label in ENTITY_LABELS:
        try:
            rows = gq(
                f"MATCH (n:{label}) "
                "WHERE toLower(n.name) CONTAINS toLower($query) "
                f"RETURN n.name, '{label}', id(n) "
                "LIMIT $lim",
                params={"query": query, "lim": limit},
            )
            for row in rows:
                name = row[0]
                if name not in seen:
                    seen.add(name)
                    entities.append({"name": name, "type": row[1], "node_id": row[2]})
        except Exception:
            continue
    return entities[:limit]


def expand_entity(entity_name: str, hops: int = 2, limit: int = 30):
    """Expand from an entity through shared memories to find co-occurring entities.
    Schema: Entity -[MENTIONS]-> Memory <-[MENTIONS]- OtherEntity
    So 2-hop traversal goes through Memory nodes to find related entities."""
    connections = []
    seen = set()

    for label in ENTITY_LABELS:
        try:
            rows = gq(
                f"MATCH (start:{label} {{name: $name}})-[:MENTIONS|MENTIONED_IN]-(m:Memory) "
                "WITH m LIMIT 100 "
                "MATCH (m)-[:MENTIONS|MENTIONED_IN]-(other) "
                "WHERE other.name IS NOT NULL AND other.name <> $name "
                "RETURN other.name, labels(other)[0], count(DISTINCT m) AS shared_memories "
                "ORDER BY shared_memories DESC "
                "LIMIT $lim",
                params={"name": entity_name, "lim": limit},
            )
            for row in rows:
                key = row[0]
                if key not in seen:
                    seen.add(key)
                    connections.append({
                        "target": row[0],
                        "type": row[1],
                        "relationship": "co_mentioned",
                        "shared_memories": row[2],
                    })
            if connections:
                break  # Found the entity under this label
        except Exception:
            continue

    # Also include direct non-Memory connections (WORKS_AT, LOCATED_IN, etc.)
    for label in ENTITY_LABELS:
        try:
            rows = gq(
                f"MATCH (start:{label} {{name: $name}})-[r]-(other) "
                "WHERE other.name IS NOT NULL AND labels(other)[0] <> 'Memory' "
                "RETURN other.name, labels(other)[0], type(r) "
                "LIMIT $lim",
                params={"name": entity_name, "lim": limit},
            )
            for row in rows:
                key = row[0]
                if key not in seen:
                    seen.add(key)
                    connections.append({
                        "target": row[0],
                        "type": row[1],
                        "relationship": row[2],
                    })
        except Exception:
            continue

    return sorted(connections, key=lambda x: x.get("shared_memories", 0), reverse=True)[:limit]


def expand_by_node_id(node_id: int, hops: int = 2, limit: int = 30):
    """Expand from a node by its internal ID."""
    hops = min(int(hops), 5)
    connections = []
    seen = set()
    try:
        rows = gq(
            f"MATCH (start) WHERE id(start) = $nid "
            f"MATCH (start)-[r*1..{hops}]-(connected) "
            "WHERE connected.name IS NOT NULL "
            "UNWIND r AS rel "
            "WITH DISTINCT connected, rel "
            "RETURN connected.name, labels(connected)[0], type(rel) "
            "LIMIT $lim",
            params={"nid": node_id, "lim": limit},
        )
        for row in rows:
            key = f"{row[0]}:{row[2]}"
            if key not in seen:
                seen.add(key)
                connections.append({
                    "target": row[0],
                    "type": row[1],
                    "relationship": row[2],
                })
    except Exception:
        pass
    return connections[:limit]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify FalkorDB connection
    try:
        r = get_redis()
        r.ping()
        print(f"[GraphAPI] Connected to FalkorDB on {FALKOR_HOST}:{FALKOR_PORT}", flush=True)
    except Exception as e:
        print(f"[GraphAPI] WARNING: FalkorDB not reachable: {e}", flush=True)
    yield


app = FastAPI(title="GraphRAG API", lifespan=lifespan)


def _verify_token(request: Request):
    """Check MEMORY_API_TOKEN if set. For local dev without token, all requests pass."""
    if not MEMORY_API_TOKEN:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {MEMORY_API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health(_=Depends(_verify_token)):
    try:
        r = get_redis()
        r.ping()
        rows = gq("MATCH (n) RETURN count(n)")
        node_count = rows[0][0] if rows else -1
        return {"status": "ok", "graph": GRAPH_NAME, "nodes": node_count}
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=503)


@app.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(5, ge=1, le=50), _=Depends(_verify_token)):
    """Find entities matching query, expand 1-2 hops, return structured JSON."""
    t0 = time.time()
    matched = fuzzy_find_entities(q, limit=limit)

    entities_out = []
    for ent in matched:
        conns = expand_entity(ent["name"], hops=2, limit=20)
        entities_out.append({
            "name": ent["name"],
            "type": ent["type"],
            "connections": conns,
        })

    return {
        "query": q,
        "entities": entities_out,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/expand")
def expand(entity: str = Query(..., min_length=1), hops: int = Query(2, ge=1, le=4), _=Depends(_verify_token)):
    """Multi-hop relationship expansion from a named entity."""
    t0 = time.time()
    conns = expand_entity(entity, hops=hops, limit=50)
    return {
        "entity": entity,
        "hops": hops,
        "connections": conns,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/related")
def related(memory_id: int = Query(...), _=Depends(_verify_token)):
    """Find entities related to a given memory by Qdrant point ID."""
    t0 = time.time()
    entities = []
    try:
        rows = gq(
            "MATCH (m:Memory {id: $mid})-[r]-(e) "
            "WHERE e.name IS NOT NULL "
            "RETURN e.name, labels(e)[0], type(r) "
            "LIMIT 30",
            params={"mid": memory_id},
        )
        for row in rows:
            entities.append({"name": row[0], "type": row[1], "relationship": row[2]})
    except Exception:
        pass

    # Also check MENTIONED_IN (reverse direction from real-time commits)
    try:
        rows = gq(
            "MATCH (e)-[:MENTIONED_IN]->(m:Memory {id: $mid}) "
            "WHERE e.name IS NOT NULL "
            "RETURN e.name, labels(e)[0], 'MENTIONED_IN' "
            "LIMIT 30",
            params={"mid": memory_id},
        )
        seen = {e["name"] for e in entities}
        for row in rows:
            if row[0] not in seen:
                entities.append({"name": row[0], "type": row[1], "relationship": row[2]})
    except Exception:
        pass

    return {
        "memory_id": memory_id,
        "entities": entities,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7778, log_level="info")
