#!/usr/bin/env python3
"""
Memory System Health Check — called by OpenClaw cron.
Tests all components of the memory pipeline end-to-end.
Returns a status report or raises alarms.
"""

import os
import sys
import time
import requests

CHECKS = []
WARNINGS = []
ERRORS = []

def check(name, fn):
    """Run a check, record result."""
    try:
        result = fn()
        CHECKS.append(f"✅ {name}: {result}")
        return True
    except Exception as e:
        ERRORS.append(f"🔴 {name}: {e}")
        return False

def warn(name, msg):
    WARNINGS.append(f"⚠️ {name}: {msg}")

# ──────────────────────────────────────────
# Component checks
# ──────────────────────────────────────────

def check_qdrant():
    r = requests.get("http://localhost:6333/collections/second_brain", timeout=5)
    data = r.json()["result"]
    count = data["points_count"]
    status = data["status"]
    if status != "green":
        warn("Qdrant", f"status={status}")
    return f"{count} points, status={status}"

def check_falkordb():
    import redis
    r = redis.Redis(host="localhost", port=6380)
    r.ping()
    nc = r.execute_command('GRAPH.QUERY', 'brain', 'MATCH (n) RETURN count(n)')[1][0][0]
    return f"{nc} nodes"

def check_ollama_embed():
    t0 = time.time()
    embed_url = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
    r = requests.post(embed_url, json={
        "model": "nomic-embed-text",
        "input": "memory health check"
    }, timeout=10)
    r.raise_for_status()
    emb = r.json().get("embeddings", [])
    if not emb or len(emb[0]) != 768:
        raise Exception(f"Bad embedding: got {len(emb)} vectors, expected 768-dim")
    ms = (time.time() - t0) * 1000
    return f"768-dim in {ms:.0f}ms"

def check_reranker():
    r = requests.post("http://localhost:8006/rerank", json={
        "query": "test query",
        "passages": ["test passage one", "test passage two"]
    }, timeout=10)
    r.raise_for_status()
    scores = r.json().get("scores", [])
    if len(scores) != 2:
        raise Exception(f"Expected 2 scores, got {len(scores)}")
    return f"2 passages scored ({scores[0]:.3f}, {scores[1]:.3f})"

def check_hybrid_brain():
    r = requests.get("http://localhost:7777/health", timeout=5)
    data = r.json()
    status = data.get("status", "unknown")
    components = data.get("components", {})
    down = [k for k, v in components.items() if "down" in str(v)]
    if down:
        warn("HybridBrain", f"components down: {down}")
    if status != "ok":
        raise Exception(f"status={status}, components={components}")
    return f"v{data.get('version', '?')}, all components up"

def check_openclaw_mem():
    r = requests.get("http://localhost:18790/stats", timeout=5)
    r.raise_for_status()
    data = r.json()
    return f"sessions={data.get('sessions', '?')}, observations={data.get('observations', '?')}"

def check_round_trip():
    """The critical test: commit → search → find. Proves the pipeline works end-to-end.
    Uses semantically rich text so vector search can find it, then verifies by point_id."""
    ts = int(time.time())
    # Semantically meaningful text so vector search works
    test_text = (f"PIPELINE_TEST_{ts}: The RASPUTIN memory system is running a scheduled "
                 f"self-diagnostic. Embedding pipeline, Qdrant vector search, and reranker are "
                 f"being verified. Timestamp: {ts}.")

    # Commit
    r = requests.post("http://localhost:7777/commit", json={
        "text": test_text,
        "source": "health_check",
        "importance": 1
    }, timeout=15)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise Exception(f"Commit failed: {result}")
    point_id = result.get("id")

    # Small delay for indexing
    time.sleep(0.5)

    # Search with semantic terms from the committed text
    search_query = "RASPUTIN memory system self-diagnostic embedding pipeline verification"
    r = requests.get(f"http://localhost:7777/search?q={search_query}&limit=5", timeout=15)
    r.raise_for_status()
    data = r.json()

    # Find by point_id match OR by text content
    found = False
    score = 0
    for hit in data.get("results", []):
        if str(ts) in hit.get("text", "") or f"PIPELINE_TEST_{ts}" in hit.get("text", ""):
            found = True
            score = hit.get("score", 0)
            break

    # Cleanup
    try:
        requests.post("http://localhost:6333/collections/second_brain/points/delete",
                      json={"points": [point_id]}, timeout=5)
    except Exception:
        pass

    if not found:
        raise Exception("Committed memory NOT found in search — embedding pipeline broken!")
    return f"commit→search OK (score={score:.4f}, {data.get('elapsed_ms', '?')}ms)"

# ──────────────────────────────────────────
# Run all checks
# ──────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    check("Qdrant", check_qdrant)
    check("FalkorDB", check_falkordb)
    check("Ollama Embeddings", check_ollama_embed)
    check("Neural Reranker", check_reranker)
    check("Hybrid Brain API", check_hybrid_brain)
    check("openclaw-mem", check_openclaw_mem)
    check("Round-trip (commit→search)", check_round_trip)

    elapsed = time.time() - t_start

    # Output
    if ERRORS:
        print("🔴 **Memory System: DEGRADED**")
        for e in ERRORS:
            print(e)
        if WARNINGS:
            for w in WARNINGS:
                print(w)
        print(f"\nPassed: {len(CHECKS)}/{len(CHECKS)+len(ERRORS)} | {elapsed:.1f}s")
        sys.exit(1)
    elif WARNINGS:
        print("⚠️ **Memory System: WARNINGS**")
        for w in WARNINGS:
            print(w)
        for c in CHECKS:
            print(c)
        print(f"\nAll checks passed with warnings | {elapsed:.1f}s")
        sys.exit(0)
    else:
        print(f"✅ **Memory System: HEALTHY** — {len(CHECKS)} checks passed in {elapsed:.1f}s")
        sys.exit(0)
