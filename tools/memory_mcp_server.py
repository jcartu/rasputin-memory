#!/usr/bin/env python3
"""
Memory MCP Server — wraps RASPUTIN's Second Brain hybrid API (port 7777).
Exposes: search, commit, proactive surfacing, graph queries, stats.

Run: python3 tools/memory-mcp-server.py          (stdio for MCP clients)
Run: python3 tools/memory-mcp-server.py --http    (HTTP/SSE on port 8101)
"""

import json
import os
import subprocess
import argparse

from mcp.server.fastmcp import FastMCP

BRAIN_URL = "${MEMORY_API_URL:-http://${MEMORY_API_HOST:-localhost:7777}}"
FALKORDB_HOST = "localhost"
FALKORDB_PORT = 6380

mcp_server = FastMCP(os.environ.get("MCP_SERVER_NAME", "memory-server"))


def _http_json(url: str, data: dict | None = None, timeout: int = 15) -> dict | list:
    cmd = ["curl", "-sf", url, "-H", "Content-Type: application/json"]
    if data:
        cmd += ["-X", "POST", "-d", json.dumps(data)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"HTTP request failed: {r.stderr.strip() or 'non-zero exit'}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def _redis_cmd(*args: str, timeout: int = 10) -> str:
    cmd = ["redis-cli", "-h", FALKORDB_HOST, "-p", str(FALKORDB_PORT)] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


# ── Tools ────────────────────────────────────────────────────────────────


@mcp_server.tool()
def memory_search(query: str, limit: int = 5) -> str:
    """Search RASPUTIN's Second Brain using semantic search + knowledge graph + reranker.
    Returns the most relevant memories from 761K+ stored entries."""
    from urllib.parse import quote

    results = _http_json(f"{BRAIN_URL}/search?q={quote(query)}&limit={limit}")
    if not results:
        return "No results found."
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    lines = []
    for i, r in enumerate(results[:limit], 1):
        text = r.get("text", r.get("payload", {}).get("text", str(r)))[:600]
        score = r.get("score", r.get("rerank_score", "?"))
        source = r.get("source", r.get("payload", {}).get("source", ""))
        lines.append(f"{i}. [{source}] (score: {score}) {text}")
    return "\n\n".join(lines)


@mcp_server.tool()
def memory_commit(text: str, source: str = "mcp") -> str:
    """Store a new memory in RASPUTIN's Second Brain.
    Use for important facts, decisions, or context worth remembering."""
    result = _http_json(f"{BRAIN_URL}/commit", {"text": text, "source": source})
    return json.dumps(result, indent=2) if result else "Committed successfully."


@mcp_server.tool()
def memory_proactive(context: list[str]) -> str:
    """Surface non-obvious related memories based on conversation context.
    Pass recent messages/context strings to get surprising connections."""
    result = _http_json(f"{BRAIN_URL}/proactive", {"messages": context})
    if not result:
        return "No proactive memories surfaced."
    if isinstance(result, dict) and "results" in result:
        items = result["results"]
    elif isinstance(result, list):
        items = result
    else:
        return json.dumps(result, indent=2)
    lines = []
    for i, r in enumerate(items, 1):
        text = r.get("text", str(r))[:500]
        lines.append(f"{i}. {text}")
    return "\n\n".join(lines) or "No proactive memories surfaced."


@mcp_server.tool()
def memory_graph_query(cypher: str) -> str:
    """Run a Cypher query against RASPUTIN's FalkorDB knowledge graph.
    Example: MATCH (n:Entity) RETURN n.name LIMIT 10"""
    raw = _redis_cmd("GRAPH.QUERY", "memory_graph", cypher)
    if not raw:
        return "Empty result or graph not found."
    return raw[:4000]


@mcp_server.tool()
def memory_stats() -> str:
    """Get stats about RASPUTIN's memory system: vector counts, collections, graph info."""
    lines = []
    # Qdrant collections via the brain's /search as a proxy — get counts from Qdrant directly
    try:
        collections_raw = subprocess.run(
            ["curl", "-sf", "http://localhost:6333/collections"], capture_output=True, text=True, timeout=10
        )
        cols = json.loads(collections_raw.stdout)
        for c in cols.get("result", {}).get("collections", []):
            name = c["name"]
            info = subprocess.run(
                ["curl", "-sf", f"http://localhost:6333/collections/{name}"], capture_output=True, text=True, timeout=10
            )
            cdata = json.loads(info.stdout).get("result", {})
            count = cdata.get("points_count", cdata.get("vectors_count", "?"))
            lines.append(f"📦 {name}: {count:,} vectors" if isinstance(count, int) else f"📦 {name}: {count} vectors")
    except Exception as e:
        lines.append(f"Qdrant stats error: {e}")

    # FalkorDB graph info
    try:
        graph_info = _redis_cmd("GRAPH.QUERY", "memory_graph", "MATCH (n) RETURN count(n) as nodes")
        lines.append(f"🔗 Graph nodes: {graph_info}")
    except Exception as e:
        lines.append(f"Graph stats error: {e}")

    # Brain API health
    try:
        _http_json(f"{BRAIN_URL}/search?q=test&limit=1")
        lines.append("✅ Brain API (port 7777): responding")
    except Exception:
        lines.append("❌ Brain API (port 7777): not responding")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run as HTTP/SSE server on port 8101")
    args = parser.parse_args()

    if args.http:
        mcp_server.run(transport="sse", host="0.0.0.0", port=8101)
    else:
        mcp_server.run(transport="stdio")
