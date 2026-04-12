"""RASPUTIN Memory MCP Server.

Thin MCP wrapper over the RASPUTIN HTTP API.  Requires the RASPUTIN
API server to be running (default: http://127.0.0.1:7777).

Usage:
    python3 tools/mcp/server.py

    # Claude Code integration
    claude mcp add --transport http rasputin http://localhost:8808/mcp
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger("rasputin.mcp")

RASPUTIN_URL = os.environ.get("RASPUTIN_URL", "http://127.0.0.1:7777")
RASPUTIN_TOKEN = os.environ.get("RASPUTIN_TOKEN", "")
BANK_ID = os.environ.get("RASPUTIN_BANK_ID", "")

MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8808"))

mcp = FastMCP(
    "rasputin-memory",
    instructions="Persistent long-term memory for AI agents. Store, search, and synthesize memories.",
)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _api(path: str, method: str = "GET", data: dict | None = None, timeout: int = 30) -> dict:
    """Call the RASPUTIN HTTP API."""
    url = f"{RASPUTIN_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    if RASPUTIN_TOKEN:
        req.add_header("Authorization", f"Bearer {RASPUTIN_TOKEN}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _collection_params() -> dict[str, str]:
    """Return collection override if BANK_ID is configured.

    BANK_ID maps to a Qdrant collection name.  Only applied on
    endpoints that support the ``collection`` parameter (currently
    ``/search`` only — ``/commit`` collection routing is planned for
    a future release).
    """
    if BANK_ID and BANK_ID != "default":
        return {"collection": BANK_ID}
    return {}


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool
def memory_store(
    content: str,
    source: str = "conversation",
    importance: int = 60,
) -> str:
    """Store information to long-term memory.

    Use this to remember facts, decisions, preferences, events, or any
    information that should persist across sessions.  Content passes
    through a quality gate (A-MAC) that rejects low-value noise.

    Args:
        content: The text to remember.  Be specific — include names, dates,
                 numbers. "Alice prefers dark mode" > "user likes dark".
        source: Origin label.  "conversation" for chat, "observation" for
                inferred facts, "decision" for choices made.
        importance: Priority 0-100.  Default 60.  Use 80+ for critical
                    decisions, 40 for background context.
    """
    payload: dict[str, Any] = {
        "text": content,
        "source": source,
        "importance": importance,
    }
    result = _api("/commit", method="POST", data=payload)

    if result.get("rejected"):
        scores = result.get("scores", {})
        return (
            f"Rejected by quality gate: {result.get('reason', 'below threshold')}\n"
            f"Scores: relevance={scores.get('relevance')}, "
            f"novelty={scores.get('novelty')}, "
            f"specificity={scores.get('specificity')} "
            f"(threshold: {result.get('threshold')})"
        )
    if result.get("ok"):
        graph_info = result.get("graph", {})
        entities = graph_info.get("connected_to", [])
        entity_str = f", entities: {entities}" if entities else ""
        dedup = result.get("dedup", {})
        action = dedup.get("action", "created") if dedup else "created"
        return f"Stored ({action}, id: {result.get('id', 'unknown')}{entity_str})"
    return f"Error: {result.get('error', 'unknown')}"


@mcp.tool
def memory_search(
    query: str,
    limit: int = 10,
) -> str:
    """Search long-term memory for relevant information.

    Returns ranked results scored by a cross-encoder reranker.  Use
    specific queries — "Alice's programming language preferences" is
    better than "preferences".

    Args:
        query: Natural language search query.
        limit: Maximum results to return (1-30, default 10).
    """
    limit = max(1, min(30, limit))
    qs: dict[str, Any] = {"q": query, "limit": limit}
    qs.update(_collection_params())
    params = urllib.parse.urlencode(qs)
    result = _api(f"/search?{params}")
    results = result.get("results", [])
    if not results:
        return "No matching memories found."

    lines = [f"Found {len(results)} memories ({result.get('elapsed_ms', '?')}ms):"]
    for i, r in enumerate(results):
        score = r.get("final_score") or r.get("rerank_score") or r.get("score", 0)
        text = (r.get("text") or "")[:300]
        date = (r.get("date") or "")[:10]
        source = r.get("source", "")
        pid = r.get("point_id", "")
        lines.append(f"\n[{i + 1}] score={score:.3f} | {date} | {source} | id={pid}")
        lines.append(f"    {text}")

    return "\n".join(lines)


@mcp.tool
def memory_reflect(
    query: str,
    limit: int = 20,
) -> str:
    """Synthesize a thoughtful answer from long-term memories.

    Unlike memory_search (which returns raw results), reflect uses an
    LLM to reason across multiple memories and produce a coherent
    answer.  Use for open-ended questions like "What do we know about X?"

    Args:
        query: The question to answer from memory.
        limit: How many memories to consider (1-30, default 20).
    """
    limit = max(1, min(30, limit))
    payload: dict[str, Any] = {"q": query, "limit": limit}
    payload.update(_collection_params())
    result = _api("/reflect", method="POST", data=payload, timeout=60)

    answer = result.get("answer", "No answer generated.")
    sources = result.get("sources", [])
    model = result.get("reflect_model", "unknown")
    search_ms = result.get("search_elapsed_ms", 0)

    parts = [answer, "", "---", f"Sources ({len(sources)} memories, {search_ms:.0f}ms search, model: {model}):"]
    for s in sources[:5]:
        text = (s.get("text") or "")[:100]
        parts.append(f"  - [{s.get('point_id', '?')}] {text}...")

    return "\n".join(parts)


@mcp.tool
def memory_stats() -> str:
    """Get memory system statistics.

    Returns counts of stored memories and knowledge graph nodes/edges.
    Use to check if the memory system is working.
    """
    result = _api("/stats")
    qdrant = result.get("qdrant", {})
    graph = result.get("graph", {})
    return (
        f"Qdrant: {qdrant.get('points', '?')} memories in collection '{qdrant.get('collection', '?')}'\n"
        f"Graph: {graph.get('nodes', '?')} entity nodes, {graph.get('edges', '?')} edges\n"
        f"Status: {result.get('status', '?')}"
    )


@mcp.tool
def memory_feedback(
    point_id: str,
    helpful: bool,
) -> str:
    """Mark a memory as helpful or not helpful.

    Positive feedback boosts importance for future retrieval.
    Negative feedback deprioritizes it.

    Args:
        point_id: The memory's identifier (from search results).
        helpful: True if useful, False if not.
    """
    # Qdrant stores point IDs as integers; convert for compatibility.
    try:
        pid: int | str = int(point_id)
    except (ValueError, TypeError):
        pid = point_id

    result = _api(
        "/feedback",
        method="POST",
        data={
            "point_id": pid,
            "helpful": helpful,
        },
    )
    if result.get("ok"):
        return (
            f"Feedback recorded: {'helpful' if helpful else 'not helpful'} for {point_id} "
            f"(importance: {result.get('importance_before')} -> {result.get('importance_after')})"
        )
    return f"Error: {result.get('error', 'unknown')}"


@mcp.tool
def memory_commit_conversation(
    turns: list[dict[str, str]],
    source: str = "conversation",
    window_size: int = 5,
    stride: int = 2,
) -> str:
    """Commit a multi-turn conversation to memory.

    Processes conversation into overlapping windows and stores
    everything.  Use at the end of a session to preserve the
    conversation for future recall.

    Args:
        turns: Conversation turns.  Each turn is a dict with "speaker"
               and "text" keys, e.g. {"speaker": "Alice", "text": "Hi"}.
        source: Origin label for these memories.
        window_size: Number of turns per window (default 5).
        stride: Window overlap stride (default 2).
    """
    result = _api(
        "/commit_conversation",
        method="POST",
        data={
            "turns": turns,
            "source": source,
            "window_size": window_size,
            "stride": stride,
        },
        timeout=120,
    )
    return (
        f"Committed: {result.get('turns_committed', '?')} turns, "
        f"{result.get('windows_committed', '?')} windows "
        f"(total: {result.get('total', '?')} memories)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
