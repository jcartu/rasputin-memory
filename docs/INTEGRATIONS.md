# Integration Guide

RASPUTIN Memory exposes two interfaces: a direct HTTP API (port 7777) and an MCP server (port 8808).  Any MCP-compatible client can use the MCP server; everything else uses HTTP.

## MCP Clients

### Claude Code

```bash
python3 tools/mcp/server.py &
claude mcp add --transport http rasputin http://localhost:8808/mcp
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "rasputin": {
      "url": "http://localhost:8808/mcp"
    }
  }
}
```

### Codex CLI

```bash
codex --mcp-server http://localhost:8808/mcp
```

### OpenClaw

The `hooks/openclaw-mem/` directory provides a native OpenClaw hook with auto-recall.  For MCP-based integration instead:

```bash
python3 tools/mcp/server.py &
# Configure your OpenClaw MCP settings to point at http://localhost:8808/mcp
```

See `docs/OPENCLAW-INTEGRATION.md` for the hook-based approach.

## Direct HTTP API

For agents and frameworks without MCP support, call the REST API directly.

### Search

```bash
curl "http://localhost:7777/search?q=database+architecture&limit=5"
```

### Commit

```bash
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text":"We chose PostgreSQL for the auth service.","source":"decision","importance":80}'
```

### Reflect

```bash
curl -X POST http://localhost:7777/reflect \
  -H 'Content-Type: application/json' \
  -d '{"q":"What do we know about the auth service?","limit":20}'
```

### Commit Conversation

```bash
curl -X POST http://localhost:7777/commit_conversation \
  -H 'Content-Type: application/json' \
  -d '{
    "turns": [
      {"speaker": "Alice", "text": "Should we use PostgreSQL or MySQL?"},
      {"speaker": "Bob", "text": "PostgreSQL — better JSON support."}
    ],
    "source": "conversation",
    "window_size": 5,
    "stride": 2
  }'
```

### Stats

```bash
curl http://localhost:7777/stats
```

### Feedback

```bash
curl -X POST http://localhost:7777/feedback \
  -H 'Content-Type: application/json' \
  -d '{"point_id": 123456789, "helpful": true}'
```

## LangChain / LangGraph

Wrap the HTTP API as a LangChain tool:

```python
import requests
from langchain_core.tools import tool

@tool
def memory_search(query: str) -> str:
    """Search RASPUTIN long-term memory."""
    resp = requests.get("http://localhost:7777/search", params={"q": query, "limit": 10})
    return resp.json()

@tool
def memory_store(text: str, source: str = "conversation") -> str:
    """Store to RASPUTIN long-term memory."""
    resp = requests.post("http://localhost:7777/commit", json={"text": text, "source": source})
    return resp.json()
```

## Authentication

Set `MEMORY_API_TOKEN` on the RASPUTIN API server to require Bearer token auth.  Pass the same token to the MCP server via `RASPUTIN_TOKEN`.

```bash
# API server
MEMORY_API_TOKEN=secret-token python3 tools/hybrid_brain.py

# MCP server
RASPUTIN_TOKEN=secret-token python3 tools/mcp/server.py
```
