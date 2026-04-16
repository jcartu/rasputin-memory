# Agent Integration

RASPUTIN Memory integrates with any AI agent framework. The recommended path is the **MCP server** for MCP-compatible clients, or the **HTTP API** for everything else.

## MCP Server (Recommended for Claude Code, Cursor, Codex)

The fastest path to integration. The MCP server (`tools/mcp/server.py`) is a thin HTTP proxy that exposes 6 tools over the Model Context Protocol:

| Tool | Description |
|------|-------------|
| `memory_store` | Store facts, decisions, preferences to long-term memory |
| `memory_search` | Search memories with cross-encoder reranking |
| `memory_reflect` | Synthesize answers from multiple memories via LLM |
| `memory_stats` | Check memory system health and counts |
| `memory_feedback` | Mark memories as helpful/unhelpful |
| `memory_commit_conversation` | Bulk-commit conversation transcripts |

```bash
pip install "fastmcp>=3.2.0"
python3 tools/mcp/server.py   # port 8808

# Claude Code
claude mcp add --transport http rasputin http://localhost:8808/mcp

# Cursor — add to .cursor/mcp.json:
# {"mcpServers": {"rasputin": {"url": "http://localhost:8808/mcp"}}}
```

See [`docs/CLAUDE-CODE.md`](CLAUDE-CODE.md) and [`docs/INTEGRATIONS.md`](INTEGRATIONS.md) for full setup.

## HTTP API (Direct Integration)

For agents without MCP support, call the REST API directly:

```bash
# Search
curl "http://localhost:7777/search?q=your+query&limit=5"

# Commit
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text": "Something worth remembering", "source": "my-agent"}'

# Reflect (LLM synthesis)
curl -X POST http://localhost:7777/reflect \
  -H 'Content-Type: application/json' \
  -d '{"q": "What do we know about X?", "limit": 20}'
```

## Auto-Recall Hook

On every incoming user message, the system automatically:
1. Searches Qdrant for relevant memories
2. Writes results to `memory/last-recall.md`
3. The agent reads this file and weaves relevant context into its response

Zero manual effort. Every response is informed by the full memory corpus.

### OpenClaw Hook (Optional)

The `hooks/openclaw-mem/` directory is an OpenClaw-specific hook that implements auto-recall. It is **entirely optional** — if you're not using OpenClaw, you can achieve the same result by calling the `/search` API endpoint directly.

See [`docs/OPENCLAW-INTEGRATION.md`](OPENCLAW-INTEGRATION.md) for OpenClaw-specific setup.

### Custom Integration

For any other agent framework, integrate via HTTP:

```bash
# Search memories
curl "http://localhost:7777/search?q=your+query&limit=5"

# Commit new memories
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text": "Something worth remembering", "source": "my-agent"}'
```

## Hot Context System

The `memory/hot-context/` directory holds time-sensitive outputs from cron jobs and automated analysis. Files have a 24-hour TTL and are auto-loaded at every new agent session — giving the agent awareness of recent events without manual injection.

## Memory Autogen

A nightly cron regenerates `MEMORY.md` with dynamic sections pulled from live Qdrant data, recent daily logs, and consolidated facts. This serves as the agent's "working memory" summary at session start.

## MCP Protocol

For MCP-compatible agents, use the built-in MCP server (`tools/mcp/server.py`) which maps all tools to the HTTP API. See the [MCP Server](#mcp-server-recommended-for-claude-code-cursor-codex) section at the top of this document.
