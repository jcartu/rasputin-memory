# RASPUTIN Memory + Claude Code

## Setup

1. Start infrastructure and API server:
   ```bash
   docker compose up -d          # Qdrant + FalkorDB
   python3 tools/hybrid_brain.py # API server on :7777
   ```

2. Start MCP server:
   ```bash
   pip install "fastmcp>=3.2.0"
   python3 tools/mcp/server.py   # MCP server on :8808
   ```

3. Connect Claude Code:
   ```bash
   claude mcp add --transport http rasputin http://localhost:8808/mcp
   ```

4. Use naturally:
   - "Remember that we decided to use PostgreSQL for the auth service"
   - "What did we decide about the database?"
   - "What do you know about our architecture?"

## Available Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store facts, decisions, preferences to long-term memory |
| `memory_search` | Search memories with cross-encoder reranking |
| `memory_reflect` | Synthesize answers from multiple memories via LLM |
| `memory_stats` | Check memory system health and counts |
| `memory_feedback` | Mark memories as helpful/unhelpful |
| `memory_commit_conversation` | Bulk-commit conversation transcripts |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RASPUTIN_URL` | `http://127.0.0.1:7777` | RASPUTIN API server URL |
| `RASPUTIN_TOKEN` | (none) | Bearer token for API auth |
| `RASPUTIN_BANK_ID` | (none) | Memory bank / collection override |
| `MCP_HOST` | `127.0.0.1` | MCP server bind address |
| `MCP_PORT` | `8808` | MCP server port |

## Docker

```bash
docker compose up -d  # Starts qdrant, falkordb, brain, rasputin-mcp
claude mcp add --transport http rasputin http://localhost:8808/mcp
```
