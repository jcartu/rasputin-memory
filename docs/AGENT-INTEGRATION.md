# Agent Integration

RASPUTIN Memory is designed to integrate with any AI agent framework via its HTTP API. This guide covers the built-in integration patterns.

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

## Honcho Integration

Theory of Mind / psychological modeling via [Honcho](https://github.com/plastic-labs/honcho). Tracks user preferences, communication style, and behavioral patterns to personalize retrieval and response generation.

## MCP Protocol

The `tools/memory_mcp_server.py` provides an MCP (Model Context Protocol) adapter, allowing any MCP-compatible agent to use RASPUTIN Memory as a tool provider.
