# OpenClaw-Mem

A persistent memory system for [OpenClaw](https://github.com/openclaw) that automatically captures conversations, generates summaries, and injects relevant context into new sessions.

## Features

- **Persistent Memory** - Context survives across sessions
- **Progressive Disclosure** - Shows index first, fetch details on demand
- **Hybrid Search** - Full-text + LIKE search with CJK support
- **AI Compression** - Automatic summarization of observations
- **Token Efficient** - Only loads what's relevant
- **Real-time Capture** - Records messages as they happen
- **MCP Compatible** - Model Context Protocol server included
- **HTTP API** - REST API for memory queries

## Installation

### As OpenClaw Hook

```bash
# Clone to OpenClaw hooks directory
git clone https://github.com/wenyupapa-sys/openclaw-mem.git ~/.openclaw/hooks/openclaw-mem
cd ~/.openclaw/hooks/openclaw-mem
npm install
```

### As npm Package

```bash
npm install openclaw-mem
```

> ⚠️ **Important:** npm installation does NOT automatically prompt for API key configuration. You MUST manually configure your DeepSeek API key after installation. See [Configuration](#configuration) section below.

**After npm install, choose one of these methods:**

```bash
# Method 1: Run the setup wizard
npx openclaw-mem-setup

# Method 2: Set environment variable directly
export DEEPSEEK_API_KEY="your-deepseek-api-key"
# Add this line to your ~/.bashrc or ~/.zshrc to persist
```

## Quick Start

1. **Install the hook** (see above)

2. **Run setup** - configure your DeepSeek API key (prompted automatically after install)
   ```bash
   # Or run manually later
   npm run setup
   # or
   npx openclaw-mem-setup
   ```

3. **Restart OpenClaw** to load the hook

4. **Start chatting** - conversations are automatically saved

5. **Query memories** - ask "what did we discuss before?" and the AI will search the memory database

## Events Captured

| Event | Description |
|-------|-------------|
| `gateway:startup` | Initialize memory system |
| `agent:bootstrap` | Inject historical context |
| `agent:response` | Capture assistant responses |
| `agent:stop` | Save session summary |
| `command:new` | Save session before reset |
| `tool:post` | Capture tool usage |
| `user:prompt` | Capture user messages |

## API Reference

### HTTP API (Port 18790)

```bash
# Search memories
curl -s -X POST "http://127.0.0.1:18790/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"keyword","limit":10}'

# Get observation details
curl -s -X POST "http://127.0.0.1:18790/get_observations" \
  -H "Content-Type: application/json" \
  -d '{"ids":[123,124]}'

# Get timeline context
curl -s -X POST "http://127.0.0.1:18790/timeline" \
  -H "Content-Type: application/json" \
  -d '{"anchor":123}'

# Health check
curl "http://127.0.0.1:18790/health"
```

### Shell Scripts

```bash
# Search (handles CJK encoding automatically)
~/.openclaw/hooks/openclaw-mem/mem-search.sh "关键词" 10

# Get details
~/.openclaw/hooks/openclaw-mem/mem-get.sh 123 124 125
```

### MCP Server

```bash
# Start MCP server (stdio mode)
node mcp-server.js
```

MCP Tools:
- `search` - Search memory index
- `timeline` - Get context around an observation
- `get_observations` - Fetch full details

## Configuration

### Environment Variables

```bash
# Required for AI summarization (optional but recommended)
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Optional: Custom DeepSeek endpoint
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"

# Optional: Custom model
export DEEPSEEK_MODEL="deepseek-chat"
```

Get your DeepSeek API key at: https://platform.deepseek.com/

> **Note:** Without `DEEPSEEK_API_KEY`, the system will still work but won't generate AI summaries for sessions.

### OpenClaw Config

Add to your OpenClaw config:

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "openclaw-mem": {
          "enabled": true,
          "observationLimit": 50,
          "fullDetailCount": 5
        }
      }
    }
  }
}
```

## Storage

Data is stored in SQLite at `~/.openclaw-mem/memory.db`:

| Table | Description |
|-------|-------------|
| `sessions` | Session records |
| `observations` | Tool calls and messages |
| `summaries` | Session summaries |
| `user_prompts` | User inputs |

## Development

```bash
# Run tests
npm test

# Start HTTP API server
npm run api

# Start MCP server
npm run mcp

# Monitor real-time activity
node debug-logger.js
```

## 3-Layer Retrieval Workflow

For efficient token usage, use progressive disclosure:

1. **Search** → Get index with IDs (~50-100 tokens/result)
2. **Timeline** → Get context around interesting results
3. **Get Observations** → Fetch full details ONLY for filtered IDs

This approach saves ~30% tokens compared to fetching everything.

## License

MIT

## Contributing

Pull requests welcome! Please ensure tests pass before submitting.

## Credits

Inspired by [claude-mem](https://github.com/anthropics/claude-code) plugin architecture.
