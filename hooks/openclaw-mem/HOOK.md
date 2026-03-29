---
name: openclaw-mem
description: "Persistent memory system - saves session context and injects history into new sessions"
homepage: https://github.com/openclaw-mem
metadata:
  {
    "openclaw":
      {
        "emoji": "🧠",
        "events": ["command:new", "gateway:startup", "agent:bootstrap", "agent:response", "agent:stop", "tool:post", "user:prompt", "message"],
        "requires": { "config": ["workspace.dir"] },
        "install": [{ "id": "local", "kind": "local", "label": "Local installation" }],
      },
  }
---

# OpenClaw-Mem: Persistent Memory System

A claude-mem inspired memory system for OpenClaw that automatically captures tool usage, generates summaries, and injects relevant context into new sessions.

## Features

- 🧠 **Persistent Memory** - Context survives across sessions
- 📊 **Progressive Disclosure** - Shows index first, fetch details on demand
- 🔍 **Hybrid Search** - Full-text + semantic search
- 🤖 **AI Compression** - Automatic summarization of observations
- ⚡ **Token Efficient** - Only loads what's relevant
- 📡 **Real-time Capture** - Records messages as they happen

## Events Captured

- `gateway:startup` - Initialize memory system
- `agent:bootstrap` - Inject historical context
- `agent:response` - Capture assistant responses in real-time
- `agent:stop` - Save session summary
- `command:new` - Save session content before reset
- `tool:post` - Capture tool usage
- `user:prompt` - Capture user messages
- `message` - Capture all messages

## Configuration

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "openclaw-mem": {
          "enabled": true,
          "observationLimit": 50,
          "fullDetailCount": 5,
          "compressWithLLM": true
        }
      }
    }
  }
}
```

## Storage

Data is stored in SQLite at `~/.openclaw-mem/memory.db`:
- `sessions` - Session records
- `observations` - Tool call observations
- `summaries` - Session summaries

## Real-time Monitoring

```bash
node ~/.openclaw/hooks/openclaw-mem/monitor.js
```

## Usage

The memory system works automatically. To search manually:

```bash
# Search memory
openclaw memory search "authentication"

# View status
openclaw memory status
```

## MCP Server (Model Context Protocol)

OpenClaw-Mem 提供 MCP Server，让 AI 可以按需查询记忆：

### MCP 工具

| Tool | Description |
|------|-------------|
| `__IMPORTANT` | 显示 3 层工作流说明 |
| `search` | 搜索记忆索引 |
| `timeline` | 获取某条记录的上下文 |
| `get_observations` | 获取完整详情 |

### 启动 MCP Server

```bash
# stdio 模式（标准 MCP）
node ~/.openclaw/hooks/openclaw-mem/mcp-server.js

# HTTP API 模式（兼容模式）
node ~/.openclaw/hooks/openclaw-mem/mcp-http-api.js
```

### HTTP API 端点

```bash
# 搜索
curl "http://127.0.0.1:18790/search?query=database&limit=10"

# Timeline
curl "http://127.0.0.1:18790/timeline?anchor=123"

# 获取详情
curl -X POST "http://127.0.0.1:18790/get_observations" -d '{"ids":[123,124]}'
```

## Disabling

```bash
openclaw hooks disable openclaw-mem
```
