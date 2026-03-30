# OpenClaw Integration

How to wire RASPUTIN into the OpenClaw agent framework for automatic memory recall and session capture.

## Architecture Overview

```
User message arrives
       │
       ▼
openclaw-mem hook (port 18790)
  ├── Captures user message to SQLite
  ├── Auto-recalls from Qdrant
  └── Writes results to memory/last-recall.md
       │
       ▼
Agent bootstrap (agent:bootstrap event)
  └── Reads memory/last-recall.md
  └── Injects hot-context/ files into session
       │
       ▼
Agent responds with memory context woven in
       │
       ▼
Session end (command:new or agent:stop)
  └── Session content saved to SQLite
  └── fact_extractor picks it up in next 4h run
```

---

## Installing the openclaw-mem Hook

```bash
# The hook lives at ~/.openclaw/hooks/openclaw-mem/
# If installing from scratch:
cd ~/.openclaw/hooks/
git clone https://github.com/phenomenoner/openclaw-mem

# Install dependencies
cd openclaw-mem
npm install

# Verify hook config in OpenClaw
openclaw config show hooks
```

The hook registers for these OpenClaw events:
- `command:new` — Save session content before reset
- `gateway:startup` — Initialize memory system
- `agent:bootstrap` — Inject historical context
- `agent:response` — Capture assistant responses
- `agent:stop` — Save session summary
- `tool:post` — Capture tool usage
- `user:prompt` — Capture user messages
- `message` — Capture all messages

---

## Auto-Recall on Every Message

The hook automatically searches memory on every user message and writes results to `memory/last-recall.md`.

**Hook configuration** (`~/.openclaw/config.json`):
```json
{
  "hooks": {
    "internal": {
      "entries": {
        "openclaw-mem": {
          "enabled": true,
          "observationLimit": 50,
          "fullDetailCount": 5,
          "compressWithLLM": false
        }
      }
    }
  }
}
```

**Agent rule** (in AGENTS.md or system prompt):
```
At the start of every response:
1. Check if memory/last-recall.md exists and was modified in the last 60 seconds
2. If yes: read it and weave relevant results into your response naturally
3. If no recent auto-recall: fall back to manual search:
   curl -s "http://localhost:7777/search?q=<keywords>&limit=5"
```

**Manual recall** (fallback):
```bash
# Direct API
curl -s "http://localhost:7777/search?q=the user+health+supplements&limit=5"

# Python CLI
python3 tools/memory_engine.py recall "What supplements is Alice taking?"

# Shell helper (mem-search.sh)
~/.openclaw/hooks/openclaw-mem/mem-search.sh "LATAM product"
```

---

## Hot Context System

Hot context files are injected into every new session automatically.

**Directory:** `memory/hot-context/`
**TTL:** 24 hours (stale entries are auto-ignored)
**Format:** Markdown files with frontmatter

### Creating hot context files

```bash
# Create a hot context entry (automatically expires in 24h)
cat > memory/hot-context/brazil-campaign.md << 'EOF'
---
created: 2026-03-15T14:00:00
ttl_hours: 24
priority: high
---
# Active: LATAM Campaign

Acme Corp launching regional only. Key decision: no Spanish language.
Campaign budget: $50K/month. Contact: marketing team.
EOF
```

### How it's injected

The `agent:bootstrap` event handler in `handler.js` reads all non-expired files from `memory/hot-context/` and injects them into the session context before the first user message.

```javascript
// handler.js (simplified)
async function injectHotContext(sessionDir) {
  const hotContextDir = path.join(workspaceDir, 'memory', 'hot-context');
  const files = await fs.readdir(hotContextDir);
  const now = Date.now();
  
  const contexts = [];
  for (const file of files) {
    const content = await fs.readFile(path.join(hotContextDir, file), 'utf8');
    const { data: frontmatter, content: body } = parseFrontmatter(content);
    
    const created = new Date(frontmatter.created).getTime();
    const ttlMs = (frontmatter.ttl_hours || 24) * 3600 * 1000;
    
    if (now - created < ttlMs) {
      contexts.push(body);
    }
  }
  
  return contexts.join('\n\n---\n\n');
}
```

---

## Session Memory (last-recall.md)

The hook writes search results to `memory/last-recall.md` in a structured format:

```markdown
# Memory Recall — 2026-03-15T14:22:01

Query: "Alice project project planning"
Results: 5 memories found

## 💬 ChatGPT conversation: project project planning session (2026-02-10)
the user and Alice planning project planning. Starting PGT-A genetic screening. 
Timeline: retrieval in April 2026.
[rerank: 0.923 | vector: 0.81]

## 📨 From planning@clinic.ru (2026-01-15)
Subject: project planning Protocol - Initial Consultation
Body: Protocol details, medication schedule, monitoring appointments...
[rerank: 0.887 | vector: 0.76]
```

**Reading it in your agent:**
```bash
# Check if fresh (modified within last 60 seconds)
if [ -f memory/last-recall.md ]; then
  AGE=$(($(date +%s) - $(date -r memory/last-recall.md +%s)))
  if [ $AGE -lt 60 ]; then
    cat memory/last-recall.md
  fi
fi
```

---

## Session Watcher (Real-time Capture)

`session-watcher.js` monitors active sessions and captures content in real-time:

```bash
# Start the session watcher
node ~/.openclaw/hooks/openclaw-mem/session-watcher.js

# Or monitor via PM2
pm2 start ~/.openclaw/hooks/openclaw-mem/session-watcher.js --name session-watcher
```

---

## MCP Server

The hook also exposes an MCP (Model Context Protocol) server at port 18790:

```bash
# Check MCP server
curl http://localhost:18790/health

# MCP config (MCP.json)
cat ~/.openclaw/hooks/openclaw-mem/MCP.json
```

This allows external MCP clients to query the memory system using the standardized MCP protocol.

---

## Storage: SQLite Database

Session conversations are stored in SQLite at `~/.openclaw-mem/memory.db`:

```bash
sqlite3 ~/.openclaw-mem/memory.db

# Check tables
.tables
# sessions  observations  summaries

# View recent sessions
SELECT id, created_at, message_count FROM sessions ORDER BY created_at DESC LIMIT 10;

# Search observations
SELECT * FROM observations WHERE content LIKE '%project planning%' ORDER BY timestamp DESC LIMIT 5;
```

**Tables:**
- `sessions` — session records (start time, end time, message count)
- `observations` — individual tool calls and messages
- `summaries` — LLM-generated session summaries

---

## Manual Memory Operations

```bash
# Quick commit from shell
curl -X POST http://localhost:7777/commit \
  -H 'Content-Type: application/json' \
  -d '{"text": "Memory to store", "source": "manual"}'

# Python CLI — full recall with multi-angle search
python3 tools/memory_engine.py recall "What did we decide about the product affiliate deal?"

# Deep dive on a topic
python3 tools/memory_engine.py deep "CHRONOS hardware wallet"

# Who is this person?
python3 tools/memory_engine.py whois "Bob"

# Morning briefing — surface urgent/upcoming items
python3 tools/memory_engine.py briefing

# Challenge a statement — find contradicting evidence
python3 tools/memory_engine.py challenge "We should spend $50K on Google Ads"
```

---

## AGENTS.md Configuration

Add these rules to your AGENTS.md (or equivalent system prompt) for automatic memory behavior:

```markdown
## Memory

- **Search:** `curl "http://localhost:7777/search?q=<query>&limit=5"`
- **Commit:** `curl -X POST http://localhost:7777/commit -H 'Content-Type: application/json' -d '{"text":"...", "source":"conversation"}'`
- **MANDATORY:** Run memory recall on EVERY message. No exceptions.

## Second Brain Auto-Recall (NON-NEGOTIABLE)

The openclaw-mem hook auto-searches Qdrant on EVERY user message and writes results to `memory/last-recall.md`.

At the start of every response:
1. Check if `memory/last-recall.md` exists and was modified in the last 60 seconds
2. If yes: read it and weave relevant results into your response naturally
3. If no recent auto-recall: fall back to manual search

## Hot Context

- `memory/hot-context/` files are auto-loaded at every new session
- 24h TTL — stale entries are automatically ignored
```
