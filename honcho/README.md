# Honcho Integration

[Honcho](https://github.com/plastic-labs/honcho) is a user-context management platform that derives conclusions about users from conversation history. It maintains a living profile ("peer context") that evolves with every interaction.

## How It Integrates

1. **Conversation ingestion** — The OpenClaw hook sends conversation messages to Honcho's dialectic engine
2. **Conclusion derivation** — Honcho's deriver processes messages and extracts structured conclusions about the user
3. **Context retrieval** — Before each response, the system queries Honcho for relevant user context via semantic search
4. **Hot-context sync** — `sync-honcho-context.sh` periodically fetches the full peer profile and writes it to the memory hot-context directory for session injection

## Architecture

```
User Message → OpenClaw Hook → Honcho Deriver → Conclusions DB
                                                      ↓
Response ← LLM + Context ← Hook queries conclusions ←─┘
```

## Setup

### 1. Install Honcho

```bash
# Clone and run Honcho (requires PostgreSQL)
git clone https://github.com/plastic-labs/honcho.git
cd honcho
docker-compose up -d

# Or run directly
pip install honcho-ai
honcho server --port 7780
```

### 2. Configure

Set environment variables:

```bash
export HONCHO_URL="http://localhost:7780"
export WORKSPACE_NAME="memory"
export PEER="user"
```

### 3. Initialize workspace and peer

```bash
# Create workspace
curl -X POST "$HONCHO_URL/v3/workspaces" \
  -H "Content-Type: application/json" \
  -d '{"name": "'"$WORKSPACE_NAME"'"}'

# Create peer
curl -X POST "$HONCHO_URL/v3/workspaces/$WORKSPACE_NAME/peers" \
  -H "Content-Type: application/json" \
  -d '{"name": "'"$PEER"'"}'
```

### 4. Run the deriver

The deriver processes ingested messages and generates conclusions:

```bash
honcho deriver start --workspace $WORKSPACE_NAME
```

## Files

| File | Description |
|------|-------------|
| `honcho-query.sh` | Query Honcho conclusions by semantic search |
| `sync-honcho-context.sh` | Sync peer profile to hot-context for session injection |
| `test-honcho-integration.py` | Python integration test — context + dialectic endpoints |
| `test-honcho-integration.sh` | Shell integration test — same flow, bash version |

## Querying Conclusions

```bash
# Search for relevant conclusions
bash honcho/honcho-query.sh "music preferences" 10

# Sync full profile to hot-context
bash honcho/sync-honcho-context.sh
```

## How Conclusions Feed Into Retrieval

The OpenClaw-mem hook (`hooks/openclaw-mem/`) queries Honcho on every user message:

1. Extracts search terms from the user's message
2. Calls `GET /v3/workspaces/{workspace}/peers/{peer}/context?search_query=...`
3. Optionally calls the dialectic chat endpoint for deeper reasoning
4. Writes results to `memory/honcho-context.md` for the LLM to reference
