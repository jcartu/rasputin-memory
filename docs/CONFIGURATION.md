# Configuration Reference

All configuration options, environment variables, ports, and model choices.

## hybrid_brain.py Constants

These are hardcoded constants at the top of `hybrid_brain.py`. Edit them to customize your deployment:

```python
# === Database Connections ===
QDRANT_URL = "http://localhost:6333"
COLLECTION = "second_brain"          # Qdrant collection name

FALKORDB_HOST = "localhost"
FALKORDB_PORT = 6380                 # FalkorDB Docker maps 6379→6380
FALKORDB_GRAPH = "brain"             # Graph name inside FalkorDB

# === Embedding Service ===
EMBED_URL = "http://localhost:11434/api/embed"  # Ollama endpoint
EMBED_MODEL = "nomic-embed-text"               # MUST match stored vectors
EMBED_DIM = 768                                # nomic-embed-text v1 dimension

# === Neural Reranker ===
RERANKER_URL = "http://localhost:8006/rerank"

# === LLM Enrichment ===
# A-MAC quality gate uses OpenAI-compatible endpoint (llama-swap, not Ollama)
AMAC_LLM_URL = "http://localhost:11436/v1/chat/completions"
AMAC_OLLAMA_MODEL = "qwen3.5:35b"

# === A-MAC Quality Gate ===
AMAC_THRESHOLD = 4.0           # Composite score minimum (0.0–10.0)
AMAC_REJECT_LOG = "/tmp/amac_rejected.log"
AMAC_TIMEOUT = 30              # Seconds before fail-open

# === Search Tuning ===
DEFAULT_SEARCH_LIMIT = 10        # Default result count
DEFAULT_SCORE_THRESHOLD = 0.50   # Minimum cosine similarity
RERANKER_CANDIDATE_POOL = 50     # Top-N passed to neural reranker
DEDUP_THRESHOLD = 0.92           # Cosine similarity = duplicate
TEXT_OVERLAP_THRESHOLD = 0.50    # Text overlap % = duplicate

# === Temporal Decay ===
TEMPORAL_HALF_LIFE_DAYS = 30     # Ebbinghaus decay half-life
```

---

## memory_engine.py Constants

```python
QDRANT_URL = "http://localhost:6333"
EMBED_URL = "http://localhost:11434/api/embed"   # Must use Ollama, not GPU embed service
EMBED_MODEL = "nomic-embed-text"
RERANKER_URL = "http://localhost:8006/rerank"
COLLECTION = "second_brain"

# Observational Memory
ENTITY_GRAPH = "~/.openclaw/workspace/memory/entity_graph.json"
OM_OBSERVATIONS = "~/.openclaw/workspace/memory/om_observations.md"
OM_MAX_AGE_HOURS = 24            # Refresh threshold for OM cache
```

---

## fact_extractor.py Constants

```python
WORKSPACE = Path.home() / '.openclaw' / 'workspace'
SESSIONS_DIR = Path.home() / '.openclaw/agents/main/sessions'
FACTS_FILE = WORKSPACE / 'memory' / 'facts.jsonl'
STATE_FILE = WORKSPACE / 'memory' / 'fact_extractor_state.json'

QDRANT_URL = "http://localhost:6333"
EMBED_URL = "http://localhost:11434/api/embeddings"

# LLM endpoint for fact extraction
LLM_PROXY_URL = "http://localhost:11436/v1/chat/completions"
LLM_MODEL = "qwen3.5-122b-a10b"    # Local model, zero cost
```

---

## Port Reference

| Port | Service | Purpose |
|------|---------|---------|
| **6333** | Qdrant | Vector database HTTP API |
| **6334** | Qdrant | gRPC (not used by default) |
| **6380** | FalkorDB | Graph database (Redis protocol) |
| **7777** | hybrid_brain.py | Main memory API |
| **8006** | bge-reranker | Neural reranker HTTP API |
| **11434** | Ollama | Embedding + LLM inference |
| **11436** | llama-swap (Qwen 35B) | A-MAC quality gate (OpenAI-compatible) |
| **18790** | openclaw-mem | MCP server + session capture |
| **8889** | llm-proxy | Optional: LLM routing proxy |

---

## Model Selection Guide

### Embedding Model

**DO NOT CHANGE** once you have vectors stored. All vectors must use the same model.

| Model | Dimensions | Endpoint | Use case |
|-------|-----------|---------|---------|
| `nomic-embed-text` v1 | 768 | Ollama :11434 | ✅ **Production default** |
| `nomic-embed-text` v1.5 | 768 | GPU embed :8003 | ❌ Incompatible with v1 vectors |
| `text-embedding-3-small` | 1536 | OpenAI API | Alternative if no local GPU |
| `mxbai-embed-large` | 1024 | Ollama | Alternative, higher quality |

### Reranker Model

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `BAAI/bge-reranker-v2-m3` | 2.3GB | ⭐⭐⭐⭐⭐ | ~50ms/batch |
| `BAAI/bge-reranker-base` | 0.5GB | ⭐⭐⭐ | ~20ms/batch |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.5GB | ⭐⭐⭐ | ~15ms/batch |

### A-MAC LLM

| Model | Cost | Quality | Speed |
|-------|------|---------|-------|
| `qwen3.5:35b` local | $0 | ⭐⭐⭐⭐ | ~5-15s |
| `qwen2.5:7b` local | $0 | ⭐⭐⭐ | ~2s |
| `claude-3-haiku` API | ~$0.001/call | ⭐⭐⭐⭐ | ~1s |

---

## Qdrant Collection Schema

```json
{
  "name": "second_brain",
  "vectors": {
    "size": 768,
    "distance": "Cosine"
  },
  "optimizers_config": {
    "memmap_threshold": 20000
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100
  }
}
```

**Payload fields used by the system:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Main content (max 4000 chars) |
| `source` | string | Origin: `email`, `chatgpt`, `perplexity`, `conversation` |
| `date` | string | ISO 8601 timestamp |
| `importance` | int | 0–100 score for ranking |
| `auto_committed` | bool | True if committed by the system |
| `subject` | string | Email: subject line |
| `from` | string | Email: sender address |
| `to` | string | Email: recipient address |
| `thread_id` | string | Email: thread dedup key |
| `title` | string | ChatGPT: conversation title |
| `question` | string | Perplexity: original query |
| `filename` | string | Perplexity: source file |

---

## FalkorDB Graph Schema

**Nodes:**
```cypher
(:Entity {name: string, type: string, last_seen: string})
(:Memory {id: string, timestamp: string, text_preview: string})
```

**Relationships:**
```cypher
(:Entity)-[:MENTIONED_IN]->(:Memory)
(:Entity)-[:RELATED_TO {strength: float}]->(:Entity)
```

**Common queries:**
```bash
# All entities for a memory
redis-cli -p 6380 GRAPH.QUERY brain "MATCH (e:Entity)-[:MENTIONED_IN]->(m:Memory {id: '12345'}) RETURN e.name"

# Memories mentioning a person
redis-cli -p 6380 GRAPH.QUERY brain "MATCH (e:Entity {name: 'Bob'})-[:MENTIONED_IN]->(m) RETURN m.id, m.text_preview LIMIT 10"

# Entity network (2 hops)
redis-cli -p 6380 GRAPH.QUERY brain "MATCH (e:Entity {name: 'Acme Corp'})-[*1..2]-(related) RETURN DISTINCT related.name LIMIT 20"
```

---

## openclaw-mem Hook Configuration

In `~/.openclaw/config.json`:

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "openclaw-mem": {
          "enabled": true,
          "observationLimit": 50,
          "fullDetailCount": 5,
          "compressWithLLM": false,
          "searchOnEveryMessage": true,
          "hotContextDir": "memory/hot-context",
          "lastRecallFile": "memory/last-recall.md"
        }
      }
    }
  },
  "workspace": {
    "dir": "~/.openclaw/workspace"
  }
}
```

---

## Cron Jobs

```bash
# Edit crontab
crontab -e

# Fact extraction every 4 hours
0 */4 * * * python3 tools/fact_extractor.py >> /tmp/fact_extractor.log 2>&1

# Second brain enrichment 6x nightly  
0 1-6 * * * python3 tools/enrich_second_brain.py --batch 100 >> /tmp/enrichment.log 2>&1

# Graph deepening daily at midnight
0 0 * * * python3 tools/consolidator.py migrate >> /tmp/consolidate.log 2>&1
```

---

## Environment Variables (Optional)

If you prefer environment-based config over constants:

```bash
export RASPUTIN_QDRANT_URL="http://localhost:6333"
export RASPUTIN_EMBED_URL="http://localhost:11434/api/embed"  
export RASPUTIN_RERANKER_URL="http://localhost:8006/rerank"
export RASPUTIN_AMAC_URL="http://localhost:11436/v1/chat/completions"
export RASPUTIN_AMAC_MODEL="qwen3.5:35b"
export RASPUTIN_AMAC_THRESHOLD="4.0"
export RASPUTIN_PORT="7777"
export RASPUTIN_WORKSPACE="~/.openclaw/workspace"
```

Most tools now read from environment variables with sensible defaults. See `.env.example` for the full list.

---

## A-MAC Quality Gate

**A-MAC** (Automated Memory Acceptance Criteria) is a quality gate that scores every incoming memory before it's committed to Qdrant. It prevents low-quality, vague, or duplicate information from polluting your memory store.

### How It Works

1. When a memory is submitted via `/commit`, A-MAC sends the text to a local LLM
2. The LLM scores it on three dimensions (0–10 each):
   - **Relevance** — Is this about topics that matter to the agent/user?
   - **Novelty** — Does this add genuinely new information?
   - **Specificity** — Is this a concrete fact with numbers/names/dates?
3. The three scores are averaged into a **composite score**
4. If the composite score is **below the threshold** (default: 4.0), the memory is **rejected** and logged to `/tmp/amac_rejected.log`
5. If the LLM is unreachable or times out, A-MAC **fails open** (accepts the commit)

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AMAC_LLM_URL` | `http://localhost:11436/v1/chat/completions` | OpenAI-compatible LLM endpoint for scoring |
| `AMAC_OLLAMA_MODEL` | `qwen3.5:35b` | Model name sent to the LLM endpoint |
| `AMAC_THRESHOLD` | `4.0` | Minimum composite score (0.0–10.0) |
| `AMAC_TIMEOUT` | `30` | Seconds before fail-open |

### Disabling A-MAC

For simpler setups or development, you can effectively disable A-MAC:

1. **Set threshold to 0:** Everything passes
   ```python
   AMAC_THRESHOLD = 0.0
   ```

2. **Force-commit:** The `/commit` API accepts a `force: true` parameter that bypasses A-MAC entirely

3. **No LLM available:** If the LLM endpoint is unreachable, A-MAC automatically fails open and accepts all commits

### Monitoring

Rejected memories are logged to `AMAC_REJECT_LOG` (default: `/tmp/amac_rejected.log`). Check this periodically to ensure the threshold isn't too aggressive.

The `/health` endpoint includes A-MAC metrics: total scored, accepted, rejected, bypassed, and fail-open counts.
