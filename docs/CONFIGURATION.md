# Configuration Reference

`config/rasputin.toml` is the single source of truth for runtime configuration.

Most services also support environment variable overrides (see `tools/config.py`), but default values should be maintained in TOML.

---

## Primary Config File

Path: `config/rasputin.toml`

Sections:

- `[server]` — API host/port
- `[qdrant]` — vector DB URL and collection
- `[graph]` — FalkorDB host/port/graph name and disable flag
- `[embeddings]` — embedding provider, model, dimensions, and fallback
- `[reranker]` — reranker provider, model, and fallback
- `[llm]` — primary LLM provider and model
- `[benchmark]` — benchmark defaults (answer model, judge, chunks)
- `[amac]` — A-MAC threshold, timeout, model, and URL
- `[scoring]` — temporal half-life controls
- `[constraints]` — constraint extraction provider and model
- `[entities]` — known entities dictionary path

---

## Provider Fallback Chain

Each provider has a primary (API-based) and fallback (local). If the API key is not set, the system falls back automatically.

### Embeddings: Gemini → Ollama

```toml
[embeddings]
provider = "gemini"       # Primary: Google AI API
fallback = "ollama"       # Fallback: local nomic-embed-text
```

| Provider | Model | Speed | Cost | Quality |
|---|---|---|---|---|
| `gemini` | gemini-embedding-001 (768d) | ~1.5s/call | ~$0.001/1K tokens | High (MTEB top-10) |
| `ollama` | nomic-embed-text (768d) | ~22ms/call | Free | Good |

Set `GEMINI_API_KEY` to enable Gemini. Without it, falls back to Ollama automatically.

### Reranking: Cohere → Local BGE

```toml
[reranker]
provider = "cohere"       # Primary: Cohere API
fallback = "local"        # Fallback: local BGE cross-encoder
```

| Provider | Model | Speed | Cost | Quality |
|---|---|---|---|---|
| `cohere` | rerank-v3.5 | ~1.5s/call | ~$0.001/search | Best-in-class |
| `local` | BGE cross-encoder | ~50ms/call | Free | Good |
| `none` | Skip reranking | 0ms | Free | Baseline |

Set `COHERE_API_KEY` to enable Cohere. Without it, uses local BGE if available.

### LLM: Anthropic → Ollama

```toml
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
```

Used for: constraint extraction, contradiction detection, benchmark answer generation.

| Provider | Model | Cost | Quality |
|---|---|---|---|
| `anthropic` | claude-haiku-4-5 | ~$0.25/M input | Production-grade |
| `anthropic` | claude-sonnet-4-6 | ~$3/M input | High quality (constraints) |
| `ollama` | qwen3.5:9b | Free | Adequate for local dev |

---

## Tiered Setups

| Setup | Embeddings | Reranker | LLM | Cost/LoCoMo run |
|---|---|---|---|---|
| Full cloud (default) | Gemini | Cohere | Claude Haiku | ~$2-5 |
| OpenAI-only | text-embedding-3-small | none | gpt-4o-mini | ~$3-6 |
| Fully local (free) | nomic via Ollama | BGE local | Ollama qwen | $0 |
| Batch mode | any | any | any | 50% of above |

---

## Override Hierarchy

Each level overrides the one above:

1. `config/rasputin.toml` — committed defaults, reflects production
2. `.env` — local overrides, gitignored
3. Environment variables — per-session overrides
4. CLI flags — per-run overrides

---

## Port Reference

| Port | Service | Purpose |
|------|---------|---------|
| 6333 | Qdrant | Vector database HTTP API |
| 6334 | Qdrant | gRPC (optional) |
| 6380 | FalkorDB | Graph database (Redis protocol) |
| 7777 | hybrid_brain | Main memory API |
| 8006 | reranker_server | Neural reranker API |
| 11434 | Ollama / local LLM API | Embeddings and chat-completions endpoint |
| 18790 | openclaw-mem | Hook/MCP integration endpoint |

---

## Environment Override Pattern

Use env vars for deployment-specific values (container URLs, host bindings, secrets).
Keep stable defaults in `config/rasputin.toml`.

Practical examples:

- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `LLM_API_URL`
- `LLM_MODEL`
- `AMAC_LLM_URL`
- `SESSIONS_DIR`

---

## Qdrant Collection Schema

Expected vector settings:

```json
{
  "name": "second_brain",
  "vectors": {
    "size": 768,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100
  }
}
```

Key payload fields used by the memory pipeline:

| Field | Type | Notes |
|---|---|---|
| `text` | string | Memory text (max 4000 chars) |
| `source` | string | Origin (`conversation`, `email`, etc.) |
| `source_weight` | float | Source tier weight (0.5–1.0) |
| `date` | string | ISO timestamp |
| `importance` | int | 0–100 ranking weight |
| `auto_committed` | bool | Always true for API commits |
| `retrieval_count` | int | Times retrieved in search |
| `last_accessed` | string | ISO timestamp of last retrieval |
| `embedding_model` | string | Embedding model identifier |
| `schema_version` | string | Current schema version (`0.7`) |
| `contradicts` | list[int] | Point IDs contradicted by this memory |
| `supersedes` | list[int] | Point IDs superseded by this memory |
| `has_contradictions` | bool | Fast filter for contradiction listing |
| `speaker` | string | Speaker name if known |
| `mentioned_names` | list[str] | Extracted capitalized names |
| `has_date` | bool | Whether text contains a date pattern |
| `connected_to` | list[int] | Graph-connected point IDs |
| `constraints` | list[dict] | Extracted implicit constraints |
| `constraint_summary` | string | Pipe-separated constraint text |
| `pending_archive` | bool | Flagged for archival by decay |
| `soft_deleted` | bool | Logically deleted |
| `pending_delete` | bool | Flagged for permanent deletion |

---

## Operational Notes

- Do not change embedding model or vector dimension on a populated collection without re-embedding.
- Keep A-MAC model/URL aligned with `config/rasputin.toml` (or explicit env override).
- Prefer config-driven behavior over hardcoded constants in scripts.
