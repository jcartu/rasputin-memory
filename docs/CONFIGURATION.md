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
- `[embeddings]` — embedding endpoint/model and query/doc prefixes
- `[reranker]` — reranker URL, timeout, and enable switch
- `[amac]` — A-MAC threshold, timeout, model, and URL
- `[scoring]` — temporal half-life controls
- `[entities]` — known entities dictionary path

Example (`[amac]`):

```toml
[amac]
threshold = 4.0
timeout = 30
model = "qwen2.5:14b"
url = "http://localhost:11434/v1/chat/completions"
```

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
|-------|------|-------|
| `text` | string | Memory text (trimmed before insert) |
| `source` | string | Origin (`conversation`, `email`, etc.) |
| `date` | string | ISO timestamp |
| `importance` | int | 0–100 ranking weight |
| `embedding_model` | string | Embedding model identifier |
| `schema_version` | string | Current schema version (`0.3`) |
| `contradicts` | list | IDs contradicted by this memory |
| `supersedes` | list | IDs superseded by this memory |
| `has_contradictions` | bool | Fast filter for contradiction listing |

---

## Operational Notes

- Do not change embedding model or vector dimension on a populated collection without re-embedding.
- Keep A-MAC model/URL aligned with `config/rasputin.toml` (or explicit env override).
- Prefer config-driven behavior over hardcoded constants in scripts.
