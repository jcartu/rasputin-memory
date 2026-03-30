# Embedding Model Setup

RASPUTIN Memory uses **Ollama** with the **nomic-embed-text** model for all vector embeddings.

## Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Pull the Embedding Model

```bash
ollama pull nomic-embed-text
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_URL` | `http://localhost:11434/api/embed` | Ollama embedding endpoint |
| `EMBED_MODEL` | `nomic-embed-text` | Model name |

Ollama runs on port **11434** by default. Override with `OLLAMA_HOST` env var if needed.

## ⚠️ CRITICAL: Model Version Compatibility

**You MUST use nomic-embed-text v1 (768 dimensions).**

nomic-embed-text v1.5 produces **different dimension embeddings** and is **INCOMPATIBLE** with existing vectors. Using v1.5 will cause **silent retrieval failures** — searches return garbage results with no error messages.

To verify your model produces 768-dim vectors:

```bash
curl -s http://localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":"test"}' | python3 -c "
import json, sys
data = json.load(sys.stdin)
dims = len(data['embeddings'][0])
print(f'Dimensions: {dims}')
assert dims == 768, f'WRONG! Expected 768, got {dims}. You may have nomic-embed-text v1.5.'
print('✅ Correct: nomic-embed-text v1 (768-dim)')
"
```

## GPU vs CPU

Ollama works on CPU but embedding is **~10x slower**. GPU is strongly recommended for production use.

To check if Ollama is using GPU:

```bash
ollama ps  # Shows running models and their device (GPU/CPU)
```

## Verify Embeddings Work

```bash
curl -s http://localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":"Hello world"}' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'✅ Got {len(data[\"embeddings\"][0])}-dim embedding')
"
```
