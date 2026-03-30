# STORM Wiki Generator

Generates comprehensive wiki-style articles from your Qdrant memory using [Stanford STORM](https://github.com/stanford-oval/storm) + a local LLM.

Instead of searching the web, STORM queries your `second_brain` Qdrant collection via a custom `QdrantLocalRM` retrieval module.

## Requirements

```bash
pip install knowledge-storm qdrant-client
```

You also need:
- **Qdrant** running on port 6333 with the `second_brain` collection populated
- **Ollama** running on port 11434 with `nomic-embed-text` for embeddings
- **An OpenAI-compatible LLM** endpoint at `http://localhost:11435/v1` (e.g., Ollama, vLLM, llama-swap)

## Usage

```bash
python3 generate.py "Topic Name"
python3 generate.py "Topic Name" --output ./articles/
python3 generate.py "Topic Name" --no-polish   # skip polishing step (faster)
```

## How It Works

1. STORM generates research questions about the topic
2. Each question is searched against Qdrant via `QdrantLocalRM` (custom retrieval module in `qdrant_rm.py`)
3. Retrieved memories are used as "sources" for article generation
4. The LLM synthesizes retrieved memories into a structured wiki article
5. Optional polishing pass improves readability

## Status

**Experimental.** Works well for topics with dense coverage in your memory corpus. Thin topics produce thin articles. Generation takes ~50s per article with a local 35B model.

## Files

| File | Description |
|------|-------------|
| `generate.py` | Main entry point — orchestrates STORM pipeline |
| `qdrant_rm.py` | Custom STORM retrieval module that queries Qdrant instead of the web |
