export CI=true DEBIAN_FRONTEND=noninteractive GIT_TERMINAL_PROMPT=0
export PYTHONUNBUFFERED=1

# ---- Embeddings: Ollama nomic-embed-text (768d, matches v0.9.1 era) ----
# Rolled back from Qwen3-Embedding-0.6B to keep Phase A delta clean
# (one variable changed at a time — Phase A tests ranking, not embedder quality).
# vLLM qwen3-embedding-0.6b on :11437 remains UP for future standalone ablation.
export EMBED_PROVIDER=ollama
export EMBED_URL="http://localhost:11434/api/embed"
export EMBED_MODEL="nomic-embed-text"
export BENCH_EMBED_URL="http://localhost:11434/api/embed"
export BENCH_EMBED_MODEL="nomic-embed-text"
export EMBED_DIM=768
export BENCH_EMBED_DIM=768
export GEMINI_API_KEY=""

# ---- Persist collections so Phase A --skip-ingest can reuse them ----
export BENCH_KEEP_COLLECTIONS=1

# ---- Cross-encoder: arcstrider Qwen3-Reranker-0.6B (matches v0.9.1 config) ----
export CROSS_ENCODER_URL="http://192.168.1.41:9091/rerank"
export CROSS_ENCODER=1
export RERANK_PROVIDER="qwen3"

# ---- Fact extraction: Cerebras (NEW key + UA patch), Haiku fallback ----
export FACT_EXTRACTION_PROVIDER=cerebras
export CEREBRAS_API_KEY="csk-5cphk5vr56wwmmjenp5ceypcnfd84556c66x899x8v3rkj5n"
export CEREBRAS_FACT_MODEL="qwen-3-235b-a22b-instruct-2507"
export FACT_EXTRACTION_MODEL=claude-haiku-4-5-20251001
export FACT_EXTRACTION=1

# ---- Silence unused providers ----
export COHERE_API_KEY=""
