#!/bin/bash
set -e
trap '' USR1 USR2 HUP
cd /home/josh/.openclaw/workspace/rasputin-memory

export QDRANT_URL="http://localhost:6333"
export EMBED_URL="http://localhost:11434/api/embed"
export EMBED_MODEL="nomic-embed-text"
export CROSS_ENCODER=1
export FACT_EXTRACTION=1
export CHUNK_WINDOWS=1
export CHUNK_WINDOW_SIZE=5
export CHUNK_STRIDE=2
export PROMPT_ROUTING=1
export BENCH_MODE=production
export BENCH_CUDA_DEVICES=""
export CROSS_ENCODER_URL="http://192.168.1.41:9091/rerank"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export GEMINI_API_KEY="$GEMINI_API_KEY"

echo "=== Phase 1: Prompt Routing Benchmark ==="
echo "Start: $(date)"
echo "Config: PROMPT_ROUTING=1 CROSS_ENCODER=1 (Arcstrider 5090 GPU) FACT_EXTRACTION=1"
echo ""

echo "--- Resuming full 10-conv (checkpoint has 3 convs done) ---"
python3 -u benchmarks/locomo_leaderboard_bench.py 2>&1

echo ""
echo "=== Phase 1 Complete: $(date) ==="
cp benchmarks/results/locomo-leaderboard-checkpoint.json \
   benchmarks/results/phase1-full-routing.json 2>/dev/null || true
