#!/bin/bash
set -e
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
export BENCH_CUDA_DEVICES=2
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export GEMINI_API_KEY="$GEMINI_API_KEY"

echo "=== Phase 1: Prompt Routing Benchmark ==="
echo "Start: $(date)"
echo "Config: PROMPT_ROUTING=1 CROSS_ENCODER=1 (GPU 2/5090) FACT_EXTRACTION=1"
echo ""

# Step 1: Conv-0 only (199 questions, ~20 min)
echo "--- Step 1: Conv-0 (baseline comparison) ---"
rm -f benchmarks/results/locomo-leaderboard-checkpoint.json
python3 benchmarks/locomo_leaderboard_bench.py --conversations 0 --reset 2>&1
echo ""
echo "Conv-0 done: $(date)"

# Copy checkpoint as conv-0 result before continuing
cp benchmarks/results/locomo-leaderboard-checkpoint.json \
   benchmarks/results/phase1-conv0-routing.json 2>/dev/null || true

echo ""
echo "--- Step 2: Full 10-conv validation ---"
# Don't reset - continue from checkpoint (conv-0 already done)
python3 benchmarks/locomo_leaderboard_bench.py 2>&1

echo ""
echo "=== Phase 1 Complete: $(date) ==="
cp benchmarks/results/locomo-leaderboard-checkpoint.json \
   benchmarks/results/phase1-full-routing.json 2>/dev/null || true
