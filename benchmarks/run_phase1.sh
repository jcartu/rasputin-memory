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
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export GEMINI_API_KEY="$GEMINI_API_KEY"

echo "=== Phase 1: Prompt Routing Benchmark ==="
echo "Start: $(date)"
echo "Config: PROMPT_ROUTING=1 CROSS_ENCODER=1 (CPU) FACT_EXTRACTION=1"
echo ""

echo "--- Step 1: Conv-0 ---"
rm -f benchmarks/results/locomo-leaderboard-checkpoint.json
python3 -u benchmarks/locomo_leaderboard_bench.py --conversations 0 --reset 2>&1
echo ""
echo "Conv-0 done: $(date)"

cp benchmarks/results/locomo-leaderboard-checkpoint.json \
   benchmarks/results/phase1-conv0-routing.json 2>/dev/null || true

echo ""
echo "--- Step 2: Full 10-conv ---"
python3 -u benchmarks/locomo_leaderboard_bench.py 2>&1

echo ""
echo "=== Phase 1 Complete: $(date) ==="
cp benchmarks/results/locomo-leaderboard-checkpoint.json \
   benchmarks/results/phase1-full-routing.json 2>/dev/null || true
