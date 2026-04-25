#!/bin/bash
set -e
trap '' USR1 USR2 HUP
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

export QDRANT_URL="http://localhost:6333"
export EMBED_URL="http://localhost:11434/api/embed"
export EMBED_MODEL="nomic-embed-text"
export CROSS_ENCODER=1
export CROSS_ENCODER_URL="http://${CROSS_ENCODER_HOST:-localhost}:9091/rerank"  # override CROSS_ENCODER_HOST for remote inference host
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
export CEREBRAS_API_KEY="$CEREBRAS_API_KEY"

echo "=== Phase 2: Consolidation Engine ==="
echo "Start: $(date)"
echo ""

# Step 1: Ingest all 10 convs with collections preserved
echo "--- Step 1: Ingest all conversations (--ingest-only) ---"
python3 -u benchmarks/locomo_leaderboard_bench.py --ingest-only 2>&1
echo "Ingest done: $(date)"
echo ""

# Step 2: Pre-compute consolidation observations
echo "--- Step 2: Consolidate facts → observations ---"
python3 -u benchmarks/precompute_consolidation.py 2>&1
echo "Consolidation done: $(date)"
echo ""

# Step 3: Run benchmark with observations as third search lane
echo "--- Step 3: Benchmark with OBSERVATIONS=1 ---"
export OBSERVATIONS=1
export BENCH_CHECKPOINT="phase2-obs-checkpoint.json"
rm -f "benchmarks/results/$BENCH_CHECKPOINT"
python3 -u benchmarks/locomo_leaderboard_bench.py --conversations 0 --reset 2>&1
cp "benchmarks/results/$BENCH_CHECKPOINT" benchmarks/results/phase2-conv0-obs.json 2>/dev/null || true
echo "Conv-0 with observations done: $(date)"
echo ""

echo "=== Phase 2 Complete: $(date) ==="
