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

echo "=== Cross-Encoder A/B Test: L-6 vs L-12 ==="
echo "Start: $(date)"
echo "Arcstrider 192.168.1.41 — L-6 on :9091, L-12 on :9092"
echo ""

echo "--- Run A: L-6 (ms-marco-MiniLM-L-6-v2) conv-0 ---"
export CROSS_ENCODER_URL="http://192.168.1.41:9091/rerank"
export BENCH_CHECKPOINT="ce-ab-L6-checkpoint.json"
rm -f "benchmarks/results/$BENCH_CHECKPOINT"
python3 -u benchmarks/locomo_leaderboard_bench.py --conversations 0 --reset 2>&1
cp "benchmarks/results/$BENCH_CHECKPOINT" benchmarks/results/ce-ab-L6-results.json 2>/dev/null || true
echo ""
echo "L-6 done: $(date)"

echo ""
echo "--- Run B: L-12 (ms-marco-MiniLM-L-12-v2) conv-0 ---"
export CROSS_ENCODER_URL="http://192.168.1.41:9092/rerank"
export BENCH_CHECKPOINT="ce-ab-L12-checkpoint.json"
rm -f "benchmarks/results/$BENCH_CHECKPOINT"
python3 -u benchmarks/locomo_leaderboard_bench.py --conversations 0 --reset 2>&1
cp "benchmarks/results/$BENCH_CHECKPOINT" benchmarks/results/ce-ab-L12-results.json 2>/dev/null || true
echo ""
echo "L-12 done: $(date)"

echo ""
echo "=== A/B Test Complete: $(date) ==="
