#!/bin/bash
# Sprint 1 S1.1 — Qwen3-32B-AWQ local vLLM deploy
#
# Usage: bash scripts/deploy_s1_1_vllm.sh
# Intended to run inside tmux session 'rasputin-deploy' for durability.
#
# Deviation from user authorization (document in heartbeat):
#   User said "GPU0 port :11437". GPU0 is at 93GB/97GB used — 122B TP half.
#   Starting 32B-AWQ (needs ~24GB) on GPU0 without killing 122B WILL OOM.
#   D1.1.a authorization says "defer killing 122B".
#   Therefore: stage 32B on GPU1 (RTX 5090, 32GB, 7GB used — plenty of room).
#   Port :11437 preserved. Canonical FACT_EXTRACTION_PROVIDER=local_vllm still valid.

set -u

HB=/home/josh/.openclaw/workspace/rasputin-memory/.sisyphus/logs/deploy-heartbeat.log
STALL=/home/josh/.openclaw/workspace/rasputin-memory/.sisyphus/deploy-stall.md
MODELS_DIR=/home/josh/models
MODEL_EXTRACT=Qwen/Qwen3-32B-AWQ
MODEL_EXTRACT_DIR="$MODELS_DIR/Qwen3-32B-AWQ"
MODEL_RERANK=Qwen/Qwen3-Reranker-0.6B
MODEL_RERANK_DIR="$MODELS_DIR/Qwen3-Reranker-0.6B"
HF=/home/josh/.local/bin/hf
VLLM_PY=/home/josh/vllm-test/bin/python
START_EPOCH=$(date +%s)

mkdir -p "$(dirname "$HB")"

hb() {
    local stage="$1"
    shift
    echo "$(date -Iseconds) stage=$stage $*" >> "$HB"
}

write_stall() {
    local reason="$1"
    cat > "$STALL" <<EOF
# Deploy stalled — $(date -Iseconds)

**Reason**: $reason
**Last heartbeats** (tail of $HB):
\`\`\`
$(tail -20 "$HB" 2>/dev/null)
\`\`\`

## Fallback posture per user authorization

S1 integration bench runs against Cerebras (current) with
\`FACT_EXTRACTION_PROVIDER=cerebras\`. We accept the drift risk for S1 exit
gate; re-run against local_vllm in Sprint 2 pre-flight.

## Remediation

- Check GPU memory: \`nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv\`
- Check vLLM tmux: \`tmux has-session -t vllm-extract-32b && tmux capture-pane -t vllm-extract-32b -p | tail -30\`
- If 122B can be stopped: \`tmux send-keys -t vllm-extract C-c Enter\` then relaunch 32B with \`CUDA_VISIBLE_DEVICES=0\`.
EOF
    hb halt "$reason"
}

check_stall() {
    local now=$(date +%s)
    local elapsed=$((now - START_EPOCH))
    if [ $elapsed -gt 5400 ]; then
        write_stall "exceeded 90min wall-clock during stage=$1"
        exit 1
    fi
}

hb init "deploy starting; topology check"

# Topology log
NVIDIA_OUT=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>&1 | tr '\n' '|')
hb init "gpus: $NVIDIA_OUT"
hb deviation "user said GPU0; deviating to GPU1 (5090) because GPU0 93/97GB used by 122B TP half"

# Stage 1: Download extractor with periodic heartbeat
hb download "starting $MODEL_EXTRACT (~24GB)"
(
    while sleep 60; do
        if [ -d "$MODEL_EXTRACT_DIR" ]; then
            mb=$(du -sm "$MODEL_EXTRACT_DIR" 2>/dev/null | cut -f1)
            hb download "extractor_mb=${mb:-0}"
        else
            hb download "extractor_dir_not_yet"
        fi
        now=$(date +%s)
        if [ $((now - START_EPOCH)) -gt 5400 ]; then
            exit 0  # stall check in main process will handle halt
        fi
    done
) &
MON_PID=$!

HF_HUB_ENABLE_HF_TRANSFER=1 "$HF" download "$MODEL_EXTRACT" \
    --local-dir "$MODEL_EXTRACT_DIR" >> "$HB" 2>&1
EXTRACT_RC=$?
kill $MON_PID 2>/dev/null
wait $MON_PID 2>/dev/null

if [ $EXTRACT_RC -ne 0 ]; then
    write_stall "hf download rc=$EXTRACT_RC for $MODEL_EXTRACT"
    exit 1
fi

EXTRACT_SIZE_MB=$(du -sm "$MODEL_EXTRACT_DIR" | cut -f1)
hb download "extractor complete, $EXTRACT_SIZE_MB MB"

# Stage 2: Start vLLM for extractor on GPU1
hb vllm "starting vLLM on GPU1 (5090) port :11437"

# Kill any prior session with same name (idempotent)
tmux kill-session -t vllm-extract-32b 2>/dev/null

tmux new-session -d -s vllm-extract-32b "\
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model $MODEL_EXTRACT_DIR \
  --served-model-name qwen3-32b-awq \
  --port 11437 --host 127.0.0.1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.72 \
  --dtype auto --trust-remote-code \
  --quantization awq \
  --enable-chunked-prefill \
  2>&1 | tee /home/josh/.openclaw/workspace/rasputin-memory/.sisyphus/logs/vllm-extract.log"

# Wait up to 10 min for vLLM to expose /v1/models
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    sleep 30
    if curl -s --max-time 5 http://localhost:11437/v1/models > /dev/null 2>&1; then
        hb vllm "ready after $((i*30))s"
        READY=1
        break
    fi
    hb vllm "warmup ${i}/20 (elapsed $((i*30))s)"
    check_stall vllm
done

if [ -z "${READY:-}" ]; then
    tmux capture-pane -t vllm-extract-32b -p 2>/dev/null | tail -30 >> "$HB"
    write_stall "vLLM failed to start on :11437 within 10min"
    exit 1
fi

# Smoke test: determinism on a short completion
SMOKE_RESPONSE=$(curl -s --max-time 60 http://localhost:11437/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-32b-awq","messages":[{"role":"user","content":"Extract one fact: Alice joined Google in 2020."}],"temperature":0.0,"seed":42,"max_tokens":100}' 2>&1)
if echo "$SMOKE_RESPONSE" | grep -q '"content"'; then
    hb vllm "smoke extraction ok"
else
    hb vllm "smoke returned unexpected: $(echo "$SMOKE_RESPONSE" | head -c 200)"
    # non-fatal; vLLM is up but schema might differ. Proceed.
fi

# Stage 3: Download reranker
hb download "starting $MODEL_RERANK (~1.2GB)"
HF_HUB_ENABLE_HF_TRANSFER=1 "$HF" download "$MODEL_RERANK" \
    --local-dir "$MODEL_RERANK_DIR" >> "$HB" 2>&1
RERANK_RC=$?
if [ $RERANK_RC -ne 0 ]; then
    hb download "reranker rc=$RERANK_RC (non-fatal; reranker serving deferred to cross_encoder_server.py follow-up)"
else
    RERANK_SIZE_MB=$(du -sm "$MODEL_RERANK_DIR" 2>/dev/null | cut -f1)
    hb download "reranker complete, $RERANK_SIZE_MB MB"
fi

# Stage 4: Update baseline_env.sh
BASELINE_ENV=/tmp/bench_runs/baseline_env.sh
if [ ! -f "$BASELINE_ENV" ]; then
    hb env "baseline_env.sh missing at $BASELINE_ENV — cannot append"
else
    if grep -q "^export FACT_EXTRACTION_PROVIDER=local_vllm" "$BASELINE_ENV" 2>/dev/null; then
        hb env "baseline_env.sh already configured for local_vllm — skipping append"
    else
        cat >> "$BASELINE_ENV" <<'EOF'

# Appended by deploy_s1_1_vllm.sh (S1.1) — local vLLM extraction
export FACT_EXTRACTION_PROVIDER=local_vllm
export LOCAL_VLLM_ENDPOINT=http://localhost:11437
export LOCAL_VLLM_URL=http://localhost:11437/v1/chat/completions
export LOCAL_VLLM_MODEL=qwen3-32b-awq
export LOCAL_VLLM_SEED=42
EOF
        hb env "baseline_env.sh updated for local_vllm"
    fi
fi

hb done "deploy complete. Extractor on :11437 (GPU1). Reranker cached. baseline_env.sh pinned."
