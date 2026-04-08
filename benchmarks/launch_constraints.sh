#!/bin/bash
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/tmp/locomo-plus-constraints.log"

echo "$(date): Waiting for baseline to finish..." | tee "$LOG"

while true; do
    if grep -q "Report:" /tmp/locomo-plus.log 2>/dev/null; then
        echo "$(date): Baseline complete! Starting constraint run..." | tee -a "$LOG"
        break
    fi
    # Also check if baseline process is still running
    if ! pgrep -f "locomo_plus_bench.py --reset" >/dev/null 2>&1; then
        echo "$(date): Baseline process not found. Starting constraint run..." | tee -a "$LOG"
        break
    fi
    sleep 30
done

sleep 10

echo "$(date): Launching LoCoMo-Plus with constraints..." | tee -a "$LOG"
cd "$REPO"
python3 benchmarks/locomo_plus_bench.py --constraints --reset 2>&1 | tee -a "$LOG"
echo "$(date): Constraint run complete. Pushing results..." | tee -a "$LOG"

cd "$REPO"
git pull --rebase origin main 2>/dev/null || true
git add benchmarks/results/locomo-plus-constraints-*.json \
       benchmarks/results/locomo-plus-constraints-*.md \
       benchmarks/locomo_plus_bench.py \
       benchmarks/launch_constraints.sh \
       benchmarks/compare_runs.py 2>/dev/null
git commit -m "bench: LoCoMo-Plus constraint-enabled results (Haiku extraction)" 2>/dev/null
git push origin main 2>/dev/null
echo "$(date): Results pushed." | tee -a "$LOG"

echo "$(date): Running comparison..." | tee -a "$LOG"
python3 benchmarks/compare_runs.py 2>&1 | tee -a "$LOG"
git add benchmarks/results/locomo-plus-comparison.md 2>/dev/null
git commit -m "bench: LoCoMo-Plus baseline vs constraints comparison" 2>/dev/null
git push origin main 2>/dev/null
echo "$(date): Comparison pushed. All done." | tee -a "$LOG"
