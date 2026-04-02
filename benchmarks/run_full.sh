#!/bin/bash
cd /home/josh/.openclaw/workspace/rasputin-memory
python3 benchmarks/locomo_benchmark.py 2>&1 | tee benchmarks/results/locomo-run-full.log
echo "DONE: exit code $?"
