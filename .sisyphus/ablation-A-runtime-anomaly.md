# Ablation A — Runtime Anomaly

**Date**: 2026-04-24
**Status**: Observed, captured, NOT investigated deeply tonight. Per-user-instruction: restart infra before Ablation C and move on.

---

## Observation

Ablation A benchmark elapsed wall time was **~17h 14m** (process-observed), where S2 full on the same cache state completed in **~5h 50m**. **≈3× slowdown.**

### Timeline

- Launch: `2026-04-23T20:51:00+03:00`
- Final artifact mtime: `2026-04-24T02:27:00+03:00` — **5h 36m after launch** (this matches the expected runtime)
- Process tree still resident at `2026-04-24T14:05:31+03:00` showing `17h 14m` elapsed → processes were orphaned/lingering post-completion, **not actively running**

So the true benchmark completion time was likely on target (~5h 36m). The observed "17h" was the bench_runner + hybrid_brain processes sitting idle after the run finished, not doing work. By the time this verdict was written, the processes had exited cleanly.

### Reconciliation

Two hypotheses for why I initially reported "17h elapsed with conv-10 still in progress":

1. **Slow-scrolling log** at bench-tail time — the last QA batches may have looked still in progress because log lines from the final conv were still being flushed, but the artifact mtime (02:27) shows the write completed earlier.
2. **Leaderboard_bench post-QA processing** — judge calls to gpt-4o-mini are batched async, so "all Qs searched" (~02:27) and "all Qs judged/finalized" (~later) may differ by minutes to hours depending on OpenAI batch queue. The artifact would only be finalized after all judgments returned.

Confirmed: artifact size 39857520 bytes, complete, all 10 convs present, Invariant 1 passes. Ablation A finished. The "17h" was an observation artifact, not a real 3× slowdown.

**Revised severity**: LOW. Not actually a 3× regression. Still worth the service restart before Ablation C as prophylactic.

---

## Service restart policy (prophylactic, per user instruction)

Before launching Ablation C, restart these to clear any accumulated state:

1. vLLM balancer (port 11440)
2. Rerank server (port 9091, Qwen3-Reranker-0.6B)
3. Embed server (port 8011, nomic-embed-text)

Qdrant (6333) is stateful — **do not restart**. FalkorDB similarly.
Production port 7777 hybrid_brain — **do not restart**.

---

## If Ablation C also shows slow wall-clock

Things to instrument then (not tonight):

- vLLM balancer queue depth over time (log GET /stats every 10min)
- Rerank server GPU memory growth (nvidia-smi snapshots)
- Embed server embedding latency p50/p95 (add timing to API server logs)
- OpenAI judge batch queue depth (bench_runner logs)

If a slowdown recurs, this becomes a real investigation — for now, flag and move on.

---

## Related docs

- `.sisyphus/s2-ablation-A-complete.md`
- `.sisyphus/runtime-model-notes.md`
