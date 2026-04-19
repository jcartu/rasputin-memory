# Quarantine 2026-04-19 — Ghost-Checkpoint Regression

## TL;DR

**The -14.1pp "Phase A regression" (72.5% → 58.44% non-adv) was a benchmark-harness lineage
bug, not an algorithm regression.** Two different bench harnesses (`bench_runner.py locomo`
and `locomo_leaderboard_bench.py`) write to disjoint filenames. Phase A used the raw
harness, which resumed from a **weeks-old stale checkpoint** whose origin is no longer in
shell history. The alpha-formula rewrite under test ran on only 4 of 10 conversations
(47–50). The other 6 (26, 30, 41–44) were skipped as "already done" from the stale
checkpoint and carried pre-degraded answers into the v1.json output.

**The alpha formula was never given a fair test.** It also could not have moved the number
by 14pp even if it had — the `proof_norm` saturation diagnostic (pool size 60 → saturated
at 0.87–0.91 across all chunks) shows the multiplicative boost is a near-no-op.

## Files in this quarantine

| File | Origin | What it is |
|---|---|---|
| `locomo-leaderboard-v1.json` | `locomo_leaderboard_bench.py`, Apr 18 20:36 | 39 MB, 1986 rows, **58.44% non-adv** — the published "Phase A" number. Byte-identical to checkpoint. |
| `locomo-leaderboard-checkpoint.json` | same run | 39 MB, same 1986 rows. The checkpoint and v1 are the same data. |
| `locomo-leaderboard-comparison.md` | same run | Markdown report that was used to compute -14.1pp. Based on corrupt v1.json. |
| `bench-server.log` | Phase A bench server, Apr 18 20:01–20:36 | Shows server started on port 7779 serving `locomo_lb_conv_50` (1841 pts). Two embedding timeouts + one broken-pipe error during Phase A run. Shut down on signal 15 at 20:36:12, i.e. clean termination. |
| `baseline.log` | `bench_runner.py locomo --mode production`, Apr 17 20:37 | **The real baseline log.** Per-conv numbers match `../59c0a369...-locomo-production.json` exactly. |
| `phasea.log` | Phase A run, Apr 18 18:01–20:36 | Shows convs 26/30/41/42/43/44 SKIPPED ("all N QA done"), only 47/48/49/50 ran fresh under Phase A alphas. |
| `baseline_env.sh` / `phasea_env.sh` | Bench env snapshots | 5-line diff: Phase A sets `RERANK_RECENCY_ALPHA=0 RERANK_TEMPORAL_ALPHA=0 RERANK_PROOF_ALPHA=0.1`. Nothing else. |
| `qwen3_reranker.log` | Local Qwen3-Reranker-0.6B server (:9091) | Clean, no errors across both runs. |
| `check_skip_ingest.py` | forensic script, Apr 19 | Hashes per-conv predicted answers between checkpoint and v1 → all 10 byte-identical. |
| `recompute_scores.py` | forensic script | Recomputed 58.44% non-adv + per-conv breakdown from each JSON. |
| `verify_real_baseline.py` | forensic script | Scored `59c0a369...-locomo-production.json` → 72.53% non-adv matches baseline.log. |
| `diagnose_proof_norm.py` | diagnostic from earlier turn | Showed proof_norm saturated at 0.87–0.91 across windows/facts/inference → Phase A multiplicative boost ≈ no-op. |
| `inspect_checkpoint.py` | utility | Structure probe for the checkpoint JSON. |

## The real baseline lives at

**`../59c0a369b4296182cf69f11d74beda56e00e14eb-locomo-production.json`** (49.7 MB, Apr 17 20:37)

- Written by `bench_runner.py locomo --mode production` on commit `59c0a369`.
- **72.53% non-adv** across 10 convs × 1986 questions.
- Per-conv matches `baseline.log` exactly.
- This is the ground-truth baseline. **Do not quarantine.**

## Root cause in one sentence

`locomo_leaderboard_bench.py`'s `BENCH_CHECKPOINT` env var defaults to
`locomo-leaderboard-checkpoint.json`, and Phase A's env script did not override it — so the
raw harness picked up a months-old checkpoint written by a long-forgotten run and treated
its answers as valid prior work.

## How to not make this mistake again

Proposed new invariant for `RASPUTIN_SESSION_STATE.md` / `AGENTS.md`:

> **Benchmark artifact provenance invariant.** Before interpreting any leaderboard score,
> hash-check the artifact's per-conv predictions against the originating log's per-conv
> printouts. If they disagree for any conversation, the artifact is corrupt — do not trust
> the score. Every bench invocation must explicitly set `BENCH_CHECKPOINT=<unique-name>`
> keyed to the experiment identifier, never the default filename. Commit-hash-prefixed
> artifacts written by `bench_runner.py` are preferred for canonical baselines.

## What to do about Phase A

Parked. Permanently. For two independent reasons:

1. **It was not actually tested.** The 4 convs that did run Phase A alphas (47–50) showed
   −5 to −10pp vs baseline, but convs 26–44 showed much larger regressions that had
   nothing to do with Phase A.
2. **Even if tested, the signal would be dead.** `proof_norm` is saturated at pool size
   60; `proof_boost = 1 + 0.1 × (proof_norm − 0.5)` lies in ~[1.037, 1.041] across all
   candidates — approximately a constant multiplier. A constant multiplier cannot change
   relative ranking.

Phase A code is preserved in git stash: `phase-a-multiplicative-reranker-PARKED-2026-04-19-ghost-checkpoint-regression`.

## Next step (pending user go-ahead)

Rerun the bench (search-only via `--skip-ingest` against the already-populated Qdrant
collections) on commit `59c0a369` with Phase A reverted, against a NEW checkpoint name, to
reproduce the 72.5% baseline as **baseline_v2**. Expect ±1pp of the original. Then move to
Phase B / the bench-payload-structured-fields fix from a clean footing.
