# Ablation D — Prep (PROPOSAL, NOT LAUNCHED)

**Status**: Awaiting user authorization. Do NOT launch without explicit `EXECUTE` from user.
**Date drafted**: 2026-04-24, post-Ablation-C completion
**Supersedes**: N/A (first D proposal)
**Triggered by**: `.sisyphus/s2-ablation-C-success.md` functional C1 verdict

---

## Hypothesis

Re-enabling RRF (Phase C) on top of the Ablation C wider lane budget (Phase B with correct parameterization) will compound two independent gains:

1. **RRF's +2.18pp temporal boost** (observed in Ablation A when toggling RRF off vs S2 full; confirms RRF specifically helps temporal ranking)
2. **Wider budget's +5.21pp multi-hop recovery** (observed in Ablation C)

Both effects are additive in principle because they operate at different pipeline stages:
- **Lane budget** fixes what evidence enters fusion (per-lane pool size)
- **RRF** fixes how evidence is ranked after fusion (reciprocal-rank blending)

With multi-hop already repaired by the wider budget (Ablation C), the question is whether RRF can still contribute its temporal boost without re-breaking multi-hop (since wider budgets mean RRF no longer operates on truncated evidence).

---

## Expected Outcome

| Category | Ablation C (baseline for D) | D prediction | Confidence | Rationale |
|---|---:|---:|---|---|
| Overall | 72.40% | **73.5 – 74.0%** | medium-high | +1.1 – 1.6pp compound from RRF temporal effect |
| Multi-hop | 47.92% | 47.0 – 48.5% | medium | Expected stable; RRF cannot regress already-sufficient evidence pool |
| Single-hop | 52.84% | 52.0 – 53.5% | high | Stable; single-hop was unaffected by RRF in A/S2 comparison |
| Open-domain | 83.59% | 83.0 – 84.0% | high | Stable |
| Temporal | 67.60% | **69.5 – 70.5%** | medium-high | Primary RRF beneficiary (+2.18pp observed in A→S2 direction) |

**Target zone**: Overall ~73.5–74%, clearing the original C1 threshold, with multi-hop stable at ~48% and temporal lifted to ~70%.

### Success / Failure criteria

- **D1 (Full Success)**: Overall ≥73% AND multi-hop ≥46% AND temporal ≥69%. This validates RRF + budget as the final Sprint 2 config. Ship this.
- **D2 (Partial)**: Overall 72.5–73% OR multi-hop 45–46% OR temporal 67–69%. RRF helps but compound is weaker than predicted. Ship but with muted expectations.
- **D3 (Neutral/Negative)**: Overall ≤72.4% (no gain vs C) OR multi-hop drops below 46%. RRF + wider budget does not compound; revert to Ablation C config as final. Unexpected but possible if RRF introduces noise the narrower budget was absorbing.

---

## Configuration

### Env vars (ALL required)

```bash
# Enable Phase B (four-lane partitioning) — same as C
export FOUR_LANE=1

# Enable Phase C (RRF fusion) — KEY DIFFERENCE from C
export RRF_FUSION=1

# Wider lane budget — same as C (proportional 2.5× scale)
export BENCH_LANE_WINDOWS=113    # Phase B windows lane (≈ 75% of 151)
export BENCH_LANE_FACT_W=20      # Facts-why lane     (≈ 13% of 151)
export BENCH_LANE_FACT_E=10      # Facts-event lane   (≈  7% of 151)
export BENCH_LANE_FACT_I=8       # Facts-identity lane(≈  5% of 151)

# Infra
export BENCH_API_PORT=7779
export BENCH_CHECKPOINT=s2-ablation-D-rrf-plus-wide.json
```

Total per expanded query: **151** (same as C). Proportional lane weighting preserved.

### Delta from Ablation C

Only ONE bit flips: `RRF_FUSION=0` → `RRF_FUSION=1`. Everything else identical. This is a clean A/B on the RRF contribution at the wider-budget operating point.

### NOT changed (explicit non-delta list)

- Commit: c623ae4f (sprint-2-integration, Phase B + Phase C code present, only env controls activation)
- Embedding provider: Qwen3-Embedding-8B on GPU0:8011
- Rerank: Qwen3-Reranker-0.6B on GPU1:9091
- Fact extraction: local vLLM Qwen3-32B-AWQ via balancer :11440 (A/B round-robin)
- Cache: all 10 `locomo_lb_conv_*` collections reused via `--skip-ingest --allow-cross-commit`
- Mode: `production` (Haiku answers + strict judge)
- Benchmark harness: `bench_runner.py locomo --mode production`

---

## Pre-flight Checklist (per standing policy)

Executed as part of D launch if authorized. Documented here so the launch script doesn't surprise the user:

1. **Service restart** (prophylactic, per user standing policy + Ablation A runtime-anomaly note):
   - Kill + relaunch `vllm-balancer` tmux session (port :11440)
   - Kill + relaunch `vllm-rerank` tmux session (port :9091)
   - Kill + relaunch `vllm-embed` tmux session (port :8011)
   - Do NOT touch `vllm-extract-A` (:11437), `vllm-extract-B` (:11439), qdrant, FalkorDB, production :7777
2. **4-endpoint health verification**:
   - `curl -sf http://127.0.0.1:11440/v1/models` → 200
   - `curl -sf http://127.0.0.1:8011/v1/models` → 200 (or equivalent)
   - `curl -sf http://127.0.0.1:9091/health` → 200 or functional POST
   - Bench API on :7779 will come up during bench; verified post-launch via `/proc/<pid>/environ`
3. **Cache integrity canary**:
   - Snapshot `locomo_lb_conv_*` point counts to `/tmp/bench_runs/s2-ablation-D/pre_launch_point_counts.json`
   - Compare post-launch to detect any accidental re-ingestion
4. **Artifact preservation**:
   - Backup current `c623ae4f...-locomo-production.json` (S2 canonical) → `c623ae4f...-locomo-production.s2-full.prelaunch-D.json`
   - Backup `history.csv` → `history.csv.pre-ablation-D.bak`
5. **Env verification (end-to-end)**:
   - After launch, check `/proc/<bench_runner_pid>/environ` and `/proc/<api_server_pid>/environ` for all 8 env vars
   - If any missing, HALT and investigate

---

## Launch Script (DRAFT, NOT EXECUTED)

Will live at `/tmp/bench_runs/s2-ablation-D/launch.sh`:

```bash
#!/bin/bash
# Ablation D — FOUR_LANE=1, RRF_FUSION=1, wider lane budget (same as C)
# Hypothesis: RRF adds +2.18pp temporal on top of C's multi-hop recovery
# Target: overall ≥73%, multi-hop ≥46%, temporal ≥69%
# Prep: .sisyphus/ablation-D-prep.md
# Triggered by: .sisyphus/s2-ablation-C-success.md (functional C1)

set -euo pipefail
source /tmp/bench_runs/baseline_env.sh

export FOUR_LANE=1
export RRF_FUSION=1                      # THE ONLY DELTA vs Ablation C
export BENCH_LANE_WINDOWS=113
export BENCH_LANE_FACT_W=20
export BENCH_LANE_FACT_E=10
export BENCH_LANE_FACT_I=8
export BENCH_API_PORT=7779
export BENCH_CHECKPOINT=s2-ablation-D-rrf-plus-wide.json

cd /home/josh/.openclaw/workspace/rasputin-memory-sprint2
date -Iseconds > /tmp/bench_runs/s2-ablation-D/start_time.txt

exec python3 benchmarks/bench_runner.py locomo \
  --mode production \
  --skip-ingest \
  --allow-cross-commit \
  --force
```

---

## Runtime Expectations

| Scenario | Duration | Reasoning |
|---|---|---|
| Best case | 5h 40m | Matches S2 (5h 50m) and Ablation C (5h 47m) pace |
| Typical | 6 – 7h | Accounts for Haiku API variability |
| Hard halt | 24h | Standing policy; never exceeded in practice |

**Expected completion** (if launched immediately after authorization): ~6h after launch timestamp.

---

## Decision Tree Post-D

Writing this ahead of time to keep post-hoc framing honest:

### If D1 (overall ≥73%, multi-hop ≥46%, temporal ≥69%)

- **Ship config**: `FOUR_LANE=1 RRF_FUSION=1 BENCH_LANE_WINDOWS=113 BENCH_LANE_FACT_W=20 BENCH_LANE_FACT_E=10 BENCH_LANE_FACT_I=8`
- **Commit to sprint-2-integration**: make these env defaults in `config/rasputin.toml` or code (user decides)
- **Sprint 2 marked as shipped with reduced scope**: overall ~73.5–74% (vs 77% exit gate). Honest framing per Real Talk section in C success doc.
- **Write**: `.sisyphus/s2-ablation-D-success.md` + `.sisyphus/sprint-2-closeout.md`

### If D2 (partial)

- **Ship Ablation C config** (no RRF): C's 72.40% is functionally equivalent and simpler. RRF-on adds complexity without the full expected gain.
- **Write**: `.sisyphus/s2-ablation-D-partial.md`
- **Flag for Phase D work**: temporal boost did not materialize as expected; may need distinct temporal-routing logic rather than RRF.

### If D3 (negative/neutral)

- **Ship Ablation C config** (no RRF): C is strictly better. RRF was broken by wider budget (possibly due to RRF's secondary 120-item cap at `search.py:443` interacting badly with the 151 pre-fusion pool).
- **Write**: `.sisyphus/s2-ablation-D-no-effect.md`
- **Investigation task**: audit the `min(limit * 2 + 60, 120)` cap in search.py:443. If RRF's cap is the bottleneck, tune or remove it as a follow-up experiment (Ablation E).

---

## Risks and Open Questions

1. **RRF's secondary 120-item cap at `tools/brain/search.py:443`**: `min(limit * 2 + 60, 120)` applies only when `RRF_FUSION=1`. With wider lane budgets feeding 151 candidates into fusion, this cap may truncate before RRF's benefit is realized. If D underperforms predictions, this is the first place to look. *Mitigation*: Ablation E would remove this cap and re-test. Not a reason to block D; the cap has been present in all RRF-on configs including S2 full, so D at minimum reproduces S2's RRF behavior at a wider budget.

2. **Compound prediction may be optimistic**: +2.18pp (RRF temporal) and +5.21pp (budget multi-hop) measured in isolation may not simply add. Interaction effects are possible. Confidence is medium, not high.

3. **Runtime**: no known issues, but if services degrade during the run (as originally suspected in Ablation A before reconciliation), hourly heartbeats will catch it. Standing 24h hard halt applies.

4. **What if D2/D3?** — this was flagged in the C success doc's "Real Talk" section: Sprint 2 may ship at ~72–73% overall rather than the 77% exit gate. The project path forward (Phase D temporal work, Phase E answer pipeline) is still clear regardless of D outcome.

---

## Non-Negotiable Constraints (from user's standing policy)

- **Do NOT launch D without explicit user EXECUTE authorization.** This document is prep only.
- **Service restart is the only autonomous infra action** during D run.
- **HALT at D completion regardless of outcome.** Write matching verdict doc, then stop.
- **Do NOT modify Phase B or Phase C code** based on D result autonomously. Any code change requires user approval.
- **Do NOT push to remote** without explicit user request.
- **24h hard halt** on wall time.

---

## Open Authorization Questions for User

Before launching D, the user may want to decide:

1. **Launch D now or accept Ablation C as final Sprint 2 config?** C is a clean functional C1 success. D is projected to add ~1.1–1.6pp overall plus temporal boost. If the user prioritizes shipping simpler (C-only) over maximum score, skip D.

2. **If D1 (success), do we also want Ablation E** (remove RRF's 120-item cap) to push further? Risk is low; expected upside is modest.

3. **Commit strategy for lane-budget change**: the wider budget is currently env-var-driven (runtime). Should the default values in code (`tools/brain/search.py:321-324` reads env with presumed defaults) change? Or should the env vars ship in a run-script / deployment config? User decision.

---

## Cross-references

- `.sisyphus/s2-ablation-C-success.md` — triggers this prep
- `.sisyphus/s2-ablation-verdict-fourlane-culprit.md` — original Ablation C authorization chain
- `.sisyphus/ablation-C-prep.md` — budget interpretation rationale (carried over)
- `.sisyphus/active-sprint-status.md` — Sprint 2 status tracking (updated alongside this doc)
- `tools/brain/search.py:306-458` — four-lane and RRF code paths (no modification needed)
- `benchmarks/bench_runner.py` — harness (no modification needed)
