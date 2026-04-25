# Ablation A — Complete Results

**Config**: `FOUR_LANE=1 RRF_FUSION=0` on commit `c623ae4`
**Purpose**: Isolate whether Phase C (RRF fusion) or Phase B (four-lane partition) caused the Sprint 2 multi-hop regression.
**Started**: 2026-04-23T20:51:00 (+03:00)
**Completed**: 2026-04-24T02:27:00 (+03:00) — artifact mtime (checkpoint writes by leaderboard_bench)
**Wall time**: 5h 36m of *benchmark-reported* runtime — but processes lingered showing ~17h 14m elapsed (see runtime anomaly note)
**Artifact**: `benchmarks/results/c623ae4f658a9100cbee5c7fb7a5b41dc6fccfaa-locomo-production.ablation-A.json`
**Checkpoint**: `benchmarks/results/s2-ablation-A-four-lane-only.json`
**MD5** (both, identical): `99b5afab865b702a40f145ec080a1640`

---

## Invariant 1: PASS

Per-conv accuracy match between bench log and artifact — all 10 conversations.

| Conv | Log % | Log Qs | Artifact % | Artifact Qs | Match |
|---|---:|---:|---:|---:|:---:|
| conv-26 | 72.4 | 152 | 72.4 | 152 | ✅ |
| conv-30 | 77.8 | 81 | 77.8 | 81 | ✅ |
| conv-41 | 77.6 | 152 | 77.6 | 152 | ✅ |
| conv-42 | 67.8 | 199 | 67.8 | 199 | ✅ |
| conv-43 | 69.1 | 178 | 69.1 | 178 | ✅ |
| conv-44 | 69.9 | 123 | 69.9 | 123 | ✅ |
| conv-47 | 72.0 | 150 | 72.0 | 150 | ✅ |
| conv-48 | 73.8 | 191 | 73.8 | 191 | ✅ |
| conv-49 | 67.9 | 156 | 67.9 | 156 | ✅ |
| conv-50 | 67.1 | 158 | 67.1 | 158 | ✅ |

---

## Full-bench non-adversarial score

**Ablation A: 1096 / 1540 = 71.17%**

### Category breakdown

| Category | S2 full | **Ablation A** | Δ (A vs S2) | S1 Mean | Δ (A vs S1) |
|---|---:|---:|---:|---:|---:|
| Multi-hop | 41/96 = 42.71% | **41/96 = 42.71%** | **+0.00pp** | 46.88% | −4.17pp |
| Single-hop | 149/282 = 52.84% | **146/282 = 51.77%** | −1.06pp | 45.39% | **+6.38pp** |
| Open-domain | 699/841 = 83.12% | **703/841 = 83.59%** | +0.48pp | 82.64% | +0.95pp |
| Temporal | 213/321 = 66.36% | **206/321 = 64.17%** | **−2.18pp** | 66.67% | −2.49pp |
| **OVERALL** | 1102/1540 = 71.56% | **1096/1540 = 71.17%** | **−0.39pp** | 70.26% | +0.91pp |

### Per-conv (Ablation A vs S2)

| Conv | S2 | A | Δ |
|---|---:|---:|---:|
| conv-26 | 73.0% | 72.4% | −0.66pp |
| conv-30 | 77.8% | 77.8% | +0.00pp |
| conv-41 | 79.6% | 77.6% | −1.97pp |
| conv-42 | 68.3% | 67.8% | −0.50pp |
| conv-43 | 68.0% | 69.1% | **+1.12pp** |
| conv-44 | 68.3% | 69.9% | **+1.63pp** |
| conv-47 | 72.7% | 72.0% | −0.67pp |
| conv-48 | 73.8% | 73.8% | +0.00pp |
| conv-49 | 67.9% | 67.9% | +0.00pp |
| conv-50 | 69.6% | 67.1% | −2.53pp |

Movement spread: −2.53 to +1.63 — mostly noise-level. Only conv-41 and conv-50 moved meaningfully.

---

## 33-Q single-hop stable-failure corpus

| Config | Recovered | |
|---|---:|---|
| v0.9.1-honest baseline | 33/33 (corpus definition) | — |
| S1 R1 | 0/33 | corpus criterion — both S1 runs had to fail |
| S1 R2 | 0/33 | corpus criterion |
| **S2 full (FOUR_LANE + RRF)** | **17/33 = 51.5%** | |
| **Ablation A (FOUR_LANE, no RRF)** | **14/33 = 42.4%** | net −3 vs S2 |

### Per-question RRF effect on the 33-Q corpus

| Pattern | Count | Interpretation |
|---|---:|---|
| Both correct (structural win) | 12 | Phase B partition itself rescued these — RRF-independent |
| S2 only (RRF rescued, lost without it) | 5 | `(conv-41, 2), (conv-43, 14), (conv-43, 18), (conv-48, 1), (conv-50, 6)` |
| Ablation A only (RRF hurt) | 2 | `(conv-47, 54), (conv-50, 11)` |
| Neither correct (unresolved) | 14 | Retrieval-pool misses regardless of config |

**Net RRF effect on 33-Q corpus: +3 questions** (rescued 5, hurt 2). This is where Phase C's marginal positive contribution comes from — and also why the 5 "RRF-rescued" questions on conv-41 and conv-50 explain most of those two convs' Δ−1.97 / Δ−2.53 in Ablation A.

---

## Sprint 2 exit-gate comparison

| Gate | Threshold | S2 full | Ablation A | Status |
|---|---:|---:|---:|:---:|
| Overall non-adv | ≥77% | 71.56% | **71.17%** | ❌ FAIL both |
| Multi-hop | ≥55% | 42.71% | **42.71%** | ❌ FAIL both (identical) |

Neither config passes the gate. The primary blocker is multi-hop, and it is unaffected by RRF.

---

## Diagnosis

**A2 outcome**: Four-partition (Phase B) is the multi-hop culprit; RRF (Phase C) had no effect on multi-hop.

**The decisive signal is identity**, not a magnitude delta:
> Multi-hop = 41/96 (42.71%) in BOTH S2 full AND Ablation A. Turning off RRF changed the answer for exactly zero multi-hop questions.

If RRF were the culprit, disabling it should have recovered some of the S1→S2 multi-hop regression (−4.17pp, 4 questions). It recovered zero.

See `.sisyphus/s2-ablation-verdict-fourlane-culprit.md` for full reasoning and next-step analysis.

---

## Artifacts preserved

- `c623ae4f...-locomo-production.json` — **S2 full** (original, mtime 02:27 when bench-runner attempted to write, but MD5 identical to backup → not overwritten)
- `c623ae4f...-locomo-production.s2-full.json` — explicit S2 backup (MD5 `ddce8d750127f2a1dd6b1508e79d57dc`)
- `c623ae4f...-locomo-production.ablation-A.json` — **NEW Ablation A** (MD5 `99b5afab865b702a40f145ec080a1640`)
- `s2-ablation-A-four-lane-only.json` — checkpoint (identical MD5 to ablation-A.json)
- `history.csv` — NOT duplicated (only one c623ae4f row, from S2 launch)
- `history.csv.pre-ablation-A.bak` — pre-launch snapshot (preserved)

**Artifact overwrite risk did NOT materialize** — `bench_runner.py --force` appears to skip the result-file write when the file already exists with the same content, or the write happens through the checkpoint path only. Either way, S2 is intact.

---

## Services + cache

- All 10 locomo_lb_conv_* collections: point counts unchanged from pre-launch snapshot
- No re-ingest log lines observed
- Same 6d365945 extraction cache lineage as S2 full

---

**Related docs**:
- Verdict: `.sisyphus/s2-ablation-verdict-fourlane-culprit.md`
- Ablation C prep: `.sisyphus/ablation-C-prep.md` (written during A run, ready to execute)
- Runtime anomaly: `.sisyphus/ablation-A-runtime-anomaly.md`
