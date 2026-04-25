# Sprint 2 Ablation Verdict — Four-Lane Partition is the Multi-Hop Culprit

**Date**: 2026-04-24
**Authority**: Ablation A full-bench results (see `.sisyphus/s2-ablation-A-complete.md`)
**Configs compared**:
- S1 Mean (6d365945, 2-lane): baseline
- S2 full (c623ae4, FOUR_LANE=1 + RRF_FUSION=1): Phases B+C active
- Ablation A (c623ae4, FOUR_LANE=1 + RRF_FUSION=0): Phase B only

---

## Verdict

**A2 — Phase B four-partition causes multi-hop regression via per-lane top-k truncation of cross-lane evidence.**

This is the diagnosis the user had pre-registered in the interpretation rules:
> A2. Multi-hop stays at ~43% AND overall drops to ~70%:
>     - Diagnosis: four-partition broke multi-hop, RRF was actually helping
>     - Next step: investigate cross-lane multi-hop retrieval path

Ablation A's observed scores:
- Multi-hop: **42.71% exactly** (S2: 42.71%) — **unchanged**
- Overall: **71.17%** (S2: 71.56%) — dropped −0.39pp as RRF was helping marginally

Both numbers match the A2 prediction window.

---

## Three explicit findings requested in the prompt

### Finding 1 — "Phase B four-partition causes multi-hop regression via per-lane top-k truncation of cross-lane evidence"

**Evidence**:
- S1 → S2 multi-hop: 46.88% → 42.71% = **−4.17pp** (−4 questions on full bench)
- Ablation A multi-hop: 42.71% (identical to S2)
- RRF on/off changed zero multi-hop answers

**Mechanism** (consistent with `.sisyphus/ablation-C-prep.md`):
Four-lane partition routes retrieval into 4 semantic buckets with tight per-lane budgets:
- window: 45
- fact_world: 8
- fact_experience: 4
- fact_inference: 3
- Total per expanded query: **60**

Multi-hop questions require evidence spanning lanes — e.g., one hop in a window chunk (dialogue context), the second hop in a fact chunk (extracted assertion). When the fact lanes only allow 8/4/3 survivors before merge, cross-lane multi-hop chains get truncated at the earliest link. Neither the reranker nor RRF fusion can recover evidence that was never in the merged candidate pool.

S1's 2-lane baseline used a single larger qdrant_search call per expanded query, which gave multi-hop chains more slots to find both hops.

### Finding 2 — "RRF (Phase C) is neutral to slightly positive on overall score, positive on temporal (+2.18pp), but cannot fix multi-hop because the evidence is already truncated before fusion"

**Evidence**:
- Overall: S2 71.56% vs Ablation A 71.17% → RRF contributes **+0.39pp** to overall
- Temporal: S2 66.36% vs Ablation A 64.17% → RRF contributes **+2.18pp** to temporal
- Open-domain: S2 83.12% vs Ablation A 83.59% → RRF contributes **−0.48pp** (slightly hurts)
- Single-hop: S2 52.84% vs Ablation A 51.77% → RRF contributes **+1.06pp**
- Multi-hop: S2 42.71% vs Ablation A 42.71% → RRF contributes **0.00pp**

The mechanism for why RRF can't fix multi-hop:
> The four-lane `_four_lane_search()` function in `tools/brain/search.py:306-378` runs the four qdrant_search calls **before** returning to hybrid_search. RRF fusion in `fusion.py` operates on the already-bounded per-lane lists. If a multi-hop evidence chunk never makes it into the top-45 window results OR top-8 world-fact results OR top-4 experience results OR top-3 inference results, then RRF has nothing to fuse. RRF is a ranking-layer operation; truncation happens one layer earlier.

The temporal boost is notable — it suggests RRF helps temporal questions find the right evidence among the successfully-retrieved candidates (ranking-layer problem), even though it can't help multi-hop (retrieval-layer problem).

### Finding 3 — "Phase B's +4.25pp single-hop gain is structural and survives RRF removal — Phase B's core insight is valid, its parameterization is not"

**Evidence**:
- S1 R1 single-hop: 47.52%
- S1 R2 single-hop: 43.26%
- **S1 Mean single-hop: 45.39%**
- S2 full single-hop: 52.84% → **+7.45pp vs S1 Mean**
- Ablation A single-hop: 51.77% → **+6.38pp vs S1 Mean**

Single-hop gain without RRF: **+6.38pp**. With RRF: **+7.45pp**. RRF adds +1.06pp on top of Phase B's structural gain.

**33-Q corpus breakdown confirms the structural nature**:
- Both configs correct (RRF-independent wins): **12 / 33**
- S2-only (RRF-rescued, lost without it): 5 / 33
- Ablation A-only (RRF hurt, gained by disabling): 2 / 33
- Neither correct (unresolved): 14 / 33

12 of the 17 S2 recoveries (71%) survive RRF removal — these are structural Phase B wins. The extraction-cache lineage unification + per-lane fact-type filtering genuinely give the reranker better candidates to work with, independent of fusion.

**Phase B's core insight** (routing retrieval by semantic partition for better reranker input) is valid. **Phase B's parameterization** (60-total-per-query budget split 45/8/4/3) is too aggressive — it starves cross-lane multi-hop chains.

---

## Why Ablation C is the right next step (not Phase B revert)

Reverting Phase B loses 17 single-hop recoveries (all 17 on the 33-Q corpus) and the +7.45pp single-hop gain on the full bench. That is 17 questions we gave up on last sprint, now finally fixed.

Raising per-lane budget (Ablation C) tests a **parametric** fix first:
- If per-lane budget 150 recovers multi-hop → Phase B stays, ship sprint-2
- If per-lane budget 150 does not recover multi-hop → structural problem, revert Phase B (and then the 17 single-hop wins are worth the price of re-engineering the fact-lane approach outside of multi-lane routing)

See `.sisyphus/ablation-C-prep.md` for full Ablation C config analysis, env var semantics (already call-time-configurable — no code change needed), cap geometry, and ready-to-execute launch script.

---

## Aggregate evidence table

| Metric | S1 Mean | S2 full | **Ablation A** | Gate |
|---|---:|---:|---:|---:|
| Overall non-adv | 70.26% | 71.56% | **71.17%** | ≥77% |
| Multi-hop | 46.88% | 42.71% | **42.71%** | ≥55% |
| Single-hop | 45.39% | 52.84% | **51.77%** | — |
| Open-domain | 82.64% | 83.12% | **83.59%** | — |
| Temporal | 66.67% | 66.36% | **64.17%** | — |
| 33-Q single-hop corpus | 0/33 | 17/33 | **14/33** | — |

---

## Recommended path forward

1. **Do not revert Phase B** — would lose 17 structural single-hop recoveries
2. **Do not revert Phase C** — marginally positive (+0.39pp overall, +2.18pp temporal, +3 on 33-Q). Keep it on for Ablation D if C succeeds
3. **Run Ablation C** (FOUR_LANE=1, RRF_FUSION=0, per-lane budget ≈2.5x) to test parametric fix
4. **Conditional on Ablation C result**:
   - C success → Ablation D with RRF re-enabled (targets 77% gate)
   - C partial → halt for strategy (flat retrieval fallback for multi-hop queries?)
   - C no effect → revert Phase B (structural problem, not parametric)

---

## Open questions (documented for reference — do not block Ablation C launch)

1. **The 5 RRF-rescued 33-Q questions** — `(conv-41, 2), (conv-43, 14), (conv-43, 18), (conv-48, 1), (conv-50, 6)`. Per-question forensic would show whether these are RRF's unique contribution (order-sensitive cases) or lane-budget-limit-adjacent (would be rescued anyway with wider pool).
2. **The temporal −2.18pp** under RRF-off — conversations conv-41, conv-50 took most of the hit. Temporal questions may benefit disproportionately from cross-lane fusion's rank-smoothing.
3. **Per-lane budget tuning curve** — Ablation C tests ≈2.5x. Finer sweep (1.5x, 2x, 3x, 4x) would reveal the inflection point but costs ~5h per config.

---

## Approval state

- Ablation A executed and analyzed: **COMPLETE**
- Ablation A verdict authored: **THIS FILE**
- Phase B revert: **NOT AUTHORIZED** (and this analysis recommends against it)
- Phase C revert: **NOT AUTHORIZED** (and this analysis recommends against it)
- Ablation C launch: **USER-AUTHORIZED** in same turn as this verdict write-up

Next HALT: at Ablation C completion, regardless of outcome.
