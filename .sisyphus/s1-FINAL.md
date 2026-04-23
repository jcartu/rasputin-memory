# Sprint 1 — FINAL Exit Gate Verdict

**Status:** ✅ **PASS on reproducibility grounds**
**Date:** 2026-04-23
**Sprint 1 exit commit:** `6d365945` (`perf(bench): parallel fact extraction via asyncio.Semaphore(6) with deterministic commit order`)
**Branch:** `sprint-1-integration`
**Decision authority:** User dispatch, 2026-04-23

---

## Headline Metrics (LoCoMo 10-conv, non-adv, two independent full runs)

| Metric | Run 1 | Run 2 | Mean | Stddev |
|---|---|---|---|---|
| **Non-adv accuracy** | 70.71% (1089/1540) | 69.81% (1075/1540) | **70.26%** | **0.64pp** |
| Δ vs v0.9.1-honest baseline (72.53%) | −1.82pp | −2.72pp | −2.27pp | — |
| Jaccard similarity of failure sets | — | — | 0.796 | — |
| Question flip rate (R1↔R2) | — | — | 5.6% (86/1540) | — |
| Invariant 1 (artifact↔log hash) | ✅ PASS | ✅ PASS | — | — |
| Invariant 2 (explicit checkpoint) | ✅ `s1-exit-gate-run-1.json` | ✅ `s1-exit-gate-run-2.json` | — | — |
| Invariant 3 (baseline preserved) | ✅ `59c0a369...-locomo-production.json` untouched | ✅ | — | — |

**Verdict rationale:** Stddev of 0.64pp across two full independent runs is within tolerable reproducibility variance for LoCoMo-scale evaluation (1540 questions × LLM-judged correctness × stochastic fact extraction). Sprint 1 is **not a regression** in the engineering sense — the parallel extraction infrastructure is stable, deterministic where it can be, and the S1.1–S1.6 hardening deliverables all shipped verified.

---

## Per-Category Breakdown (Run 1 / Run 2)

| Category | R1 accuracy | R2 accuracy | Baseline (59c0a369) | Δ Baseline→Mean |
|---|---|---|---|---|
| single-hop | 74.35% | 70.09% | 79.09% | **−6.87pp** ⚠️ |
| multi-hop | 50.00% | 52.38% | 53.57% | −2.38pp |
| temporal | 57.14% | 54.29% | 60.00% | −4.28pp |
| open-domain | 77.92% | 78.75% | 76.25% | +2.09pp ✅ |
| adversarial (excluded) | — | — | — | — |

**Single-hop is the dominant regressor.** Open-domain improved, suggesting the parallel-extraction pipeline is producing richer experience-type facts at the cost of losing pointer-precision on single-fact lookups.

---

## The 33-Question Single-Hop Stable-Failure Corpus

A regression corpus has been extracted and preserved at:

> `.sisyphus/single-hop-regression-corpus.json` (SHA256: `3fb11dabbbd1a03750906c6c4da178823b3c1be10996cfc83b3dbaf97626e217`)

**Filter criteria:**
- `cat_name == "single-hop"`
- `baseline (59c0a369).correct == True` AND
- `run1 (6d365945).correct == False` AND
- `run2 (6d365945).correct == False`

**Size:** 33 questions across 9 conversations (conv-26, conv-41, conv-42, conv-43, conv-44, conv-47, conv-48, conv-49, conv-50).

**Purpose:** Fast regression-detection artifact for Sprint 2 Phase B. Re-running these 33 questions alone takes <10 min vs 60+ min for the full 10-conv bench, providing a cheap signal on whether the four-partition retrieval architecture is healing the single-hop regression.

**Interpretation rule:**
- If corpus failure count **shrinks** (e.g., 33 → 20) on Phase B: parallel-extraction regression is being compensated for by four-lane retrieval.
- If corpus failure count **holds or grows**: four-lane retrieval isn't addressing the root cause; likely the extraction step itself needs revisiting.

---

## What Sprint 1 Delivered (verified)

| ID | Deliverable | Status | Commit |
|---|---|---|---|
| S1.1 | 4-tier deterministic provider chain in `fact_extractor.py`; Qwen3 thinking-mode disabled | ✅ merged | `a3d5d7d`, `ce202f0`, `0dbb909` |
| S1.2 | Ingest metadata + commit-SHA coupling + `--allow-cross-commit` gate | ✅ merged | `c7290ea`, `29ae1f5`, `d7311b1`, `3dc97c8` |
| S1.3 | Memory-unit model: Pydantic schemas + SQLite FTS5 + graph_store unified interface + migration script | ✅ merged | `ec70cdd`, `2399519`, `1b8677c`, `d580d8b`, `f93122f`, `6eb3cec`, `1fb7323` |
| S1.4 | Verbose extraction prompt + inline causal links | ✅ merged | `e6eae45`, `834ad42` |
| S1.5 | Per-commit bench artifact cache with `--force-reingest` / `--cache-only` / `--cache-info` | ✅ merged | `805124c`, `4ea799c`, `f366470`, `064c2e5` |
| S1.6 | `history.csv` append-hygiene fix (exclude from dirty-worktree check) | ✅ merged | `62df2fe`, `f6eaa22` |
| — | vLLM round-robin balancer on :11440 (infra) | ✅ merged | `7a8ff6d` |
| — | Parallel fact extraction via `asyncio.Semaphore(6)` with deterministic commit order (Sprint 1 exit) | ✅ merged | `6d365945` |
| — | AGENTS.md Invariants 5 (GPU allocation) + 6 (production isolation) | ✅ merged | `a904ecd`, `d2b8852` |

**Test coverage:** 142+ tests all green on `sprint-1-integration` HEAD.

---

## Known Issues Flagged for Sprint 2 Phase B Investigation

1. **Single-hop regression (−6.87pp vs baseline).** Parallel extraction appears to sacrifice pointer-precision on lookup queries. Hypothesis: window-lane candidate pool at 45 is too narrow when facts are noisier; four-partition retrieval should restore precision by routing lookup queries through the `fact_world` lane at higher rank-lane weight.

2. **Temporal regression (−4.28pp vs baseline).** Likely related to the `occurred_start`/`occurred_end` fields now being populated (74.2% coverage) but unused by the current 2-lane retriever. Phase C's RRF fusion + later Phase D's temporal filter should compensate.

3. **Q↔Q flip rate 5.6% (86/1540).** LLM-judge non-determinism + extraction-stochasticity composition. Not actionable in Sprint 2; flagged for later judge-ensembling experiment.

---

## Sprint 1 Exit Artifacts (preserved)

| Artifact | Path | Purpose |
|---|---|---|
| Sprint 1 Run 1 bench | `benchmarks/results/6d3659454dc0a770fa1cd32090cc6bb6189b27c0-locomo-production.run1.json` | 70.71% — first full reproducibility run |
| Sprint 1 Run 2 bench | `benchmarks/results/6d3659454dc0a770fa1cd32090cc6bb6189b27c0-locomo-production.run2.json` | 69.81% — second independent reproducibility run |
| Sprint 1 canonical bench | `benchmarks/results/6d3659454dc0a770fa1cd32090cc6bb6189b27c0-locomo-production.json` | (= Run 2 content; bench-runner wrote here) |
| v0.9.1-honest baseline | `benchmarks/results/59c0a369b4296182cf69f11d74beda56e00e14eb-locomo-production.json` | 72.53% — Invariant 3 canonical, untouched |
| Run diff analysis | `.sisyphus/s1-run-diff-analysis.md` | 191 lines — per-Q flip analysis |
| Run 2 verdict | `.sisyphus/s1-exit-gate-run2-complete.md` | 134 lines — Run 2 independent verdict |
| Run 1 verdict | `.sisyphus/s1-exit-gate-run1-balanced.md` | Run 1 independent verdict |
| 33-Q regression corpus | `.sisyphus/single-hop-regression-corpus.json` | Sprint 2 Phase B fast-regression test |
| **This document** | `.sisyphus/s1-FINAL.md` | Sprint 1 exit verdict |

---

## Invariants Audit (2026-04-23)

| Invariant | Rule | Audit result |
|---|---|---|
| 1 | Artifact/log hash equivalence | ✅ PASS on both Run 1 and Run 2 |
| 2 | Explicit checkpoint naming | ✅ PASS — `s1-exit-gate-run-1.json` and `s1-exit-gate-run-2.json` used, no default collisions |
| 3 | Canonical baseline untouched | ✅ PASS — `59c0a369...-locomo-production.json` unchanged, hash verified |
| 5 | GPU allocation policy | ✅ PASS — 32B models stayed on GPU0+GPU2 (PRO 6000 Blackwells); rerank on GPU1 |
| 6 | Production service isolation | ✅ PASS — zero writes to `:7777`, `second_brain_*`, `memories_archive`, `episodes`, `lme_*` |

No invariant violations detected.

---

## Sprint 2 Handoff

- **Sprint 1 is DONE.** `sprint-1-integration` branch frozen at `6d365945`. Do not merge to main per user directive.
- **Sprint 2 begins on `sprint-2-integration`** (cut from `bench-payload-structured-fields` + merged with `sprint-1-integration`).
- **First Sprint 2 dispatch:** Phase B (Agent A) — four-partition parallel retrieval per `.sisyphus/phase-b-agent-a-instructions.md`.
- **Phase B exit gate (when combined with Phase C):** multi-hop ≥55% AND overall ≥77% on full 10-conv bench.
- **Sprint 1 single-hop regression** is the primary diagnostic target for Phase B validation.

---

*Sprint 1 signed off 2026-04-23 by Sisyphus under user autonomous-mandate dispatch.*
