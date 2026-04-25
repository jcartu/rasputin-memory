# RASPUTIN Memory — Project Wind-Down

**Date:** 2026-04-25
**Final commit:** `29bbcb7` on `sprint-2-integration`
**Final benchmark:** 72.40% LoCoMo non-adv (full 10-conversation, 1540 questions)
**Status:** Archived as a public research artifact.

---

## 1. Closure rationale

RASPUTIN was a research project exploring fact-type partitioned retrieval for LLM agent memory, evaluated against the LoCoMo conversational-memory benchmark.

The project closed at Sprint 2 with the following honest comparison:

| System | LoCoMo non-adv |
|---|---|
| RASPUTIN v0.9.1 baseline | 72.53% |
| RASPUTIN Sprint 1 (mean of three runs) | 70.26% |
| **RASPUTIN Sprint 2 final (Ablation C)** | **72.40%** |
| Hindsight (Gemini-3, reference SOTA) | **89.61%** |

A 17-percentage-point gap separates RASPUTIN from the current state of the art on the benchmark it was chasing.

### Why stop now

- **Sprint 3 plan** (Phase D temporal lane integration + per-category prompt routing) projected a +3 to +5 percentage-point ceiling at a cost of 9–11 additional working days. Best-case post-Sprint-3 score: ~75–77%, still 12+ points behind Hindsight.

- **Sprint 4 plan** (cloud answer-model swap to claude-opus-4-7 or gemini-3-pro plus reranker A/B) projected a ceiling of ~80–87% over an additional ~5 weeks of work and additional cloud spend. Even at the optimistic end, RASPUTIN would not reach Hindsight parity.

- **Hindsight is open source, MIT licensed, independently validated, and already shipped.** Continuing to invest in RASPUTIN to close the gap is sunk-cost reasoning — Hindsight already won the benchmark RASPUTIN was chasing.

- The **research value** of RASPUTIN is the architectural insights it surfaced (see Section 4), not the leaderboard position. Those insights are documented here and in the Sprint verdict files (`.sisyphus/sprint-1-verdict.md`, `.sisyphus/sprint-2-verdict.md`).

### Decision

Ship the codebase as a public research artifact at Sprint 2 final state. Document the architecture, the methodology, and the insights. Stop development. Archive the GitHub repository.

---

## 2. What ships

The repository is published at `sprint-2-integration` HEAD `29bbcb7` with:

### Code
- **Phase B four-partition retrieval** with baked Ablation C lane budget defaults (113 windows, 20 facts default, 10 entities, 8 events).
- `FOUR_LANE` defaults to `ON`. `RRF_FUSION` defaults to `OFF` (Sprint 2 found RRF and wider lane budgets are budget-dependent alternatives, not additive — see Section 4).
- A-MAC commit-time quality gate (LLM-judged importance scoring before write).
- Two-lane hybrid search (window + fact lanes) with Qwen3-Reranker-0.6B reranking.
- Optional FalkorDB graph layer (parked for bench, present in code).
- HTTP API server on `:7777`.
- MCP server on `:8808` as a thin HTTP proxy.

### Research artifacts (embedded as `.sisyphus/` documentation)
- `sprint-1-verdict.md` — Sprint 1 closure (three full LoCoMo runs, mean 70.26%, ROLLBACK verdict).
- `sprint-2-verdict.md` — Sprint 2 closure (Ablation A/B/C/D progression, final 72.40%).
- `AUDIT-FINDINGS-2026-04-25.md` — pre-archive sensitive-data audit.
- This wind-down document.

### Diagnostic methodology
- **33-question regression corpus** as a permanent diagnostic artifact for fast iteration.
- **Invariants 1–6** (documented in `AGENTS.md`):
  1. Artifact/log hash equivalence (catches stale-checkpoint corruption)
  2. Explicit checkpoint naming (no default-filename collisions)
  3. Canonical baseline pinned by commit-hash artifact
  4. (… see `AGENTS.md`)
  5. GPU placement discipline
  6. Production service isolation (port 7777 protected from bench/sprint work)
- **Ablation discipline:** one bit at a time, full per-category breakdown, regression corpus before full LoCoMo.

### Infrastructure
- Docker Compose stack (Qdrant + FalkorDB).
- vLLM topology deployment scripts (`/tmp/bench_runs/baseline_env.sh` not shipped, but referenced — recreate from `config/rasputin.toml` defaults).
- Embedding service contracts for Qwen3-Embedding-8B (4096d native).
- Reranker service contracts for Qwen3-Reranker-0.6B.

---

## 3. Honest framing

> **RASPUTIN is not state of the art on LoCoMo.**

RASPUTIN scores 72.40% non-adv on LoCoMo full 10-conversation (1540 questions). Hindsight scores 89.61% on the same benchmark and is the current SOTA.

The codebase is published for the **architectural insights**, not as a competitive memory system. Anyone evaluating memory systems for production use should evaluate Hindsight first.

Specific findings worth documenting:

- **Lane budget vs. RRF tradeoff** — they solve the same problem (per-lane top-k truncation) and are alternatives, not additive. Ablation C (wider lane budgets, no RRF) and an RRF-only configuration converge to similar scores from opposite directions.
- **Per-lane truncation as a hidden cost** of fact-type partitioning. Splitting retrieval into N lanes means each lane only contributes top-k/N of its native ranking; the global top-k is no longer the global best top-k. This breaks multi-hop questions unless lane budgets are explicitly calibrated.
- **RRF is budget-dependent** — it helps at narrow budgets (where per-lane truncation is acute) and goes neutral or slightly negative at wider budgets (where the truncation problem is already solved by sheer width).
- **Ablation discipline as methodology.** The Sprint 1 ROLLBACK + Sprint 2 recovery happened because we could isolate single-bit changes against the 33-Q corpus and the full 10-conversation LoCoMo run. Without that discipline, the regression that appeared in Sprint 1 would have been attributed to the wrong cause.

---

## 4. Key architectural insights for future readers

### Insight 1 — Fact-type partitioning works for single-hop, breaks multi-hop without lane calibration

Splitting retrieval into separate lanes (windows, facts, entities, events) improves precision for single-hop questions where the answer lives cleanly in one lane. But each lane independently truncates to top-k_lane, and the union of those top-k_lane sets is not the global top-k. Multi-hop questions, which need evidence from multiple lanes, are degraded unless lane budgets are wide enough to keep the cross-lane evidence in the candidate pool.

**Implication for forks:** if you partition retrieval, calibrate per-lane budgets against multi-hop performance, not just single-hop. The default budgets baked into Sprint 2 (113/20/10/8) were chosen by ablation, not theory.

### Insight 2 — RRF and wider lane budgets are alternatives, not additive

Reciprocal Rank Fusion (RRF) addresses per-lane truncation by re-ranking the union by reciprocal rank, which softens the effect of any single lane's hard cutoff. Wider lane budgets address it by simply not cutting off as aggressively. Both achieve the same goal. Combining them does not stack.

**Implication for forks:** pick one. RRF is cheap to add but its benefit shrinks as lane budgets grow. If you have headroom on lane budgets, use it; if not, RRF is a reasonable cheaper substitute.

### Insight 3 — Ablation discipline (one bit at a time) is essential

The Sprint 1 regression was traceable because each phase changed exactly one variable (Phase A: dense/sparse rebalance; Phase B: four-lane partition; Phase C: RRF) and was scored against the same 33-Q corpus and the same canonical baseline artifact. The Sprint 2 recovery happened because we could narrow the regression to specific changes and reverse them.

This is a methodology lesson, not a code one. The 33-Q corpus is in the repo; the ablation discipline is in `AGENTS.md`.

### Insight 4 — Cross-cutting graph traversal is the missing piece

Hindsight's 89.61% likely requires a piece RASPUTIN parked: cross-cutting graph traversal (`memory_links`) over the fact-type partitions, combined with a reflection layer (their CARA — Compose, Analyze, Reflect, Answer) that re-reads retrieved memories before answering. RASPUTIN has the graph infrastructure (FalkorDB, schema, edge population code) but never integrated it into the bench retrieval path.

**Implication for forks:** if you fork RASPUTIN aiming to close the gap, this is where to look first.

---

## 5. What was NOT shipped

These were designed, partially implemented, or planned but did not make it into the final ship:

| Component | Status | Why not shipped |
|---|---|---|
| Phase D temporal lane | Designed + unit-tested + on-disk extractor (`.sisyphus/d-alpha-2-preflight-blocked-config.md`) | Sprint 3 cancelled before integration |
| Phase E prompt routing improvements | Not started | Sprint 3 cancelled |
| Phase G reranker A/B (claude-haiku, gemini-flash, voyage-rerank-2.5) | Not started | Sprint 4 cancelled |
| Phase I-early answer-model swap (claude-opus-4-7 / gemini-3-pro) | Planned, not run | Sprint 4 cancelled |
| `memory_links` graph cross-lane traversal | Parked from v0.8 | Never unparked for bench eval |
| Hindsight CARA-style reflection layer | Not designed | Out of scope for RASPUTIN |
| Sprint 3 work (commits `cc30e28`, `f046dcf`, `687e27c`, `f8b8935`) | Code exists on `sprint-3-integration` branch | Will be deleted before archive — no shipped Sprint 3 work |

---

## 6. Future work for anyone forking

If you fork this codebase to push the score higher, here is the prioritized list based on the Sprint 1/2 evidence:

1. **Integrate the temporal lane.** The structured fields are already populated (`occurred_start`, `occurred_end` in fact payloads). Phase D Phase D-α-2 spec and extractor are in `.sisyphus/`. Hook the lane into `search.hybrid_search()` and pick a lane budget by ablation against the 33-Q corpus.

2. **Unpark `memory_links` graph traversal.** The graph layer exists, the schema exists, the writer exists. The bench retrieval path doesn't use it. Adding a graph-traversal step that follows `memory_links` edges from each retrieved memory to expand the candidate pool is plausibly the largest single retrieval improvement available.

3. **Test answer model swap.** The current answer model is local (`qwen3.5:35b`). Swapping to `claude-opus-4-7` or `gemini-3-pro` for the answer step (not the retrieval step) is straightforward and has been the largest score-mover in similar systems. Sprint 4 projected +5 to +10 points from this alone.

4. **Compare against Hindsight's CARA reflection layer.** Hindsight publishes the reflection prompt and architecture. Adding a Compose-Analyze-Reflect-Answer step over RASPUTIN's retrieved memories is a direct experiment.

---

## 7. Final benchmark numbers

LoCoMo non-adv, full 10-conversation, 1540 questions:

| Run | Score | Artifact |
|---|---|---|
| RASPUTIN v0.9.1 baseline | 72.53% | `benchmarks/results/59c0a369...-locomo-production.json` |
| RASPUTIN Sprint 1 mean | 70.26% | three artifacts in `benchmarks/results/` (see `sprint-1-verdict.md`) |
| **RASPUTIN Sprint 2 final (Ablation C)** | **72.40%** | `benchmarks/results/c623ae4f...-locomo-production.s2-full.json` |
| Hindsight reference (Gemini-3) | 89.61% | https://github.com/zaplabs/hindsight |

The canonical baseline (72.53%) and Sprint 2 final (72.40%) are pinned by commit-hash artifacts per Invariant 3. Do not replace, rename, or quarantine these files.

---

## 8. Acknowledgments

- **Hindsight team** ([zaplabs/hindsight](https://github.com/zaplabs/hindsight)) for the SOTA system, methodology, and open-source reference implementation that defined the bar.
- **LoCoMo benchmark authors** ([snap-research/locomo](https://github.com/snap-research/locomo), ACL 2024) for the evaluation framework.
- **Qwen team** for `Qwen3-Embedding-8B` (retrieval embedding) and `Qwen3-Reranker-0.6B` (reranking).
- **vLLM project** for the inference serving layer.

---

## 9. Project status

**Archived.** No further development. No issue triage. No PR review. Forks welcome under MIT.
