# Ablation C — Lane Budget Increase: Prep Analysis

**Status**: PREP ONLY — do NOT execute. Awaiting Ablation A verdict + user approval.
**Author**: Sisyphus, 2026-04-24 during Ablation A run
**Context**: Ablation A (FOUR_LANE=1 RRF_FUSION=0) running. Early read on 7/10 convs
shows multi-hop unchanged from S2 (22/66 in both configs), single-hop still recovered
(+7.45pp). Leaning A2 (four-partition structural break) — but user hypothesis is that
the issue is parametric, not conceptual: per-lane top-k too aggressive, truncating
cross-lane multi-hop evidence before fusion/dedup. Ablation C tests that.

---

## 1. User's Proposed Ablation C Config

```
FOUR_LANE=1           # keep four-lane
RRF_FUSION=0          # Ablation A already showed RRF doesn't fix multi-hop, keep off
LANE_BUDGET_PER_TYPE=150   # currently "60 per lane" per user
# All other config identical to Ablation A
```

**Goal**: Preserve Phase B single-hop recovery (+7.45pp) while fixing multi-hop
collapse by relaxing per-lane truncation pressure.

---

## 2. Budget Constants — Exact Location

### Primary read point (retrieval code)

**File**: `tools/brain/search.py`
**Function**: `_four_lane_search()` (line 306)
**Constants** (lines 321-324):

```python
320#ZK|    # Read budgets at call time so env overrides take effect without reload.
321#XB|    lane_windows = int(_os.environ.get("BENCH_LANE_WINDOWS", "45"))
322#QQ|    lane_fact_w  = int(_os.environ.get("BENCH_LANE_FACT_W", "8"))
323#TH|    lane_fact_e  = int(_os.environ.get("BENCH_LANE_FACT_E", "4"))
324#XV|    lane_fact_i  = int(_os.environ.get("BENCH_LANE_FACT_I", "3"))
```

**Default per-call budget** = 45 + 8 + 4 + 3 = **60 total per expanded query**.
Up to 5 expanded queries × 60 = 360 raw candidates max pre-dedupe.

**Read at call time** (line 320 comment confirms): env var overrides take effect
without a reload. Verified matching this pattern in test
`test_four_lane_respects_budget_env_vars` (tests/test_four_lane.py:82-104) which
sets the 4 env vars to 99/11/7/5 and asserts each `qdrant_search` receives those
limits.

### Shape of budget: 4 separate env vars, not one

**Critical clarification**: The user's `LANE_BUDGET_PER_TYPE=150` does not map to a
single constant. There is no unified "per-type" knob. The budget is split across
4 env vars because the four lanes have different semantic expected densities:

| Lane | Env var | Default |
|---|---|---|
| window | `BENCH_LANE_WINDOWS` | 45 |
| fact_world | `BENCH_LANE_FACT_W` | 8 |
| fact_experience | `BENCH_LANE_FACT_E` | 4 |
| fact_inference | `BENCH_LANE_FACT_I` | 3 |

The user's intent ("raise budget to 150 per type") has two reasonable
interpretations:

**Interpretation A — Uniform 150 per lane** (literal reading):
```
BENCH_LANE_WINDOWS=150
BENCH_LANE_FACT_W=150
BENCH_LANE_FACT_E=150
BENCH_LANE_FACT_I=150
```
Total = 600 per expanded query × 5 expansions = 3000 raw candidates.

**Interpretation B — Scale existing proportions ~2.5x** (spirit reading):
```
BENCH_LANE_WINDOWS=113    # 45 × 2.5
BENCH_LANE_FACT_W=20      # 8 × 2.5
BENCH_LANE_FACT_E=10      # 4 × 2.5
BENCH_LANE_FACT_I=8       # 3 × 2.5
```
Total = 151 per expanded query × 5 expansions = 755 raw candidates.

**Interpretation C — Total bucket budget of 150** (matches prior tuning guide):
README Retrieval Pool Tuning table:
- Default: 45w + 15f = 60 per expanded query
- "Wide" (historical): 75w + 25f = 100 per expanded query
- Ablation C proposal: ~113w + ~37f = 150 per expanded query

Both could map to:
```
BENCH_LANE_WINDOWS=113
BENCH_LANE_FACT_W=20
BENCH_LANE_FACT_E=10
BENCH_LANE_FACT_I=7
```

**RECOMMENDATION**: Surface this ambiguity to user in the verdict doc. Default
to Interpretation B (proportional scale) unless user corrects — it preserves
the world/experience/inference priority weighting that was tuned into the
current defaults, and respects fact-type sparsity (inference facts are rarer
than world facts in the LoCoMo corpus).

### Secondary read point (benchmark harness — vestigial)

**File**: `benchmarks/locomo_leaderboard_bench.py`
**Lines 75-76**:
```python
75#ZY|LANE_WINDOWS = int(os.environ.get("BENCH_LANE_WINDOWS", "45"))
76#QM|LANE_FACTS = int(os.environ.get("BENCH_LANE_FACTS", "15"))
```

**Analysis**: These are loaded in the bench runner but **the 4-lane code path
never reads them** — `hybrid_search()` calls `_four_lane_search()` which reads
the 4 env vars directly from the API server's process env. These bench-side
constants control the legacy 2-lane bench path that's bypassed when FOUR_LANE=1.

**Note**: `BENCH_LANE_FACTS=15` is the AGENTS.md-documented env var (and was the
default in v0.9 before Phase B split facts into W/E/I). Phase B replaced this
single knob with three — but the old env var name **still lives** in the bench
runner as vestigial config. For Ablation C, setting `BENCH_LANE_FACTS` does
**not** affect the 4-lane code path; must set the 3 subtypes individually.

---

## 3. Cap Points Downstream of the Budget

There are **three** potential truncation points after `_four_lane_search()` returns
the raw lane hits. Each is a candidate "top-k truncation" that could be killing
multi-hop evidence. The user's hypothesis is that raising per-lane budget relaxes
pressure on #1; but #2 and #3 are also gating the results before reranker sees them.

### Cap #1 — Per-lane qdrant `limit=` at retrieval time (tools/brain/search.py:327-363)

Each lane's `qdrant_search()` call uses `limit=lane_windows` (or fact_w/e/i). This
is the truncation the user's LANE_BUDGET_PER_TYPE=150 directly addresses.

**Effect of raising**: More candidates per lane survive into the cross-lane merge.
Multi-hop questions often need evidence split across lanes (e.g., one hop in a
window chunk, the second hop in a fact chunk). If cap #1 is pinching, raising it
lets more cross-lane candidates reach the merge step.

### Cap #2 — Post-merge flat concat (tools/brain/search.py:444-446, non-RRF path)

```python
444#YY|            else:
445#TN|                for lane_name, lane_hits in lane_results.items():
446#HT|                    all_qdrant_results.extend(lane_hits)
```

**No cap here.** All 4 lanes' hits extend into `all_qdrant_results`. Pre-dedupe.
This path is what Ablation A is running. Ablation C also uses this path (RRF_FUSION=0).

### Cap #3 — RRF-path secondary cap (tools/brain/search.py:441-443, RRF path only)

```python
441#ZR|                fused = reciprocal_rank_fusion(lane_results, k=60)
442#XZ|                # Cap at limit*2+60, matching parity plan §4.3 pre-CE budget
443#BR|                all_qdrant_results.extend(fused[: min(limit * 2 + 60, 120)])
```

**Hard ceiling = 120 items** (with bench `limit=60`, calc = `min(180, 120)=120`).
**Only triggered when RRF_FUSION=1.** Ablation C has RRF off, so this cap does
not apply. BUT: this cap is worth noting because **S2 full ran with RRF on** and
this 120 ceiling may have been a contributing factor — if multi-hop evidence
ranked #121+ in RRF order, it was thrown away before the reranker saw it.

Note: `k=60` in `reciprocal_rank_fusion()` is the TREC smoothing constant, **not**
a top-k truncation. fusion.py:3 explicitly notes "k=60 is TREC-standard; do not
tune as part of this phase".

### Cap #4 — Reranker top-k (tools/brain/search.py:567)

```python
567#XK|    all_candidates = _ce.rerank_with_recency(query, all_candidates, top_k=limit)
```

**Effect**: Qwen3-Reranker selects top-60 from whatever merged pool reaches it.
Raising lane budgets grows the pool the reranker chooses from — this is the
mechanism by which Ablation C could recover multi-hop *if* the reranker is
actually good at finding the right candidates in a larger pool.

### Summary of cap geometry

With Ablation A (FOUR_LANE=1 RRF_FUSION=0, current defaults):
```
qdrant per lane → 45w+8W+4E+3I = 60/query   (cap #1)
  × 5 expansions                = 300 raw per query group
post-merge flat concat          = 300 + constraint/graph/BM25 additions, no cap (cap #2)
→ dedupe + score aggregation
→ Qwen3-Reranker top-60         = 60 chunks to Haiku (cap #4)
```

With Ablation C (FOUR_LANE=1 RRF_FUSION=0, Interpretation B proportional scale):
```
qdrant per lane → 113w+20W+10E+8I = 151/query   (cap #1, 2.5x)
  × 5 expansions                  = 755 raw per query group
post-merge flat concat            = 755 + constraint/graph/BM25, no cap (cap #2)
→ dedupe + score aggregation
→ Qwen3-Reranker top-60           = 60 chunks to Haiku (cap #4 unchanged)
```

**Net effect**: reranker gets ~2.5x more candidates to choose from. If multi-hop
evidence was being truncated at cap #1, this recovers it. If it was never in the
vector-neighbor pool at all (semantic miss), this doesn't help.

---

## 4. Code Changes Required

### Interpretation A or B (pure env var change): ZERO lines modified.

The 4 env vars already exist, already read at call-time, already tested.
Ablation C deploys as env overrides on the hybrid_brain process launched by
the bench harness. Same pattern as Ablation A.

**Example launch** (Interpretation B proportional scale):
```bash
BENCH_LANE_WINDOWS=113 \
BENCH_LANE_FACT_W=20 \
BENCH_LANE_FACT_E=10 \
BENCH_LANE_FACT_I=8 \
FOUR_LANE=1 \
RRF_FUSION=0 \
BENCH_API_PORT=7779 \
BENCH_CHECKPOINT=s2-ablation-C-wide-budget.json \
bench_runner.py locomo --mode production --skip-ingest --allow-cross-commit --force
```

### Interpretation C (add unified LANE_BUDGET_PER_TYPE env var): ~8 LOC, optional.

If user wants a single knob as a convenience (maps to all 4 env vars when set):

```python
# tools/brain/search.py, in _four_lane_search() after line 324
_unified = _os.environ.get("LANE_BUDGET_PER_TYPE")
if _unified:
    _u = int(_unified)
    lane_windows = _u
    lane_fact_w = _u
    lane_fact_e = _u
    lane_fact_i = _u
```

**Not recommended** — user's existing 4-knob structure is more expressive, and
adding a unified var introduces ambiguity (which wins if both are set?). Skip
unless user explicitly requests.

### Deploy path

- **Ablation C requires NO commit**. Env var override only.
- Same bench-runner pattern as Ablation A: sprint-2-integration @ `c623ae4`,
  env vars set at API server launch.
- Artifact preservation: same `--force` overwrite risk as Ablation A — must
  pre-backup the Ablation A artifact before launching C (rename to
  `.ablation-A.json` as planned post-completion).
- Invariant 2 compliant checkpoint: `s2-ablation-C-wide-budget.json` (or
  similar, matching pattern set by A).

---

## 5. Test Dependencies

Tests that assert on current budget defaults or exercise the 4-env-var read path:

**`tests/test_four_lane.py`** (sprint-2 worktree):
- `test_four_lane_respects_budget_env_vars` (line 82): sets env to 99/11/7/5,
  asserts each qdrant_search call receives matching limit. **Raising env vars in
  Ablation C does not break this test** — test uses monkeypatch to set its own
  env values, so Ablation C's runtime env has no effect on test.
- `test_four_lane_fires_four_qdrant_lanes` (line ~40): asserts 4 lanes fire with
  correct chunk_type / fact_type filters. Budget-agnostic.
- `test_fact_type_filter_pushed_into_qdrant_filter` (line 107): asserts qdrant
  FieldCondition structure. Budget-agnostic.

**No test asserts on specific default values** (45/8/4/3). Changing defaults in
code (for a real commit, not Ablation C) would not break any test. Ablation C
is env-only so test surface is zero.

### Hidden dependency: expansion-count × budget interaction

Ablation C's effective pool size depends on how many query expansions are
generated per question. `expand_queries(..., max_expansions=5)` in
`tools/brain/search.py:424` — up to 5 additional queries. Single-hop questions
may yield only 1-2 expansions; multi-hop yield more. This means Ablation C's
**multi-hop pool grows more than single-hop pool**, which is actually what we
want — multi-hop is the target category.

---

## 6. Other Code Paths Depending on These Env Vars

Searched across the sprint-2 repo for all reads:

```
tools/brain/search.py:321-324     # primary read (4-lane)
benchmarks/locomo_leaderboard_bench.py:75-76  # vestigial bench-runner load (2-lane path only)
tests/test_four_lane.py:84-87     # test monkeypatch
README.md:314-315, 120            # docs: wide-pool tuning example
CHANGELOG.md:43                   # docs: v0.9 wide-pool note
experiments/2026-04-07_twolane_search.md:40  # history: BENCH_LANE_FACTS= usage
experiments/2026-04-14_wider_retrieval_pool.md:47  # history: 75w+25f tuning
```

**No other code path reads these.** No config/rasputin.toml entry (the env
vars are bench/override-only, not a runtime config). No test that pins
defaults. No production code branch that would regress.

---

## 7. Expected Ablation C Outcomes (Hypothesis Matrix)

If Ablation A verdict is A2 (four-partition structural break) and user's
hypothesis (parametric not conceptual) is correct:

| Outcome | Multi-hop | Single-hop | Interpretation |
|---|---:|---:|---|
| **C1 (hypothesis confirmed)** | recovers to ≥47% | stays ≥52% | Cap #1 was pinching. Ship Phase B + wider pool. |
| **C2 (partial fix)** | 44-46% | stays ≥52% | Pool width helps some. Consider Ablation D (weighted RRF or cap #3 lift). |
| **C3 (no change)** | stays ~43% | stays ≥52% | Not a truncation problem. Evidence isn't in vector-neighbor pool at all. Revert Phase B. |
| **C4 (regression)** | <43% or single-hop drops | — | Larger pool hurts reranker discrimination. Revert Phase B. |

**Cost of Ablation C run**: ~5-6h (same cache state as Ablation A, QA phase
dominant, cache-hit ingest). Spend: ~$2-5 (Haiku answers + gpt-4o-mini judge,
same as A).

**Go-no-go criteria** (recommend in verdict doc):
- Run Ablation C only if Ablation A verdict is **A2 or A3**.
- If Ablation A is **A1** (RRF was culprit), Phase B is fine and user should
  address RRF instead — Ablation C skipped.

---

## 8. Compatibility with Invariants

- **Invariant 1** (artifact/log hash): Ablation C artifact must be hash-verified
  against its log before scoring, same as Ablation A.
- **Invariant 2** (explicit checkpoint): Ablation C must set
  `BENCH_CHECKPOINT=s2-ablation-C-wide-budget.json` (or similar unique name).
- **Invariant 3** (canonical baseline): Does not affect 59c0a369.
- **Invariant 5** (GPU): Same footprint as Ablation A — reranker on GPU1, answer
  model local 32B-AWQ on GPU0 or GPU2.
- **Invariant 6** (service isolation): API server on :7779, collections
  `locomo_lb_conv_*`. No production touch.

---

## 9. Artifact Preservation Plan for Ablation C

Ablation C runs on the **same commit** as S2 full (`c623ae4`) and Ablation A.
Without mitigation, `bench_runner.py --force` overwrites the commit-prefixed
artifact. Same risk identified before Ablation A launch.

**Pre-launch checklist for Ablation C**:
1. Rename Ablation A artifact to `c623ae4f...-locomo-production.ablation-A.json`
   (this is already planned as post-A-completion action).
2. Snapshot history.csv to `history.csv.pre-ablation-C.bak`.
3. Snapshot Qdrant point counts (cache-invalidation detection) — same as A.
4. Record pre-launch env at `/tmp/bench_runs/s2-ablation-C/launch.sh`.
5. Post-completion: rename new artifact to `.ablation-C.json`, verify Invariant 1,
   deduplicate history.csv.

---

## 10. Open Questions for User

Before launching Ablation C, user should confirm:

1. **Budget interpretation**: A (uniform 150), B (proportional 2.5x scale), or
   C (total 150 split proportionally)?
2. **Skip if A1 verdict**: If Ablation A shows RRF was the culprit, do we skip
   Ablation C and address RRF directly?
3. **Expansion cap**: Should Ablation C also cap `max_expansions` lower (e.g. 3)
   to avoid exploding total pool? Currently 5 × 151 = 755 per query group pre-dedupe.

---

## 11. Ready-to-Execute Launch Script (DO NOT EXECUTE)

Template for when user approves:

```bash
#!/bin/bash
set -euo pipefail

# Interpretation B (proportional scale ~2.5x)
export FOUR_LANE=1
export RRF_FUSION=0
export BENCH_LANE_WINDOWS=113
export BENCH_LANE_FACT_W=20
export BENCH_LANE_FACT_E=10
export BENCH_LANE_FACT_I=8
export BENCH_API_PORT=7779
export BENCH_CHECKPOINT=s2-ablation-C-wide-budget.json
export LOCAL_VLLM_URL=http://127.0.0.1:11440/v1/chat/completions
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd /home/josh/.openclaw/workspace/rasputin-memory-sprint2
mkdir -p /tmp/bench_runs/s2-ablation-C
date -Iseconds > /tmp/bench_runs/s2-ablation-C/start_time.txt

# Pre-backup (Ablation A overwrite risk)
cp benchmarks/results/c623ae4f*-locomo-production.json \
   benchmarks/results/c623ae4f658a9100cbee5c7fb7a5b41dc6fccfaa-locomo-production.ablation-A.json
cp benchmarks/results/history.csv benchmarks/results/history.csv.pre-ablation-C.bak

# Snapshot cache state
python3 -c "
import json, urllib.request
snap = {}
for conv in ['26','30','41','42','43','44','47','48','49','50']:
    col = f'locomo_lb_conv_{conv}'
    with urllib.request.urlopen(f'http://127.0.0.1:6333/collections/{col}', timeout=3) as r:
        snap[col] = json.loads(r.read())['result']['points_count']
json.dump(snap, open('/tmp/bench_runs/s2-ablation-C/pre_launch_point_counts.json', 'w'), indent=2)
print(snap)
"

# Launch detached
setsid nohup python3 benchmarks/bench_runner.py locomo \
    --mode production --skip-ingest --allow-cross-commit --force \
    > /tmp/bench_runs/s2-ablation-C/bench.log 2>&1 &
disown
```

---

**End of prep doc. Awaiting Ablation A completion + user approval.**
