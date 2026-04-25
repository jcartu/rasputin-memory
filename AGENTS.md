# AGENTS.md

Guide for AI coding agents working on this repository.

## Project Structure

```
tools/brain/         Core memory engine (search, commit, graph, scoring, etc.)
tools/brain/server.py  HTTP API server (BaseHTTPRequestHandler)
tools/brain/search.py  Two-lane hybrid search (windows + facts + Qwen3-Reranker)
tools/brain/commit.py  Memory commit with A-MAC gate, dedup, entity extraction
tools/brain/reflect.py LLM synthesis over retrieved memories
tools/mcp/           MCP server (thin HTTP proxy over the API)
tools/pipeline/      Shared utilities (scoring constants, dateparse, etc.)
tools/config.py      TOML config loader with env overrides
benchmarks/          LoCoMo/LongMemEval evaluation harness
experiments/         Documented experiment log with keep/revert verdicts
config/rasputin.toml Runtime configuration
tests/               pytest suite with MockQdrant/MockRedis fixtures
```

## Key Invariants

- **MCP server is a thin proxy** — it calls the HTTP API via `urllib.request`, never imports brain modules directly.
- All search goes through `search.hybrid_search()` in `tools/brain/search.py`.
- All commits go through `commit.commit_memory()` in `tools/brain/commit.py`.
- Qwen3-Reranker-0.6B reranking is essential for two-lane search — do not disable or downgrade to ms-marco-MiniLM.
- A-MAC quality gate must stay enabled on the commit path.
- The `_state.py` module holds all global config — other brain modules import from it.

## Configuration

Runtime config lives in `config/rasputin.toml` with env var overrides.  See `tools/config.py` for the override mapping.  Key sections: `[server]`, `[qdrant]`, `[graph]`, `[embeddings]`, `[reranker]`, `[amac]`, `[reflect]`.

Retrieval pool size is tunable via `BENCH_LANE_WINDOWS` (default 113), `BENCH_LANE_FACT_W` (default 20), `BENCH_LANE_FACT_E` (default 10), and `BENCH_LANE_FACT_I` (default 8). Total per-query budget: 151. Four-partition routing is on by default (Sprint 2 Ablation C, 2026-04-25, +2.14pp vs S1 Mean); set `FOUR_LANE=0` to disable. RRF fusion is off by default (Ablation D 2026-04-25 verified RRF doesn't help at lane budget 151); set `RRF_FUSION=1` to enable for narrow-budget operating points.

## Running

```bash
docker compose up -d              # Qdrant + FalkorDB
python3 tools/hybrid_brain.py     # API server on :7777
python3 tools/mcp/server.py       # MCP server on :8808
```

## Testing

```bash
pytest tests/ -v -k "not integration"   # Unit tests
pytest tests/test_integration.py -v     # Integration (needs Qdrant)
ruff check .                            # Lint
cd tools && mypy brain/ pipeline/ --ignore-missing-imports  # Type check
```

## Benchmarks

```bash
python3 benchmarks/run_benchmark.py --check-thresholds      # Synthetic
python3 benchmarks/bench_runner.py locomo --mode production  # Full LoCoMo
```

Do not modify files in `benchmarks/` without explicit instruction.

## Benchmark discipline

Added 2026-04-19 after the ghost-checkpoint regression where Phase A appeared
to regress -14.1pp. Root cause was a stale checkpoint file being resumed from
under a default filename that Phase A did not override. See
`benchmarks/results/quarantine_2026-04-19/README.md` for the full forensic.

**Invariant 1 — Artifact/log hash equivalence.**
Before interpreting any leaderboard score, hash-check the artifact's per-conv
predictions against the originating log's per-conv printouts. If they disagree
for any conversation, the artifact is corrupt — do not trust the score.

A helper script `scripts/verify_bench_artifact.py` should do this check
automatically (not yet written — file as the first task after Phase B's
payload fix lands). Until then, verify manually by comparing per-conv
accuracy lines in the run log against per-conv accuracy computed from the
artifact JSON.

**Invariant 2 — Explicit checkpoint naming.**
Every `locomo_leaderboard_bench.py` invocation must explicitly set
`BENCH_CHECKPOINT=<unique-experiment-id>.json` (e.g.
`BENCH_CHECKPOINT=phase-b-four-lane-checkpoint.json`). Never rely on the
default `locomo-leaderboard-checkpoint.json` — a stale file by that name may
exist from a previous experiment and the raw harness will silently resume
from it. Prefer `bench_runner.py locomo --mode production` over direct
`locomo_leaderboard_bench.py` invocation: the former writes commit-prefixed
artifacts that cannot collide with stale lineage.

**Invariant 3 — Canonical baseline is the commit-hash-prefixed artifact.**
The canonical v0.9.1-honest baseline is
`benchmarks/results/59c0a369...-locomo-production.json` at 72.53% non-adv.
Recorded in `history.csv` row 59c0a369. Do not replace, quarantine, or rename
this file. Any future baseline comparisons must cite it by filename + row.

**Invariant 5 — GPU allocation.**
Added 2026-04-22 after the Sprint 1 Exit Gate throughput failure where Qwen3-32B-AWQ
was deployed to GPU1 (RTX 5090, 32GB) and ran at 4.6 tok/s under memory pressure,
causing the bench to time out on conv 1/10 after 60 minutes.

- Models with VRAM footprint > 25GB deploy to the **RTX PRO 6000 Blackwells (nvidia-smi index 0 or index 2)** only.
- The **RTX 5090 (nvidia-smi index 1)** is reserved for small models (<25GB footprint) and low-duty services (e.g. reranker-0.6B, embed-4B). Never deploy large (>25GB) models there even if VRAM technically fits — the 32GB budget leaves no headroom for chat-template + prefix cache + KV growth, and throughput collapses.
- Always use `CUDA_DEVICE_ORDER=PCI_BUS_ID` + explicit `CUDA_VISIBLE_DEVICES=<0|1|2>` when launching vLLM. Never rely on default ordering.
- 122B tensor-parallel deployments on GPU0+GPU2 are deprecated for Sprint 1+; local 32B-AWQ on a single Pro 6000 is sufficient.

**Invariant 6 — Production service isolation.**
Added 2026-04-22 after an overnight Sprint 1 Exit Gate halt where restarting the bench
API server on `:7777` would have disrupted production services holding 134,919 Qdrant
points and serving `cartu-proxy`, `casino-dashboard`, `honcho-api`, `enrich-daemon`.

- Port `:7777` (pm2 `hybrid-brain`, `second_brain_*` / `memories_archive` / `episodes` collections) and any Rasputin MCP server running from the main `rasputin-memory` worktree are **PRODUCTION**. Sprint/bench work never binds to these ports, never restarts these services, and never writes to their Qdrant collections.
- Production Qdrant collections (forbidden write targets from bench code): `second_brain_v3`, `second_brain_v2`, `memories_archive`, `episodes`, and all `lme_*` collections.
- Bench API services bind to `:7779` or higher. Bench Qdrant collections use the `locomo_lb_conv_*` or `memory_units_*` prefixes exclusively.
- If any bench code path is found writing to a forbidden collection, treat as a stop-the-world bug. Fix before the next bench run.

## Code Style

- `from __future__ import annotations` on every file
- Line length: 120 (ruff)
- Type annotations on all function signatures
- Logging via `_state.logger` (JSON-formatted)
- HTTP calls use `urllib.request` (not `requests`) for LLM API calls
- Python 3.10+ required, mypy checks against 3.11
