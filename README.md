# RASPUTIN Memory — Experimental Conversational Memory System

> A research codebase exploring fact-type partitioned retrieval for LLM agent memory. **Not state of the art.** Published for the architectural insights.

---

## Status: Archived

**This project is archived and no longer maintained.**

It is published as a public research artifact at the final state of `sprint-2-integration` (commit `29bbcb7`, 2026-04-25). No issues will be triaged, no pull requests reviewed, no further development undertaken. Forks are welcome under the MIT license.

The closure rationale, final benchmark, key insights, and pointers for future work are documented in [`.sisyphus/PROJECT-WINDDOWN.md`](.sisyphus/PROJECT-WINDDOWN.md).

---

## What this is

RASPUTIN is a long-term conversational memory system for LLM agents. It maintains a persistent store of memories extracted from conversations and exposes them via a search API and an MCP server. Internally, retrieval is partitioned across four fact-type lanes (windows, facts, entities, events) and reranked with a cross-encoder.

The architectural exploration the project is published for:

- **Fact-type partitioned retrieval** — splitting the candidate pool into typed lanes before global reranking.
- **Per-lane budget calibration** as the dominant tuning surface (vs. the global top-k).
- **Reciprocal Rank Fusion (RRF) vs. wider lane budgets** as alternative solutions to per-lane truncation.
- **Ablation discipline** as a methodology for distinguishing real improvements from noise on a 1540-question benchmark.

Full architectural details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/HYBRID-BRAIN.md`](docs/HYBRID-BRAIN.md).

---

## Final benchmark

LoCoMo non-adv, full 10-conversation, 1540 questions:

| System | Score |
|---|---|
| RASPUTIN v0.9.1 baseline | 72.53% |
| **RASPUTIN Sprint 2 final (this archive)** | **72.40%** |
| Hindsight (Gemini-3) — current SOTA | **89.61%** |

There is a 17-percentage-point gap between RASPUTIN and the current state of the art on the benchmark RASPUTIN was designed to evaluate against.

**For production memory needs, evaluate [Hindsight](https://github.com/zaplabs/hindsight) first.** It is open source, MIT licensed, independently validated, and significantly stronger on LoCoMo.

LoCoMo benchmark: [snap-research/locomo](https://github.com/snap-research/locomo) (ACL 2024).

---

## Key architectural insights

These are the findings worth reading the code for:

### 1. Fact-type partitioning works for single-hop, breaks multi-hop without lane calibration

Splitting retrieval into typed lanes improves precision when an answer lives cleanly in one lane. But each lane independently truncates to top-k_lane, and the union is no longer the global top-k. Multi-hop questions, which need evidence from multiple lanes, are degraded unless lane budgets are explicitly tuned. The defaults baked into this archive (113 windows / 20 facts / 10 entities / 8 events) were chosen by ablation.

### 2. RRF and wider lane budgets are alternatives, not additive

Reciprocal Rank Fusion and wider per-lane budgets address the same problem (per-lane truncation) from opposite directions. Combining them does not stack. RRF helps at narrow budgets and goes neutral or slightly negative at wider budgets.

### 3. Ablation discipline catches noise that benchmark deltas don't

Per-category breakdowns over a 33-question regression corpus before every full LoCoMo run was the discipline that surfaced both findings above. Without it, single-bit changes are indistinguishable from regression noise. See [`AGENTS.md`](AGENTS.md) (Invariants 1–3) for the operational rules.

### 4. The likely missing piece: cross-cutting graph traversal

RASPUTIN has graph infrastructure (FalkorDB, schema, edge population code) for cross-memory links but never integrated graph traversal into the bench retrieval path. Hindsight's reported score likely depends on traversal-style retrieval combined with a reflection layer (their CARA architecture). For anyone forking, this is the first thing to try.

---

## What's in the repo

```
tools/brain/         Core memory engine (search, commit, graph, scoring, reranker)
tools/brain/server.py    HTTP API server (port 7777)
tools/brain/search.py    Two-lane hybrid search + Qwen3-Reranker
tools/brain/commit.py    Memory commit with A-MAC quality gate
tools/brain/reflect.py   LLM synthesis over retrieved memories
tools/mcp/           MCP server (thin HTTP proxy, port 8808)
tools/pipeline/      Shared utilities

benchmarks/          LoCoMo evaluation harness + canonical artifacts
benchmarks/results/  Pinned baseline + Sprint 2 final artifacts (per Invariant 3)
experiments/         Documented experiment log with keep/revert verdicts

config/rasputin.toml Runtime configuration (TOML + env overrides)
docs/                Setup, architecture, integration, operations
hooks/               Optional agent-framework integration (OpenClaw)

.sisyphus/           Project research history (Sprint verdicts, wind-down doc)
AGENTS.md            Invariants and conventions for code agents
```

---

## How to run

For a local deployment:

```bash
# 1. Start backing services (Qdrant + FalkorDB)
docker compose up -d

# 2. Configure your LLM/embedding/reranker endpoints
cp .env.example .env
# Edit .env — minimum required: QDRANT_URL, EMBED_PROVIDER + key, LLM_PROVIDER + key

# 3. Run the API server
python3 tools/brain/server.py
# Listens on :7777 by default

# 4. (Optional) Run the MCP server
python3 tools/mcp/server.py
# Listens on :8808 by default
```

Detailed setup: [`docs/SETUP.md`](docs/SETUP.md), [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).
Operations / monitoring: [`docs/OPERATIONS.md`](docs/OPERATIONS.md).
Cloud provider configuration: [`docs/CLOUD_PROVIDERS.md`](docs/CLOUD_PROVIDERS.md).

A `quickstart.sh` script wraps the above for a single-command bring-up.

---

## What ships, what doesn't

### In the repo
- Phase B four-partition retrieval with the Sprint 2 final lane budgets
- A-MAC commit-time quality gate
- Two-lane hybrid search with Qwen3-Reranker-0.6B
- HTTP API + MCP server
- 33-question regression corpus + LoCoMo evaluation harness
- Sprint 1 + Sprint 2 verdict documents (`.sisyphus/`)
- Invariants and ablation discipline (`AGENTS.md`)

### Not in the repo
- Phase D temporal lane (designed and unit-tested, not integrated)
- Phase E prompt routing improvements (not started)
- Phase G reranker A/B (not started)
- Phase I-early answer model swap (not run)
- `memory_links` graph traversal in the retrieval path (parked from v0.8)
- Cross-cutting graph + reflection layer (designed for future work, not started)

The full status list and pointers for forks: [`.sisyphus/PROJECT-WINDDOWN.md`](.sisyphus/PROJECT-WINDDOWN.md).

---

## Acknowledgments

- **[Hindsight](https://github.com/zaplabs/hindsight)** team — for the SOTA system, methodology, and open-source reference implementation that defined the bar this archive is honest about not reaching.
- **[LoCoMo](https://github.com/snap-research/locomo)** benchmark authors (ACL 2024) — for the evaluation framework.
- **Qwen team** — for `Qwen3-Embedding-8B` and `Qwen3-Reranker-0.6B`.
- **vLLM project** — for the inference serving layer.

---

## License

[MIT](LICENSE).
