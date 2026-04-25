# Rasputin Memory — Backlog

> **Project archived 2026-04-25.** This file is preserved as historical context, not an active roadmap. None of the items below will be picked up. They remain documented for anyone forking the codebase who wants a starting list of known follow-ups.

Things noticed during sessions but deliberately deferred. Originally maintained as a queue for the next session; preserved here unchanged.

## Bench harness hardening

- **`--reset` silently deletes checkpoints** — no confirmation, no backup. Not a
  bug today but a footgun of the same class as the ghost-checkpoint bug. Add a
  confirmation flag like `--i-know-what-im-doing` OR back up the deleted
  checkpoint to `benchmarks/results/trash/<timestamp>/`. Filed 2026-04-20.

## Bench payload

- **`benchmarks/locomo_leaderboard_bench.py:662-674` strips structured fact
  fields** (`fact_type`, `occurred_start`, `occurred_end`, `confidence`) from
  Qdrant payload before upsert. Blocks Phase B four-lane retrieval. Fix on
  branch `bench-payload-structured-fields`. Filed 2026-04-17, still open.

- **`entities` field serialization in locomo_leaderboard_bench.py fact upsert.**
  Code does `json.dumps(fact.get("entities", []))`. Works currently because
  extract_facts() returns dicts via model_dump(), which converts
  ExtractedEntity objects to plain dicts. Still, this is implicit — if someone
  later changes extract_facts() to return Pydantic objects, `json.dumps` will
  fail silently or raise. Consider explicit normalization
  (`[e if isinstance(e, dict) else e.model_dump() for e in fact.get("entities", [])]`)
  during a future harness cleanup. Filed 2026-04-20.

## Housekeeping

- **19 untracked `benchmarks/results/*.json` files** from prior experiments
  (entity-resolver, fix1-wider-facts, fix2-inference-penalty, knn-full-conv0..8,
  knn-links-v2, structured-facts-*). Decide whether to commit as historical
  artifacts, delete, or leave untracked. Filed 2026-04-20.

- **Move `RASPUTIN_SESSION_STATE.md` into the repo** at `docs/SESSION_STATE.md`.
  Currently lives at `/tmp/files_extract/` — ephemeral. Every future session
  depends on its content so it should be durable + git-tracked. Filed 2026-04-19.

- **Decide fate of untracked `tools/qwen3_reranker_server.py`,
  `tools/web_server.py`, `tools/demo_cli.py`, `tests/test_cross_encoder.py`,
  `web/`.** These are legitimate work that hasn't been absorbed into the repo.
  Review individually, commit the keepers. Filed 2026-04-20.

## Phase A

- **Revisit Phase A multiplicative reranker** after Phase E ships. Phase E
  produces real observation-level proof_count as a stored integer on consolidated
  memories; Phase A's token-overlap proxy is disqualified because it saturates
  at pool size 60. Code preserved in stash
  `phase-a-multiplicative-reranker-PARKED-2026-04-19-ghost-checkpoint-regression`.
  Filed 2026-04-19.

## Known-working infrastructure

- Local Qwen3-Reranker-0.6B server at `tools/qwen3_reranker_server.py` → port
  9091. Running as a user process; convert to systemd user service so it
  survives reboots. Filed 2026-04-20.
  - Observed live launch command on 2026-04-20: `python3 <repo-root>/tools/qwen3_reranker_server.py 9091`
  - Observed working directory: `<repo-root>` (the rasputin-memory checkout)
  - Observed non-secret launch env vars: none beyond the default user shell environment.


## Deferred

- **Standalone BM25 keyword lane (5th retrieval partition).** Sprint 2 Phase B
  initially speced 5 lanes (window + 3 fact_types + BM25). The BM25 lane was
  removed from the four-partition path on 2026-04-23 because
  `tools/bm25_search.py` only exposes rerank-stage fusion helpers
  (`BM25Scorer`, `hybrid_rerank`, `reciprocal_rank_fusion`) — not a standalone
  `search()` function. BM25 keeps its existing rerank-stage role, unchanged
  from Sprint 1. A standalone BM25 retrieval lane would require: SQLite FTS5
  table over memory passage text (the existing `entities_fts` table only
  indexes entities), ingest-time indexing hook in `commit.py`, query-time
  MATCH, plus tests. Estimated ~200-400 LoC new code + tests. Aspirational
  in README. Re-evaluate after Phase B/C+D exit gates to see if retrieval
  recall is a remaining gap. Filed 2026-04-23.