# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-04-13

**Production: 74.2% non-adv** (+6.7pp from baseline). **Compare: 77.7% non-adv** (+10.2pp).
Qwen3-Reranker-0.6B replaces ms-marco-MiniLM-L-6-v2. BM25 keyword search via SQLite FTS5.
30+ documented experiments with scientific methodology.

### Added — Retrieval Quality
- **Qwen3-Reranker-0.6B**: Foundation-model reranker with 0.99/0.0001 score separation (+4.5pp production, +8.6pp compare). Replaces ms-marco-MiniLM-L-6-v2 (0.15/0.15 separation).
- **BM25 keyword search**: SQLite FTS5 in-memory sidecar with Reciprocal Rank Fusion (+0.6pp). First positive BM25 result — enabled by the stronger reranker filtering out keyword-matched but irrelevant facts.
- **Compare mode**: Haiku answers + generous judge methodology for field-comparable numbers.
- **Reranker server**: `tools/brain/cross_encoder_server.py` supports both classic CrossEncoder and Qwen3 chat-template inference (yes/no logit extraction).

### Added — MCP & Synthesis
- **MCP server** (`tools/mcp/server.py`): 6 tools via FastMCP 3.2 streamable-http transport — native support for Claude Code, Cursor, Codex, and any MCP client
- **`/reflect` endpoint**: LLM synthesis over retrieved memories — retrieves, formats context, calls Anthropic or Ollama, returns coherent answer with source citations
- `tools/brain/reflect.py`: reflect module with Anthropic + Ollama providers, automatic fallback
- Docker service for MCP server
- 36 new tests (22 MCP + 14 reflect), total 142 tests

### Tested and Parked (30+ experiments)
- **Consolidation** (6 variants): 636 obs Groq, 33 obs gpt-4o-mini, separate/same collection, gated/additive. Net negative in all configurations.
- **Graph expansion** (kNN links): −4.4pp. Similar-but-irrelevant facts drown real answers.
- **Entity search** (3 variants): −10pp to −14pp. Entity matches flood context.
- **BM25 with L-6 CE** (3 variants): −14pp to −28pp. Weak CE can't filter keyword matches.
- **CE L-12**: +1.3pp overall but −12.6pp single-hop. Reverted.
- **gpt-4o-mini as answer model**: −10.8pp vs Haiku. Reverted.

### Changed
- Default reranker: `Qwen/Qwen3-Reranker-0.6B` (was `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `config/rasputin.toml`: reranker provider updated to `qwen3`
- `fastmcp>=3.2.0` replaces `mcp>=1.0.0` in optional dependencies

## [0.8.0] - 2026-04-10

Full 10-conversation LoCoMo validation: **69.1% non-adv** (1986 questions, production mode).
21 documented experiments with scientific methodology. Major pipeline simplification.

### Benchmarks — Honest Numbers
- **LoCoMo full 10-conv production: 69.1%** non-adv (1540 non-adversarial questions)
  - Open-domain: 81.1% (841 Qs) — rock solid
  - Temporal: 66.4% (321 Qs)
  - Multi-hop: 55.2% (96 Qs) — +16.7pp from prompt routing
  - Single-hop: 41.1% (282 Qs)
  - Adversarial: 11.7% (446 Qs) — not an optimization target
- Previous conv-0-only claim (69.7%) replaced with honest full-dataset number
- 21 experiments in `experiments/` with keep/revert verdicts and full data

### Added — Retrieval Architecture
- **Prompt routing**: per-question classification (inference/factual/temporal) with tailored answer prompts — validated +16.7pp multi-hop, +3.9pp single-hop
- **Cross-encoder GPU server**: `tools/brain/cross_encoder_server.py` for remote GPU inference (33ms/60 pairs on RTX 5090)
- **Structured fact extraction module**: `tools/brain/fact_extractor.py` — 5-dimension decomposition (what/when/where/who/why) with coreference resolution
- **Consolidation engine**: `benchmarks/precompute_consolidation.py` — Hindsight-style observation synthesis (gpt-4o-mini, 30-80 obs/conv)
- **kNN link computation**: `benchmarks/precompute_links.py` — semantic graph (48,200 links across 10 conversations)
- **Graph expansion**: follow kNN links from search seeds to find connected facts
- **CE A/B test infrastructure**: compare cross-encoder models (L-6 vs L-12)

### Changed — Pipeline Simplification
- `search.py` stripped from 700 → 427 lines via ablation-proven dead stage removal
- BM25, keyword boost, entity boost, temporal boost, MMR diversity — all removed (proven 0pp)
- Cross-encoder reranking enabled by default (`CROSS_ENCODER=1`)
- Remote cross-encoder support via `CROSS_ENCODER_URL` env var

### Tested and Parked
- **Consolidation**: 6 variants tested (636 obs Groq, 33 obs gpt-4o-mini, separate collection, same collection, gated, additive). Net negative in all configurations with dense-only retrieval. Parked pending multi-path retrieval infrastructure (graph + entity + temporal search).
- **L-12 cross-encoder**: +1.3pp non-adv but -12.6pp single-hop. L-6 remains default.
- **BM25 third lane**: -3.9pp regression. Qdrant text search doesn't provide real corpus-level IDF.

### Proven Dead Weight (ablation-tested, 0pp contribution)
- BM25 + RRF fusion
- Keyword/entity/temporal additive boosts
- MMR diversity filtering
- Cohere reranker at 60-chunk context
- Cross-encoder at 60-chunk single-lane (essential at two-lane)

### Fixed
- Cloudflare User-Agent blocking on API calls
- Ephemeral port exhaustion from search sleeps
- GPU deadlock with Flask cross-encoder server

## [0.7.0] - 2026-04-03

LoCoMo conv-0: 69.7% production, 72.4% compare (non-adversarial). Scientific ablation program proved which pipeline stages contribute and which don't.

### Benchmarks
- **LoCoMo 69.7%** (conv-0, 199 QA) — production mode (Haiku answers, neutral judge)
- **LoCoMo 72.4%** (conv-0, 199 QA) — compare mode (gpt-4o-mini answers, generous judge)
- Benchmark harnesses: `locomo_leaderboard_bench.py`, `longmemeval_bench.py`, `frames_bench.py`, `locomo_plus_bench.py`
- `bench_runner.py` with production/compare modes, git integrity, batch API (50% savings)
- `analyze_failures.py` — retrieval oracle diagnostic (Gold-in-Top-N, failure taxonomy)
- 12 documented experiments in `experiments/` with keep/revert verdicts

### Added — Retrieval Pipeline
- Two-lane search: windows (45 slots) + facts (15 slots) with guaranteed coverage
- LLM fact extraction at ingest (Claude Haiku): date resolution, coreference, self-contained facts
- Cross-encoder reranker (ms-marco-MiniLM-L-6-v2, 22MB, CPU, 76ms/60 docs)
- Windows-only chunking: 5-turn overlapping windows, stride 2 (individual turns add 0pp — proven)
- Ablation config: env vars to toggle every pipeline stage independently
- Score breakdown instrumentation: `_score_breakdown` dict per result

### Proven Dead Weight (ablation-tested, 0pp contribution)
- BM25 + RRF fusion (IDF computed on retrieved set, not corpus)
- Keyword/entity/temporal additive boosts (too small to change rank order)
- MMR diversity filtering
- Cohere reranker at 60-chunk context
- Cross-encoder at 60-chunk context (helps only at smaller context)

### Added — Constraint Extraction (experimental)
- `tools/brain/constraints.py`: extracts implicit constraints via LLM
- Disabled by default; designed for cognitive memory evaluation

### Changed
- Multi-query search: 5 sub-queries per query, merged and deduplicated
- Answer generation context window: 60 chunks
- Benchmark methodology: production mode (Haiku, neutral) vs compare mode (gpt-4o-mini, generous)

### Fixed — Security & Quality
- `hmac.compare_digest` for timing-safe auth token comparison
- `datetime.now()` → `datetime.now(timezone.utc)` (commit.py, search.py, amac.py)
- `schema_version` 0.3 → 0.7
- `SECURITY.md` updated: v0.7.x/v0.6.x supported (was v0.3.x)
- Protected fields expanded: `speaker`, `mentioned_names`, `has_date`, `source_weight`, `has_contradictions`, `connected_to`, `contradicts`, `supersedes`, `pending_archive`, `soft_deleted`, `pending_delete`, `last_accessed`, `constraints`, `constraint_summary`
- Duplicate `graph_enrichment` key removed from search response
- Health endpoint version: 0.7.0

### Removed
- `requirements.txt` (pinned broken `qdrant-client==1.9.0`)
- `benchmarks/ground_truth.jsonl` (medical PII references)

## [0.6.0] - 2026-04-03

**LoCoMo Benchmark: 89.81% — #2 on the leaderboard.**

### Added
- LLM reranker using Claude Haiku — replaces broken BGE cross-encoder on conversational turns
- Professional LoCoMo benchmark harness (`benchmarks/locomo_leaderboard_bench.py`) with Claude Opus answer generation + GPT-4o-mini judge
- Collection override on `/search` endpoint for benchmark isolation
- LLM-judge scoring alongside token-level F1 in benchmark output

### Changed
- Scoring: removed double decay penalty (recency_bonus eliminated from multifactor scoring — temporal decay already handles age)
- Search pipeline: keyword and entity boosting moved after neural reranking (deterministic adjustments are final, not inputs to reranker)
- Benchmark fixtures: 30 templated decay records replaced with semantically diverse records

### Removed
- 15 dead files: scripts/, memory_engine.py, embed_server_gpu1.py, memory_autogen.py, memory_health_check.py, memory_mcp_server.py, empty experiment dirs
- 7 stale benchmark scripts (consolidated to 2)
- Capitalisation regex fallback from entity extraction (config-only path now)

### Fixed
- F1 computation corrected from set-based to Counter-based (matching LoCoMo paper)
- Extraction prompt tightened: max_tokens 200→50, temperature 0.0, few-shot examples
- Context increased from 10→60 chunks per query
- Embed config now supports env var overrides (EMBED_URL, EMBED_MODEL, EMBED_PREFIX_*)
- vLLM /v1/embeddings response format supported alongside Ollama

## [0.5.0] - 2026-04-02

Search quality breakthrough: keyword overlap boosting and entity-aware scoring push recall well past mem0 benchmarks.

### Added
- Token-level keyword overlap boosting with stopword filtering (up to 5x multiplicative boost)
- Entity focus ratio scoring: primary-entity texts boosted higher than diluted mentions (1.5x–3.0x)
- Entity position weighting: earlier mentions in text = higher relevance signal

### Changed
- Search scoring pipeline (`tools/brain/search.py`) now applies keyword and entity boosts before final ranking
- Integration test assertion updated for flexible error message formats

### Benchmarks
- recall@5: 0.67 → **0.82** (+22%)
- recall@10: 0.745 → **0.885** (+19%)
- MRR@10: 0.56 → **0.68** (+21%)
- Entity recall@5: 0.20 → **0.63** (3x improvement)
- Decay recall@5: 0.23 → **0.40** (74% improvement)
- Contradiction recall@5: 0.48 → **0.96** (2x improvement)

## [0.4.0] - 2026-04-01

Architecture overhaul: modular codebase, unified scoring, language-agnostic retrieval.

### Added
- `brain/` package: 11 focused modules extracted from 1800-line god class (`hybrid_brain.py` is now a 93-line facade)
- Structured JSON logging with per-request IDs
- Shared utilities: `pipeline/locking.py`, `pipeline/qdrant_batch.py`, `pipeline/dateparse.py`
- `pipeline/scoring_constants.py`: single source of truth for source importance weights
- Cyrillic NER pattern for entity extraction
- Cypher injection guard on MCP graph queries
- Auth warning at startup when no API token is configured
- PR template and Dependabot config

### Changed
- **Scoring unified**: 5 competing source-weight systems replaced by one `SOURCE_IMPORTANCE` dict
- **English-only string layer deleted**: keyword routing, stop words, supersedes token check — all removed in favor of the language-agnostic embedding layer
- **Source weight stamped at ingest**: stored on payload once, never re-derived at search time
- **Score ordering fixed**: multifactor scoring now runs before neural reranking
- **Graph failures surfaced**: commit response includes `warnings` list instead of burying `graph.written: false`
- **Async I/O fixed**: embedding and reranker servers no longer block the event loop
- **Batch embedding**: `consolidate_second_brain.py` sends full batch instead of one-at-a-time HTTP calls
- Migration scripts moved to `scripts/` directory
- Redundant Redis container removed from docker-compose (FalkorDB is Redis-compatible)
- Dead code and unused references cleaned up across all modules

### Fixed
- Broken bash-style `${VAR:-default}` URLs in MCP server and health check (Python doesn't expand these)
- `IndexError` crash in consolidator when workers > endpoints
- Hardcoded GPU device string in embed server response
- Redundant `os.environ.get()` nesting in fact extractor
- `import re` inside function body in memory decay (called per-point during full scan)
- Archive recovery now checks for existence before re-upserting (idempotent)
- Fragile double-import pattern replaced with `safe_import()` helper

## [0.3.0] - 2026-03-31

Major release: hybrid retrieval pipeline hardened, knowledge graph overhauled, and full test/CI coverage added.

### Added
- Unified graph extraction and query flow with safer FalkorDB query behavior.
- Contradiction detection for conflicting memories.
- Feedback-assisted retrieval tuning and expanded scoring signals.
- Comprehensive unit and integration test suite with benchmark dataset support.
- Modern CI workflow with lint, typing, coverage, and integration-test gates.

### Changed
- Fixed high-impact correctness issues in decay, BM25 flow, extraction, and concurrency handling.
- Consolidated retrieval path to reduce drift between components and improve consistency.
- Config normalization, dead-code removal, and type-hint improvements for maintainability.
- README refreshed for v0.3 architecture, setup, config, API usage, and developer workflow.

### Fixed
- Race-prone and edge-case behavior in maintenance paths and ingestion filters.

### CI / DevEx
- Python matrix CI standardized around 3.11/3.12.
- Coverage and type-checking gates added to pull request validation.

[0.9.0]: https://github.com/jcartu/rasputin-memory/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/jcartu/rasputin-memory/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/jcartu/rasputin-memory/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jcartu/rasputin-memory/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jcartu/rasputin-memory/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jcartu/rasputin-memory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jcartu/rasputin-memory/releases/tag/v0.3.0
