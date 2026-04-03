# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-04-03

**LoCoMo: 91.36% — #1 on the leaderboard.** Also: LongMemEval 89.40%, FRAMES 50.4%.

### Benchmarks
- **LoCoMo 91.36%** (1,986 QA) — #1, beating Backboard (90.00%). LLM-judge accuracy, non-adversarial.
- **LongMemEval 89.40%** (500 QA, ICLR 2025) — conversational memory across 6 question types
- **FRAMES 50.4%** (824 QA, Google 2024) — multi-hop factual reasoning over Wikipedia
- **LoCoMo-Plus** (2,387 QA, ARR 2026) — cognitive memory evaluation with implicit constraint recall (harness shipped, baseline running)
- Benchmark harnesses: `locomo_leaderboard_bench.py`, `longmemeval_bench.py`, `frames_bench.py`, `locomo_plus_bench.py`

### Added — Retrieval Pipeline
- Conversation-window chunking: 5-turn overlapping windows (stride 2) alongside individual turns
- Multi-query retrieval: name + topic decomposition with merged deduplication
- Token-overlap deduplication (>75% overlap removed before answer generation)
- Temporal-aware retrieval boost: 1.5× for date-bearing passages on temporal queries
- MMR diversity selection: token-overlap filtering reduces redundant results
- Boost cap at `MAX_TOTAL_BOOST = 3.0` (was unbounded)
- Multi-query name + topic decomposition backported to production `hybrid_search`
- `/commit_conversation` endpoint for windowed multi-turn ingestion
- `speaker`, `mentioned_names`, `has_date`, `constraint_summary` payload fields on commit
- Latin name extraction fallback (no longer requires `known_entities.json`)
- Configurable search rate limit (`RATE_LIMIT_SEARCH` env var)
- `ThreadPoolExecutor` for access tracking (replaces fire-and-forget `Thread()`)

### Added — Constraint Extraction (experimental)
- `tools/brain/constraints.py`: extracts implicit constraints (goals, states, values, causal chains) from text via local LLM
- Query intent decomposition: for non-factual queries, generates intent-based sub-queries
- `[constraints]` config section with `enabled`, `model`, `timeout`
- Disabled by default; designed for cognitive memory evaluation

### Changed
- Benchmark answer prompt rewritten for adversarial resistance (entity-swap tolerant)
- Multi-query search: 5 sub-queries × top-60 per query, merged and deduplicated
- Answer generation context window: 30 → 50 chunks
- Temporal regex: `\d{4}` → `\b(19|20)\d{2}\b` (avoids false positives on apartment numbers, IDs)
- Unified name extraction regex + stopwords in `pipeline/scoring_constants.py`

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

[unreleased]: https://github.com/jcartu/rasputin-memory/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/jcartu/rasputin-memory/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jcartu/rasputin-memory/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jcartu/rasputin-memory/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jcartu/rasputin-memory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jcartu/rasputin-memory/releases/tag/v0.3.0
