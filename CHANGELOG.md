# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/jcartu/rasputin-memory/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/jcartu/rasputin-memory/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jcartu/rasputin-memory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jcartu/rasputin-memory/releases/tag/v0.3.0
