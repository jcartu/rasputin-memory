# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [2.0.0] - 2026-03-30

### Added
- Honcho integration for peer-derived conclusions
- Graph Brain knowledge graph with FalkorDB
- Predictive Memory (access tracking, pattern analysis, anticipatory prefetch)
- OpenClaw-Mem hook (auto-recall on every message)
- MCP server for external tool integration
- Memory Consolidator v4 with A-MAC quality gate
- STORM Wiki generator from memory
- BrainBox procedural (Hebbian) memory
- Multi-tenant isolation (per-agent memory partitioning)
- Cloud provider guide (OpenAI, Cohere, Voyage, OpenRouter alternatives)
- Comprehensive cron job documentation
- End-to-end Getting Started guide
- Quickstart bootstrap script

### Changed
- Renamed kebab-case Python files to snake_case
- Updated all tools to use environment variables instead of hardcoded values
- Overhauled README with architecture diagram, comparison tables, budget stacks

### Fixed
- Shell interpolation syntax in Python files (${VAR:-default} → os.environ.get())
- GPU UUID hardcodes removed

## [1.0.0] - 2026-03-28

### Added
- Hybrid Brain server (4-stage retrieval: Vector + BM25 + Graph + Reranker)
- Memory engine (commit/search/recall API)
- BM25 keyword search
- Reranker server (BGE cross-encoder)
- Memory dedup pipeline
- Fact extractor
- Embed server
- Docker Compose infrastructure
- Initial documentation
