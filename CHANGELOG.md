# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Ongoing hardening and operational improvements.

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

[unreleased]: https://github.com/jcartu/rasputin-memory/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jcartu/rasputin-memory/releases/tag/v0.3.0
