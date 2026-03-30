# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Ongoing hardening and operational improvements.

## [3.0.0] - 2026-03-31

This release completes the 78-task work order across seven phases.

### Added
- **Phase 4 (Graph Layer Overhaul):** unified graph extraction/query flow and safer graph query behavior.
- **Phase 5 (New Features):** contradiction detection, feedback-assisted retrieval tuning, and expanded scoring signals.
- **Phase 6 (Test Suite):** comprehensive unit/integration test coverage plus benchmark dataset support.
- **Phase 7 (CI/CD & Documentation):** modern CI workflow with lint, typing, coverage, and integration-test path.

### Changed
- **Phase 1 (Critical Bug Fixes):** fixed high-impact correctness issues in decay, BM25 flow, extraction, and concurrency handling.
- **Phase 2 (Pipeline Unification):** consolidated retrieval path to reduce drift between components and improve consistency.
- **Phase 3 (Architecture Cleanup):** config normalization, dead-code removal, and type-hint improvements for maintainability.
- **Documentation:** README refreshed for v3.0 architecture, setup, config, API usage, and developer workflow.

### Fixed
- Race-prone and edge-case behavior in maintenance paths and ingestion filters identified during the audit-driven work order.

### CI / DevEx
- Python matrix CI standardized around 3.11/3.12.
- Coverage and type-checking gates added to pull request validation.

---

## Work-order phase summary

- **Phase 1:** Tasks 1-25 — Critical bug fixes
- **Phase 2:** Tasks 26-29 — Pipeline unification
- **Phase 3:** Tasks 30-45 — Architecture cleanup
- **Phase 4:** Tasks 46-53 — Graph layer overhaul
- **Phase 5:** Tasks 54-59 — New features
- **Phase 6:** Tasks 60-68 — Test suite
- **Phase 7:** Tasks 69-78 — CI/CD and documentation

[unreleased]: https://github.com/jcartu/rasputin-memory/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/jcartu/rasputin-memory/releases/tag/v3.0.0
