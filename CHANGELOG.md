# Changelog

All notable changes to this project are documented in this file.

## [2026-04-08]

### Added
- Markdown architecture diagram upgrade to Mermaid in the main README.
- Competitor positioning section in README (Mem0, Zep, LangMem, MemGPT).
- Contributor pre-commit configuration in `.pre-commit-config.yaml`.
- Demo deployment guide in `docs/DEMO_SETUP.md`.
- Demo compose override in `docker-compose.demo.yml`.

### Changed
- README phase status updated to reflect all 80 shipped cognitive phases.
- README and docs edition tables aligned to explicit SLA tiers (99.5% self-managed, 99.9% managed).
- README opening value proposition and navigation links improved.
- Benchmark table clarified so heuristic-mode LLM rates are interpreted correctly.

### Security
- Removed hardcoded demo passwords from README quickstart section.

## [2026-04-07]

### Added
- Python SDK publishing workflow for PyPI and SDK packaging metadata updates.
- OpenAPI publish workflow scaffolding and developer workspace improvements.

### Fixed
- Backend warning cleanup and deterministic narrative archive behavior.
- Async task wrapper behavior and test warning cleanup.
