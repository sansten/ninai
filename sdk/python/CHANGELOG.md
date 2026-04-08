# Changelog

All notable changes to the Ninai Python SDK are documented in this file.

## 1.1.0a1 (2026-04-08)

### Changed
- Bumped SDK version to `1.1.0a1` for alpha release distribution on PyPI.

### Release
- Alpha release is published via GitHub Actions tag workflow trigger: `sdk-python-v*`.

## 1.0.1 (2026-04-08)

### Changed
- Updated package metadata and project URLs in `pyproject.toml` to align with public docs, repository, issues, and changelog links.
- Added SDK `README.md` with installation, async quick start, resource overview, and support/documentation references.

### Packaging
- Kept PyPI package name as `ninai`.
- Confirmed Python support range: `>=3.10`.
- Confirmed runtime dependencies: `httpx>=0.24.0`, `pydantic>=2.0.0`.

## 1.0.0 (2026-04-07)

### Added
- First stable release of the Ninai Python SDK.
- Core client: `NinaiClient` with API key and JWT authentication support.
- Resource APIs:
	- `MemoriesResource` (CRUD, search, attachments upload/download/delete)
	- `OrganizationsResource`
	- `TeamsResource`
	- `SelfModelResource`
	- `ToolsResource`
	- `LLMResource`
	- `TopicsResource`
	- `CausalResource`
	- `ConsolidationResource`
	- `TemporalResource`
	- `MetaCognitiveResource`
	- `CompositionResource`
	- `EnrichmentResource`
	- `InsightsResource`
	- `DigestResource`
	- `ComplianceResource`
	- `ProofResource`
- Agent helpers:
	- `GoalPlannerAgent`
	- `GoalLinkingAgent`
	- `MetaAgent`
- SDK exports and helpers:
	- `ToolInvoker`
	- `InMemoryEventSink`
	- typed exception hierarchy (`NinaiError`, `AuthenticationError`, `NotFoundError`, `ValidationError`, `RateLimitError`)

### Notes
- Semver policy: no breaking changes within major version.
