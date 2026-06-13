# Changelog

All notable changes to the Ninai Python SDK are documented in this file.

## 0.2.0 (2026-05-21)

### Added — v2 Graph-RAG + DNC Engine support

- `NinaiClient(engine_version="v1"|"v2")` — constructor-level default engine selector.
  `"v1"` (default) keeps all existing behaviour. `"v2"` routes `memories.create()` and
  `memories.search()` to the new Graph-RAG + DNC pipeline.

- `engine_version` per-call override on `memories.create()` and `memories.search()`.
  Takes precedence over the constructor default, enabling mixed usage within one client.

- `session_id` parameter on `memories.create()` and `memories.search()` for v2 turn chaining.
  Auto-generated (UUID4) when omitted.

- `client.v2` — dedicated `V2EngineResource` with three methods:
  - `interact(user_input, session_id, prev_utterance_id)` — full 3-phase cognitive loop
    (dual-path retrieval → Graph-RAG inference → FalkorDB write-back + decay).
  - `graph_inspect(entity_ids, hops, limit)` — inspect the FalkorDB knowledge graph.
  - `health()` — verify FalkorDB + vLLM connectivity.

- `V2InteractResult` model — response, `cited_node_ids`, `extracted_entities`,
  `graph_nodes_retrieved`, `qdrant_chunks_retrieved`, `graph_writes`, `decay_stats`, `latency_ms`.

- `V2GraphInspectResult`, `V2GraphNode`, `V2HealthResult` models.

### Deploy
Tag `sdk-python-v0.2.0` to publish to PyPI via the existing GitHub Actions workflow.

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
