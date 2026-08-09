from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.services.cognitive_ingestion_service as ingestion_module
from app.core.requester_context import RequesterContext
from app.services.cognitive_evidence_service import CognitiveEvidenceService
from app.services.cognitive_ingestion_service import CognitiveIngestionService


@pytest.mark.asyncio
async def test_cognitive_ingestion_finalize_commits_and_enqueues(monkeypatch):
    session = AsyncMock()
    session.commit = AsyncMock()
    memory = SimpleNamespace(id="m-1")

    increments: list[tuple[str, int]] = []

    class _FakeUsageService:
        def __init__(self, db, org_id):
            self.db = db
            self.org_id = org_id

        async def increment(self, *, metric: str, value: int) -> None:
            increments.append((metric, value))

    embed_calls: list[dict] = []
    memory_calls: list[dict] = []
    episode_calls: list[dict] = []
    fact_calls: list[dict] = []

    monkeypatch.setattr(ingestion_module, "UsageService", _FakeUsageService)
    monkeypatch.setattr(ingestion_module, "enqueue_embed_and_index", lambda **kwargs: embed_calls.append(kwargs))
    monkeypatch.setattr(ingestion_module, "enqueue_memory_pipeline", lambda **kwargs: memory_calls.append(kwargs))
    monkeypatch.setattr(ingestion_module, "enqueue_episode_pipeline", lambda **kwargs: episode_calls.append(kwargs))
    monkeypatch.setattr(ingestion_module, "enqueue_fact_pipeline", lambda **kwargs: fact_calls.append(kwargs))

    svc = CognitiveIngestionService(
        session=session,
        user_id="user-1",
        org_id="org-1",
        clearance_level=2,
        roles_string="org_admin",
    )

    result = await svc.finalize_created_memory(
        memory=memory,
        content="hello world",
        request_id="req-1",
    )

    session.commit.assert_awaited_once()
    assert increments == [("memory_writes", 1)]
    assert result.pipelines_enqueued == ["embed", "memory_pipeline", "episode_pipeline", "fact_pipeline"]
    assert embed_calls[0]["memory_id"] == "m-1"
    assert memory_calls[0]["initiator_roles"] == "org_admin"
    assert episode_calls[0]["storage"] == "long_term"
    assert fact_calls[0]["initiator_clearance_level"] == 2


def test_build_gateway_memory_create_merges_requester_metadata():
    requester = RequesterContext(
        user_id="u1",
        org_id="o1",
        roles=["org_admin"],
        timezone="America/New_York",
        location="NYC",
        job_role="engineer",
        dominant_domains=["platform", "reliability"],
        profile_confidence=0.8,
        urgency_signal="routine",
    )

    body = CognitiveIngestionService.build_gateway_memory_create(
        content="Database latency increased",
        title="Incident note",
        tags=["incident"],
        metadata={"source": "manual"},
        requester=requester,
        context_id="ctx-1",
    )

    assert body.source_type == "cognitive_gateway"
    assert body.extra_metadata["source"] == "manual"
    assert body.extra_metadata["_requester_job_role"] == "engineer"
    assert body.extra_metadata["gateway_context_id"] == "ctx-1"


@pytest.mark.asyncio
async def test_cognitive_evidence_package_aggregates_layers(monkeypatch):
    svc = CognitiveEvidenceService(AsyncMock(), org_id="org-1")

    async def _episodes(memory_ids):
        return {
            "m1": [
                {
                    "episode_id": "ep-1",
                    "source": "memory_episode",
                    "title": "Release train",
                    "summary": "Deployment chain",
                    "status": "closed",
                    "linked_memory_ids": ["m1"],
                    "topic_id": "topic-1",
                }
            ]
        }

    async def _semantic(memory_ids, episode_ids):
        return [
            {
                "semantic_node_id": "sn-1",
                "content": "The release train depends on DB migrations.",
                "composite_quality": 0.9,
                "topic_id": "topic-1",
            }
        ]

    async def _topics(topic_ids):
        return [{"topic_id": "topic-1", "label": "deployments", "keywords": ["release", "migration"]}]

    async def _graph(episodes, semantic_nodes, topics):
        return [{"source_type": "semantic_node", "source_id": "sn-1", "target_type": "topic", "target_id": "topic-1", "similarity": 0.88, "k_rank": 1}]

    async def _facts(memory_ids):
        return {
            "facts": [
                {
                    "fact_id": "fact-1",
                    "subject": "release train",
                    "predicate": "depends_on",
                    "object": "db migrations",
                    "confidence": 0.93,
                    "status": "active",
                }
            ],
            "contradictions": [],
        }

    async def _goal_context():
        return {
            "active_goals": [{"goal_id": "g1", "title": "Ship the release", "status": "active"}],
            "knowledge_gaps": [{"gap_id": "kg1", "description": "Need migration status", "gap_type": "missing_fact"}],
            "suggested_goals": [],
            "world_state": {"recent_changes": [], "highlighted_entities": []},
            "loop_health": {"active_goal_count": 1, "knowledge_gap_count": 1},
        }

    monkeypatch.setattr(svc, "_load_unified_episodes", _episodes)
    monkeypatch.setattr(svc, "_load_semantic_nodes", _semantic)
    monkeypatch.setattr(svc, "_load_topics", _topics)
    monkeypatch.setattr(svc, "_load_graph_neighbors", _graph)
    monkeypatch.setattr(svc, "_load_fact_layers", _facts)
    monkeypatch.setattr(svc.goal_loop_service, "build_context", _goal_context)

    package = await svc.build_package(
        query="What did the release depend on?",
        memories=[
            {
                "id": "m1",
                "title": "Release memo",
                "content_preview": "Migration must complete before rollout.",
                "score": 0.72,
                "occurred_at": datetime(2026, 5, 1, tzinfo=timezone.utc).isoformat(),
                "extra_metadata": {"retrieval_learning": {"relevance_score": 0.5}},
            }
        ],
    )

    assert package["evidence_quality"]["coverage_mode"] == "semantic_fact_hybrid"
    assert package["evidence_quality"]["memory_count"] == 1
    assert package["evidence_quality"]["fact_count"] == 1
    assert package["evidence_quality"]["semantic_node_count"] == 1
    assert package["goal_context"]["active_goals"][0]["title"] == "Ship the release"
    assert package["facts"][0]["object"] == "db migrations"
    assert package["temporal_anchors"][0]["memory_id"] == "m1"
    assert package["topics"][0]["label"] == "deployments"
    assert package["entity_context"]["primary_subject"] is None


@pytest.mark.asyncio
async def test_cognitive_evidence_package_merges_duplicate_episode_sources(monkeypatch):
    svc = CognitiveEvidenceService(AsyncMock(), org_id="org-1")

    async def _episodes(memory_ids):
        return {
            "m1": [
                {
                    "episode_id": "ep-shared",
                    "source": "case_episode",
                    "title": "Release investigation",
                    "summary": "Case-side summary",
                    "linked_memory_ids": ["m1"],
                    "tags": ["case"],
                    "entities": {"owner": "alice"},
                }
            ],
            "m2": [
                {
                    "episode_id": "ep-shared",
                    "source": "memory_episode",
                    "title": "Release investigation",
                    "summary": "Durable summary",
                    "linked_memory_ids": ["m2"],
                    "tags": ["durable"],
                    "entities": {"system": "database"},
                }
            ],
        }

    async def _empty_semantic(memory_ids, episode_ids):
        return []

    async def _empty_topics(topic_ids):
        return []

    async def _empty_graph(episodes, semantic_nodes, topics):
        return []

    async def _empty_facts(memory_ids):
        return {"facts": [], "contradictions": []}

    async def _goal_context():
        return {
            "active_goals": [],
            "knowledge_gaps": [],
            "suggested_goals": [],
            "world_state": {"recent_changes": [], "highlighted_entities": []},
            "loop_health": {},
        }

    monkeypatch.setattr(svc, "_load_unified_episodes", _episodes)
    monkeypatch.setattr(svc, "_load_semantic_nodes", _empty_semantic)
    monkeypatch.setattr(svc, "_load_topics", _empty_topics)
    monkeypatch.setattr(svc, "_load_graph_neighbors", _empty_graph)
    monkeypatch.setattr(svc, "_load_fact_layers", _empty_facts)
    monkeypatch.setattr(svc.goal_loop_service, "build_context", _goal_context)

    package = await svc.build_package(
        query="What happened during the release investigation?",
        memories=[
            {"id": "m1", "title": "Case memory", "content_preview": "Investigating a release regression."},
            {"id": "m2", "title": "Durable memory", "content_preview": "Deployment sequence captured."},
        ],
    )

    assert len(package["episodes"]) == 1
    episode = package["episodes"][0]
    assert episode["source"] == "memory_episode"
    assert set(episode["linked_memory_ids"]) == {"m1", "m2"}
    assert set(episode["tags"]) == {"case", "durable"}
    assert episode["entities"]["owner"] == "alice"
    assert episode["entities"]["system"] == "database"


@pytest.mark.asyncio
async def test_cognitive_evidence_package_merges_inline_state_facts(monkeypatch):
    svc = CognitiveEvidenceService(AsyncMock(), org_id="org-1")

    async def _episodes(memory_ids):
        return {}

    async def _semantic(memory_ids, episode_ids):
        return []

    async def _topics(topic_ids):
        return []

    async def _graph(episodes, semantic_nodes, topics):
        return []

    async def _facts(memory_ids):
        return {"facts": [], "contradictions": []}

    async def _goal_context():
        return {
            "active_goals": [],
            "knowledge_gaps": [],
            "suggested_goals": [],
            "world_state": {"recent_changes": [], "highlighted_entities": []},
            "loop_health": {},
        }

    monkeypatch.setattr(svc, "_load_unified_episodes", _episodes)
    monkeypatch.setattr(svc, "_load_semantic_nodes", _semantic)
    monkeypatch.setattr(svc, "_load_topics", _topics)
    monkeypatch.setattr(svc, "_load_graph_neighbors", _graph)
    monkeypatch.setattr(svc, "_load_fact_layers", _facts)
    monkeypatch.setattr(svc.goal_loop_service, "build_context", _goal_context)

    package = await svc.build_package(
        query="What does the release depend on?",
        memories=[
            {
                "id": "state::entity::release_train::0",
                "title": "State entity:release train",
                "content_preview": "[State] Release train depends_on: DB migrations",
                "score": 0.98,
                "extra_metadata": {
                    "fact_support": {
                        "subject": "Release train",
                        "predicate": "depends_on",
                        "object": "DB migrations",
                        "confidence": 0.97,
                        "status": "active",
                    },
                    "fact_supporting_facts": [
                        {
                            "subject": "Release train",
                            "predicate": "owner",
                            "object": "Platform team",
                            "confidence": 0.81,
                            "status": "active",
                        }
                    ],
                },
            }
        ],
    )

    assert len(package["facts"]) == 2
    assert package["facts"][0]["source_type"] == "state_space"
    assert package["facts"][0]["object"] == "DB migrations"


@pytest.mark.asyncio
async def test_cognitive_evidence_package_builds_entity_resolution_context(monkeypatch):
    svc = CognitiveEvidenceService(AsyncMock(), org_id="org-1")

    async def _episodes(memory_ids):
        return {}

    async def _semantic(memory_ids, episode_ids):
        return []

    async def _topics(topic_ids):
        return []

    async def _graph(episodes, semantic_nodes, topics):
        return []

    async def _facts(memory_ids):
        return {
            "facts": [
                {
                    "fact_id": "fact-1",
                    "subject": "Caroline",
                    "predicate": "friend",
                    "object": "Melanie",
                    "confidence": 0.92,
                    "status": "active",
                    "source_memory_id": "m1",
                }
            ],
            "contradictions": [],
        }

    async def _goal_context():
        return {
            "active_goals": [],
            "knowledge_gaps": [],
            "suggested_goals": [],
            "world_state": {"recent_changes": [], "highlighted_entities": []},
            "loop_health": {},
        }

    snapshot = SimpleNamespace(
        scope_key="caroline",
        state_version=4,
        symbolic_state={
            "aliases": ["Caroline", "Carrie"],
            "facts": [
                {
                    "subject": "Caroline",
                    "predicate": "friend",
                    "object": "Melanie",
                    "confidence": 0.92,
                    "status": "active",
                    "source_memory_id": "m1",
                },
                {
                    "subject": "Caroline",
                    "predicate": "summer_plan",
                    "object": "charity race training",
                    "confidence": 0.88,
                    "status": "active",
                    "source_memory_id": "m2",
                },
            ],
            "recent_memories": [
                {
                    "memory_id": "m1",
                    "title": "Chat",
                    "content_preview": "Caroline told Melanie about the charity race.",
                }
            ],
        },
    )

    execute_result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [snapshot]))
    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result)
    svc = CognitiveEvidenceService(session, org_id="org-1")

    monkeypatch.setattr(svc, "_load_unified_episodes", _episodes)
    monkeypatch.setattr(svc, "_load_semantic_nodes", _semantic)
    monkeypatch.setattr(svc, "_load_topics", _topics)
    monkeypatch.setattr(svc, "_load_graph_neighbors", _graph)
    monkeypatch.setattr(svc, "_load_fact_layers", _facts)
    monkeypatch.setattr(svc.goal_loop_service, "build_context", _goal_context)

    package = await svc.build_package(
        query="What did Caroline tell Melanie?",
        memories=[
            {
                "id": "m1",
                "title": "Chat",
                "content_preview": "Caroline told Melanie about the charity race.",
                "entities": {"people": ["Caroline", "Melanie"]},
            }
        ],
        query_intelligence={"extracted_entities": ["Caroline", "Melanie"]},
        planner_context={"question_frame": {"primary_subject": "Caroline", "secondary_entities": ["Melanie"]}},
    )

    entity_context = package["entity_context"]
    assert entity_context["primary_subject"] == "Caroline"
    assert entity_context["entities"][0]["canonical_name"] == "Caroline"
    assert "Carrie" in entity_context["entities"][0]["aliases"]
    assert entity_context["entities"][0]["entity_links"][0]["entity"] == "Melanie"
