from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.services.cognitive_read_planner as planner_module
from app.services.cognitive_read_planner import CognitiveReadPlanner
from app.services.grounded_answer_service import GroundedAnswerService


@pytest.mark.asyncio
async def test_cognitive_read_planner_merges_multi_source_candidates(monkeypatch):
    gateway = SimpleNamespace(
        read=AsyncMock(
            return_value=SimpleNamespace(
                memories=[{"id": "m1", "title": "Base memory"}, {"id": "m2", "title": "Hierarchy memory"}],
                total=2,
                context_assembled=True,
                retrieval_confidence=0.77,
                reasoning_steps=[{"step": "gateway_rank"}],
                compression_ratio=0.45,
                information_density=0.81,
            )
        )
    )
    evidence_service = SimpleNamespace(
        build_package=AsyncMock(
            return_value={
                "memory_hits": [{"memory_id": "m1"}],
                "evidence_quality": {"memory_count": 2},
            }
        )
    )
    planner = CognitiveReadPlanner(
        AsyncMock(),
        user_id="user-1",
        org_id="org-1",
        gateway=gateway,
        memory_service=SimpleNamespace(),
        evidence_service=evidence_service,
    )

    monkeypatch.setattr(
        planner_module,
        "run_query_intelligence",
        lambda query, enrichment: {
            "query_intent": "analyze",
            "extracted_entities": ["release train"],
            "dynamic_filters": {"tags": ["ops"]},
        },
    )

    async def _search_memories(**kwargs):
        return [{"id": "m1", "title": "Search hit", "score": 0.3, "tags": ["ops"]}]

    async def _hierarchical_candidates(**kwargs):
        return [{"id": "m2", "title": "Hierarchy hit", "score": 0.4}]

    async def _coverage_candidates(**kwargs):
        return [
            {"id": "m1", "title": "Coverage hit", "score": 0.9},
            {"id": "m3", "title": "Coverage hit 2", "score": 0.2},
        ]

    monkeypatch.setattr(planner, "_search_memories", _search_memories)
    monkeypatch.setattr(planner, "_fact_backed_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_hierarchical_candidates", _hierarchical_candidates)
    monkeypatch.setattr(planner, "_coverage_candidates", _coverage_candidates)

    planned = await planner.plan_and_read(
        query="Why did the release slip?",
        limit=2,
        scope="personal",
        hybrid=True,
    )

    assert planned.retrieval_strategy == "memory_search+hierarchy_search+coverage_retrieval"
    assert planned.target_memory_level == "semantic+facts+graph"
    assert "release train" in planned.expanded_query
    assert planned.reasoning_steps[-1]["step"] == "gateway_rank"
    gateway.read.assert_awaited_once()
    gateway_kwargs = gateway.read.await_args.kwargs
    assert [item["id"] for item in gateway_kwargs["memories"]] == ["m1", "m2", "m3"]
    assert gateway_kwargs["memories"][0]["score"] == 0.9
    evidence_service.build_package.assert_awaited_once()
    planner_context = evidence_service.build_package.await_args.kwargs["planner_context"]
    assert planner_context["sources_used"] == ["memory_search", "hierarchy_search", "coverage_retrieval"]
    assert planned.evidence_package["evidence_quality"]["memory_count"] == 2


@pytest.mark.asyncio
async def test_cognitive_read_planner_routes_timeline_queries_through_temporal_reasoning(monkeypatch):
    async def _gateway_read(**kwargs):
        memories = list(kwargs["memories"])
        return SimpleNamespace(
            memories=memories,
            total=len(memories),
            context_assembled=bool(memories),
            retrieval_confidence=0.82,
            reasoning_steps=[{"step": "gateway_rank"}],
            compression_ratio=0.52,
            information_density=0.74,
        )

    async def _build_package(**kwargs):
        return {
            "memory_hits": kwargs["memories"],
            "temporal_reasoning": kwargs["planner_context"].get("temporal_reasoning"),
            "evidence_quality": {"memory_count": len(kwargs["memories"])},
        }

    planner = CognitiveReadPlanner(
        AsyncMock(),
        user_id="user-1",
        org_id="org-1",
        gateway=SimpleNamespace(read=AsyncMock(side_effect=_gateway_read)),
        memory_service=SimpleNamespace(),
        evidence_service=SimpleNamespace(build_package=AsyncMock(side_effect=_build_package)),
    )

    monkeypatch.setattr(
        planner_module,
        "run_query_intelligence",
        lambda query, enrichment: {
            "query_intent": "find_timeline",
            "extracted_entities": ["release train"],
            "dynamic_filters": {"has_temporal_data": True},
        },
    )

    async def _search_memories(**kwargs):
        return [
            {"id": "m1", "title": "Rollout started", "score": 0.4, "occurred_at": "2026-05-03T10:00:00+00:00"},
            {"id": "m2", "title": "Migration finished", "score": 0.8, "occurred_at": "2026-05-01T10:00:00+00:00"},
        ]

    async def _hierarchical_candidates(**kwargs):
        return [{"id": "m3", "title": "Validation completed", "score": 0.5, "occurred_at": "2026-05-04T10:00:00+00:00"}]

    async def _episode_neighbors(memory_ids, *, limit):
        return [{"id": "m4", "title": "Rollback blocked", "score": 0.6, "occurred_at": "2026-05-02T10:00:00+00:00"}]

    monkeypatch.setattr(planner, "_search_memories", _search_memories)
    monkeypatch.setattr(planner, "_fact_backed_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_hierarchical_candidates", _hierarchical_candidates)
    monkeypatch.setattr(planner, "_episode_neighbor_candidates", _episode_neighbors)
    monkeypatch.setattr(
        planner.temporal_service,
        "temporal_query",
        AsyncMock(
            return_value={
                "timeline": [
                    {"memory_id": "m2", "occurred_at": "2026-05-01T10:00:00+00:00", "title": "Migration finished"},
                    {"memory_id": "m4", "occurred_at": "2026-05-02T10:00:00+00:00", "title": "Rollback blocked"},
                    {"memory_id": "m1", "occurred_at": "2026-05-03T10:00:00+00:00", "title": "Rollout started"},
                    {"memory_id": "m3", "occurred_at": "2026-05-04T10:00:00+00:00", "title": "Validation completed"},
                ],
                "memory_ids": ["m2", "m4", "m1", "m3"],
                "anchor_count": 4,
            }
        ),
    )

    planned = await planner.plan_and_read(
        query="When did the release sequence happen?",
        limit=4,
        scope="personal",
    )

    assert planned.retrieval_strategy == "memory_search+hierarchy_search+episode_neighbors+temporal_reasoning"
    assert [item["id"] for item in planned.memories] == ["m2", "m4", "m1", "m3"]
    assert planned.reasoning_steps[-2]["step"] == "temporal_reasoning"
    assert planned.evidence_package["temporal_reasoning"]["memory_ids"][0] == "m2"


@pytest.mark.asyncio
async def test_cognitive_read_planner_routes_multi_hop_queries_through_chain(monkeypatch):
    async def _gateway_read(**kwargs):
        memories = list(kwargs["memories"])
        return SimpleNamespace(
            memories=memories,
            total=len(memories),
            context_assembled=bool(memories),
            retrieval_confidence=0.79,
            reasoning_steps=[{"step": "gateway_rank"}],
            compression_ratio=0.5,
            information_density=0.78,
        )

    async def _build_package(**kwargs):
        return {
            "memory_hits": kwargs["memories"],
            "multi_hop_trace": kwargs["planner_context"].get("multi_hop_trace"),
            "evidence_quality": {"memory_count": len(kwargs["memories"])},
        }

    planner = CognitiveReadPlanner(
        AsyncMock(),
        user_id="user-1",
        org_id="org-1",
        gateway=SimpleNamespace(read=AsyncMock(side_effect=_gateway_read)),
        memory_service=SimpleNamespace(),
        evidence_service=SimpleNamespace(build_package=AsyncMock(side_effect=_build_package)),
    )

    monkeypatch.setattr(
        planner_module,
        "run_query_intelligence",
        lambda query, enrichment: {
            "query_intent": "compare",
            "extracted_entities": ["Project Atlas", "Vendor Beta"],
            "dynamic_filters": {},
        },
    )

    search_calls: list[str] = []

    async def _search_memories(**kwargs):
        query = kwargs["query"]
        search_calls.append(query)
        if "Project Atlas" in query and "Vendor Beta" in query and "{step_" not in query:
            return [{"id": "m3", "title": "Dependency note", "content_preview": "Atlas depends on Beta migration.", "score": 0.88}]
        if "Project Atlas" in query:
            return [{"id": "m1", "title": "Atlas update", "content_preview": "Atlas launch moved after vendor review.", "score": 0.8}]
        if "Vendor Beta" in query:
            return [{"id": "m2", "title": "Vendor note", "content_preview": "Beta migration delayed the rollout.", "score": 0.82}]
        return [{"id": "m0", "title": "Planning summary", "content_preview": "The launch plan and migration owner were discussed.", "score": 0.45}]

    monkeypatch.setattr(planner, "_search_memories", _search_memories)
    monkeypatch.setattr(planner, "_fact_backed_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_hierarchical_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_coverage_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_episode_neighbor_candidates", AsyncMock(return_value=[]))

    planned = await planner.plan_and_read(
        query="Compare the launch plan and the migration owner",
        limit=4,
        scope="personal",
    )

    assert planned.retrieval_strategy == "memory_search+multi_hop_chain"
    assert any("Project Atlas" in query for query in search_calls)
    assert any("Vendor Beta" in query for query in search_calls)
    assert planned.evidence_package["multi_hop_trace"]
    assert planned.reasoning_steps[-2]["step"] == "multi_hop_chain"
    assert set(item["id"] for item in planned.memories[:3]) == {"m3", "m1", "m2"}


@pytest.mark.asyncio
async def test_cognitive_read_planner_routes_compositional_single_entity_queries_through_chain(monkeypatch):
    async def _gateway_read(**kwargs):
        memories = list(kwargs["memories"])
        return SimpleNamespace(
            memories=memories,
            total=len(memories),
            context_assembled=bool(memories),
            retrieval_confidence=0.8,
            reasoning_steps=[{"step": "gateway_rank"}],
            compression_ratio=0.48,
            information_density=0.79,
        )

    async def _build_package(**kwargs):
        return {
            "memory_hits": kwargs["memories"],
            "multi_hop_trace": kwargs["planner_context"].get("multi_hop_trace"),
            "evidence_quality": {"memory_count": len(kwargs["memories"])},
        }

    planner = CognitiveReadPlanner(
        AsyncMock(),
        user_id="user-1",
        org_id="org-1",
        gateway=SimpleNamespace(read=AsyncMock(side_effect=_gateway_read)),
        memory_service=SimpleNamespace(),
        evidence_service=SimpleNamespace(build_package=AsyncMock(side_effect=_build_package)),
    )

    monkeypatch.setattr(
        planner_module,
        "run_query_intelligence",
        lambda query, enrichment: {
            "query_intent": "retrieve",
            "extracted_entities": ["Rollout"],
            "dynamic_filters": {},
        },
    )

    search_calls: list[str] = []

    async def _search_memories(**kwargs):
        query = kwargs["query"]
        search_calls.append(query)
        lowered = query.lower()
        if "causes, blockers, dependencies" in lowered:
            return [{"id": "m2", "title": "Blocker", "content_preview": "The rollout stayed blocked because the data migration failed.", "score": 0.92}]
        if "timeline, sequence, and dated events" in lowered:
            return [{"id": "m3", "title": "Timeline", "content_preview": "Migration failed on Tuesday before the rollout checkpoint.", "score": 0.87}]
        return [{"id": "m1", "title": "Seed", "content_preview": "The rollout was still blocked after the migration check.", "score": 0.6}]

    monkeypatch.setattr(planner, "_search_memories", _search_memories)
    monkeypatch.setattr(planner, "_fact_backed_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_hierarchical_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_coverage_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_episode_neighbor_candidates", AsyncMock(return_value=[]))

    planned = await planner.plan_and_read(
        query="Why was the rollout still blocked after the migration?",
        limit=4,
        scope="personal",
    )

    assert planned.retrieval_strategy == "memory_search+multi_hop_chain"
    assert planned.evidence_package["multi_hop_trace"]
    assert any("Causes, blockers, dependencies" in query for query in search_calls)
    assert any("Timeline, sequence, and dated events" in query for query in search_calls)


@pytest.mark.asyncio
async def test_cognitive_read_planner_merges_fact_backed_candidates(monkeypatch):
    gateway = SimpleNamespace(
        read=AsyncMock(
            return_value=SimpleNamespace(
                memories=[{"id": "m1", "title": "Search hit"}, {"id": "m_fact", "title": "Fact-backed hit"}],
                total=2,
                context_assembled=True,
                retrieval_confidence=0.84,
                reasoning_steps=[{"step": "gateway_rank"}],
                compression_ratio=0.5,
                information_density=0.8,
            )
        )
    )
    evidence_service = SimpleNamespace(
        build_package=AsyncMock(return_value={"memory_hits": [{"memory_id": "m_fact"}], "evidence_quality": {"memory_count": 2}})
    )
    planner = CognitiveReadPlanner(
        SimpleNamespace(),
        user_id="user-1",
        org_id="org-1",
        gateway=gateway,
        memory_service=SimpleNamespace(),
        evidence_service=evidence_service,
    )

    monkeypatch.setattr(
        planner_module,
        "run_query_intelligence",
        lambda query, enrichment: {
            "query_intent": "retrieve",
            "extracted_entities": ["Release train"],
            "dynamic_filters": {},
        },
    )
    monkeypatch.setattr(planner, "_search_memories", AsyncMock(return_value=[{"id": "m1", "title": "Search hit", "score": 0.4}]))
    monkeypatch.setattr(planner, "_fact_backed_candidates", AsyncMock(return_value=[{"id": "m_fact", "title": "Fact-backed hit", "score": 0.9}]))

    planned = await planner.plan_and_read(
        query="What did the release depend on?",
        limit=2,
        scope="personal",
    )

    assert "fact_retrieval" in planned.retrieval_strategy
    assert any(step["step"] == "fact_retrieval" for step in planned.reasoning_steps)
    gateway_kwargs = gateway.read.await_args.kwargs
    assert [item["id"] for item in gateway_kwargs["memories"]] == ["m1", "m_fact"]


@pytest.mark.asyncio
async def test_cognitive_read_planner_prefers_state_space_candidates(monkeypatch):
    gateway = SimpleNamespace(
        read=AsyncMock(
            return_value=SimpleNamespace(
                memories=[{"id": "state::entity::release_train::0", "title": "State entity:release_train"}],
                total=1,
                context_assembled=True,
                retrieval_confidence=0.91,
                reasoning_steps=[{"step": "gateway_rank"}],
                compression_ratio=0.35,
                information_density=0.88,
            )
        )
    )
    evidence_service = SimpleNamespace(
        build_package=AsyncMock(return_value={"memory_hits": [{"memory_id": "state::entity::release_train::0"}], "evidence_quality": {"memory_count": 1}})
    )
    state_service = SimpleNamespace(
        lookup_candidates=AsyncMock(
            return_value=[
                {
                    "id": "state::entity::release_train::0",
                    "title": "State entity:release_train",
                    "content_preview": "[State] Release train depends_on: DB migrations",
                    "score": 0.97,
                    "extra_metadata": {
                        "fact_support": {
                            "subject": "Release train",
                            "predicate": "depends_on",
                            "object": "DB migrations",
                        }
                    },
                }
            ]
        )
    )
    planner = CognitiveReadPlanner(
        SimpleNamespace(),
        user_id="user-1",
        org_id="org-1",
        gateway=gateway,
        memory_service=SimpleNamespace(),
        evidence_service=evidence_service,
        state_service=state_service,
    )

    monkeypatch.setattr(
        planner_module,
        "run_query_intelligence",
        lambda query, enrichment: {
            "query_intent": "retrieve",
            "extracted_entities": ["Release train"],
            "dynamic_filters": {},
        },
    )
    monkeypatch.setattr(planner, "_search_memories", AsyncMock(return_value=[]))
    monkeypatch.setattr(planner, "_fact_backed_candidates", AsyncMock(return_value=[]))

    planned = await planner.plan_and_read(
        query="What did the release depend on?",
        limit=2,
        scope="personal",
    )

    assert planned.retrieval_strategy == "state_space+memory_search"
    assert planned.reasoning_steps[1]["step"] == "state_space"
    state_service.lookup_candidates.assert_awaited_once()


@pytest.mark.asyncio
async def test_grounded_answer_service_returns_uncertainty_without_evidence():
    gateway = SimpleNamespace(answer=AsyncMock())

    result = await GroundedAnswerService(gateway).answer(
        question="What happened?",
        evidence_package={},
        memories=[],
    )

    assert result.grounded is False
    assert result.confidence == 0.0
    assert result.uncertainty_reason == "no_grounded_evidence"
    gateway.answer.assert_not_called()


@pytest.mark.asyncio
async def test_grounded_answer_service_builds_prompt_and_support():
    gateway = SimpleNamespace(
        answer=AsyncMock(
            return_value=SimpleNamespace(
                answer="It depended on DB migrations.",
                model="test-model",
                used_llm=True,
                context_turns=2,
                llm_error=None,
            )
        )
    )
    service = GroundedAnswerService(gateway)

    result = await service.answer(
        question="What did the release depend on?",
        evidence_package={
            "memory_hits": [
                {
                    "memory_id": "m1",
                    "title": "Release memo",
                    "content_preview": "Migration must complete before rollout.",
                }
            ],
            "facts": [
                {
                    "subject": "release train",
                    "predicate": "depends_on",
                    "object": "db migrations",
                    "status": "active",
                    "confidence": 0.93,
                }
            ],
            "contradictions": [
                {
                    "reason": "One fact says the migration was blocked while another says it was complete.",
                    "severity": "high",
                    "fact_a_object": "blocked",
                    "fact_b_object": "complete",
                }
            ],
            "episodes": [{"episode_id": "ep-1", "title": "Release train", "summary": "Deployment chain"}],
            "semantic_nodes": [{"semantic_node_id": "sn-1", "content": "DB migrations gate rollout."}],
            "temporal_reasoning": {
                "timeline": [
                    {
                        "memory_id": "m1",
                        "occurred_at": "2026-05-01T10:00:00+00:00",
                        "title": "Migration finished",
                    }
                ]
            },
            "entity_context": {
                "primary_subject": "release train",
                "entities": [
                    {
                        "canonical_name": "release train",
                        "is_primary_subject": True,
                        "aliases": ["release train"],
                        "facts": [
                            {
                                "subject": "release train",
                                "predicate": "depends_on",
                                "object": "db migrations",
                            }
                        ],
                        "entity_links": [
                            {"direction": "out", "predicate": "depends_on", "entity": "DB migrations"}
                        ],
                        "memory_mentions": [
                            {"content_preview": "Migration must complete before rollout."}
                        ],
                    }
                ],
            },
            "multi_hop_trace": [
                {"step_index": 0, "query": "release dependencies", "memory_count": 2, "confidence": 0.8}
            ],
            "goal_context": {
                "active_goals": [{"title": "Ship the release", "status": "active", "urgency": 0.8}],
                "knowledge_gaps": [{"description": "Need migration status", "gap_type": "missing_fact"}],
                "world_state": {
                    "recent_changes": [{"entity": "deployment", "change_type": "delayed", "description": "Migration blocked rollout"}]
                },
            },
            "evidence_quality": {
                "avg_memory_score": 0.72,
                "avg_semantic_quality": 0.9,
                "avg_feedback_signal": 0.4,
            },
        },
    )

    gateway.answer.assert_awaited_once()
    gateway_kwargs = gateway.answer.await_args.kwargs
    prompt = gateway_kwargs["prompt_override"]
    assert "QUESTION PROFILE:" in prompt
    assert "primary_subject=release train" in prompt
    assert "ANSWER HINTS:" in prompt
    assert "FACTS:" in prompt
    assert "TIMELINE:" in prompt
    assert "ENTITY RESOLUTION:" in prompt
    assert "SEMANTIC NODES:" in prompt
    assert "GRAPH SIGNALS:" in prompt
    assert "GOAL CONTEXT:" in prompt
    assert "MULTI-HOP TRACE:" in prompt
    assert "Prefer FACT lines over narrative memory text." in prompt
    assert "Answer about the PRIMARY SUBJECT, not a distractor." in prompt
    assert "return the object or outcome" in prompt
    assert "QUESTION: What did the release depend on?" in prompt
    assert gateway_kwargs["memories"] == []
    assert gateway_kwargs["num_ctx"] == 1536
    assert result.grounded is True
    assert result.answer == "It depended on DB migrations."
    assert result.uncertainty_reason == "contradictory_evidence_present"
    assert result.used_llm is True
    assert result.confidence > 0.0
    assert result.support[0].startswith("fact:")
    assert any(line.startswith("memory:") for line in result.support)
    assert any(line.startswith("timeline:") for line in result.support)
    assert any(line.startswith("entity:") for line in result.support)


@pytest.mark.asyncio
async def test_grounded_answer_service_returns_state_space_direct_answer_without_llm():
    gateway = SimpleNamespace(answer=AsyncMock())
    service = GroundedAnswerService(gateway)

    result = await service.answer(
        question="What did the release depend on?",
        evidence_package={
            "memory_hits": [
                {
                    "memory_id": "state::entity::release_train::0",
                    "title": "State entity:release_train",
                    "content_preview": "[State] Release train depends_on: DB migrations",
                }
            ],
            "facts": [
                {
                    "subject": "Release train",
                    "predicate": "depends_on",
                    "object": "DB migrations",
                    "status": "active",
                    "confidence": 0.97,
                    "source_memory_id": "state::entity::release_train::0",
                    "source_type": "state_space",
                }
            ],
            "evidence_quality": {"avg_memory_score": 0.9, "avg_semantic_quality": 0.7},
        },
    )

    assert result.answer == "DB migrations"
    assert result.answer_source == "state_space_direct"
    assert result.used_llm is False
    gateway.answer.assert_not_called()


@pytest.mark.asyncio
async def test_grounded_answer_service_compresses_verbose_single_value_answers():
    gateway = SimpleNamespace(
        answer=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    answer="Caroline identifies as a transgender woman",
                    model="test-model",
                    used_llm=True,
                    context_turns=2,
                    llm_error=None,
                    answer_source="gateway",
                    llm_failure_mode=None,
                    llm_endpoint="http://vllm-primary:11434",
                ),
                SimpleNamespace(
                    answer="Transgender woman",
                    model="test-model",
                    used_llm=True,
                    context_turns=0,
                    llm_error=None,
                    answer_source="gateway",
                    llm_failure_mode=None,
                    llm_endpoint="http://vllm-primary:11434",
                ),
            ]
        )
    )
    service = GroundedAnswerService(gateway)

    result = await service.answer(
        question="What is Caroline's identity?",
        evidence_package={
            "memory_hits": [
                {
                    "memory_id": "m1",
                    "title": "Identity note",
                    "content_preview": "Caroline identifies as a transgender woman.",
                }
            ],
            "facts": [
                {
                    "subject": "Caroline",
                    "predicate": "identity",
                    "object": "Transgender woman",
                    "status": "active",
                    "confidence": 0.95,
                }
            ],
            "contradictions": [
                {"reason": "Conflicting identity phrasing in source evidence.", "severity": "low"}
            ],
            "evidence_quality": {"avg_memory_score": 0.8, "avg_semantic_quality": 0.7},
        },
    )

    assert gateway.answer.await_count == 2
    compression_prompt = gateway.answer.await_args_list[1].kwargs["prompt_override"]
    assert "Rewrite the draft answer into the shortest grounded answer" in compression_prompt
    assert "DRAFT ANSWER: Caroline identifies as a transgender woman" in compression_prompt
    assert result.answer == "Transgender woman"
    assert result.answer_source == "gateway_compressed"
    assert result.llm_endpoint == "http://vllm-primary:11434"


def test_grounded_answer_service_question_profile_tracks_primary_subject_and_distractors():
    profile = GroundedAnswerService._question_profile(
        "What are Melanie's plans for the summer with respect to adoption?",
        {
            "query_intelligence": {"extracted_entities": ["Melanie", "Caroline"]},
            "facts": [
                {"subject": "Melanie", "predicate": "summer_plan", "object": "researching adoption agencies"}
            ],
        },
    )

    assert profile["primary_subject"] == "Melanie"
    assert "Caroline" in profile["distractors"]
