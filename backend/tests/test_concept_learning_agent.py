from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.agents.concept_learning_agent import (
    ConceptLearningAgent,
    _concept_name,
    _tokenize,
    jaccard_similarity,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult
from app.models.learned_concept import LearnedConcept
from app.services.concept_registry_service import ConceptRegistryService


def _mem(mem_id: str, content: str, tags: list[str] | None = None) -> dict:
    return {
        "id": mem_id,
        "content": content,
        "tags": tags or [],
    }


def _ctx(memories: list[dict], existing: list[dict], min_cluster_size: int = 3) -> dict:
    return {
        "memory": {
            "enrichment": {
                "memories": memories,
                "existing_concepts": existing,
                "min_cluster_size": min_cluster_size,
            }
        },
        "runtime": {"job_id": "trace-59"},
    }


class TestHelpers:
    def test_tokenize(self):
        assert "database" in _tokenize("Database")

    def test_jaccard_identical(self):
        s = {"a", "b"}
        assert jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint(self):
        assert jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_concept_name_two_terms(self):
        assert _concept_name(["database", "latency", "timeout"]) == "database_latency"

    def test_concept_name_single_term(self):
        assert _concept_name(["database"]) == "database"


class TestHeuristic:
    def test_three_similar_memories_one_concept_zero_noise(self):
        memories = [
            _mem("m1", "database latency timeout", ["database", "latency"]),
            _mem("m2", "slow database queries timeout", ["database", "timeout"]),
            _mem("m3", "latency in database connection pool", ["database", "latency"]),
        ]
        out = run_heuristic(memories=memories, existing_concepts=[], min_cluster_size=3)
        assert len(out["new_concepts"]) == 1
        assert len(out["noise_memories"]) == 0

    def test_two_similar_one_dissimilar_concept_plus_noise(self):
        memories = [
            _mem("m1", "database latency timeout", ["database"]),
            _mem("m2", "database timeout latency", ["database"]),
            _mem("m3", "sales pipeline conversion", ["sales"]),
        ]
        out = run_heuristic(memories=memories, existing_concepts=[], min_cluster_size=2)
        assert len(out["new_concepts"]) == 1
        assert out["noise_memories"] == ["m3"]

    def test_existing_concept_absorbs_new_member(self):
        existing = [{"concept_name": "database_latency", "canonical_terms": ["database", "latency"], "member_memory_ids": ["m0"]}]
        memories = [_mem("m1", "database latency spike", ["database", "latency"])]
        out = run_heuristic(memories=memories, existing_concepts=existing, min_cluster_size=3)
        assert len(out["updated_concepts"]) == 1
        assert "m1" in out["updated_concepts"][0]["new_member_ids"]

    def test_concept_name_top_tokens_joined(self):
        memories = [
            _mem("m1", "database latency timeout", ["database"]),
            _mem("m2", "database timeout", ["database"]),
            _mem("m3", "latency timeout database", ["database"]),
        ]
        out = run_heuristic(memories=memories, existing_concepts=[], min_cluster_size=3)
        assert out["new_concepts"][0]["concept_name"] in {"database_timeout", "database_latency", "timeout_database", "latency_database"}

    def test_canonical_terms_max_five(self):
        memories = [
            _mem("m1", "a b c d e f g", []),
            _mem("m2", "a b c d e", []),
            _mem("m3", "a b c d", []),
        ]
        out = run_heuristic(memories=memories, existing_concepts=[], min_cluster_size=3)
        assert len(out["new_concepts"][0]["canonical_terms"]) <= 5

    def test_min_cluster_size_blocks_small_clusters(self):
        memories = [
            _mem("m1", "database latency", []),
            _mem("m2", "database timeout", []),
            _mem("m3", "latency timeout", []),
        ]
        out = run_heuristic(memories=memories, existing_concepts=[], min_cluster_size=5)
        assert out["new_concepts"] == []
        assert set(out["noise_memories"]) == {"m1", "m2", "m3"}

    def test_noise_memories_contains_unassigned(self):
        out = run_heuristic(
            memories=[_mem("m1", "alpha", []), _mem("m2", "beta", [])],
            existing_concepts=[],
            min_cluster_size=3,
        )
        assert set(out["noise_memories"]) == {"m1", "m2"}

    def test_total_concepts_found_counts_new_and_updated(self):
        existing = [{"concept_name": "database_latency", "canonical_terms": ["database", "latency"], "member_memory_ids": ["m0"]}]
        memories = [
            _mem("m1", "database latency spike", ["database", "latency"]),
            _mem("m2", "api outage timeout", ["api", "timeout"]),
            _mem("m3", "api timeout error", ["api", "timeout"]),
        ]
        out = run_heuristic(memories=memories, existing_concepts=existing, min_cluster_size=2)
        assert out["total_concepts_found"] >= 1

    def test_confidence_in_range(self):
        out = run_heuristic(memories=[], existing_concepts=[], min_cluster_size=3)
        assert 0.0 <= out["confidence"] <= 1.0

    def test_empty_memories_graceful(self):
        out = run_heuristic(memories=[], existing_concepts=[], min_cluster_size=3)
        assert out["new_concepts"] == []
        assert out["updated_concepts"] == []


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = ConceptLearningAgent()
        with patch("app.agents.concept_learning_agent.settings") as mock_settings:
            mock_settings.CONCEPT_LEARNING_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([], []))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = ConceptLearningAgent()
        with patch("app.agents.concept_learning_agent.settings") as mock_settings:
            mock_settings.CONCEPT_LEARNING_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([], []))
        assert result.trace_id == "trace-59"

    @pytest.mark.asyncio
    async def test_strategy_fallback(self):
        agent = ConceptLearningAgent()
        with patch("app.agents.concept_learning_agent.settings") as mock_settings:
            mock_settings.CONCEPT_LEARNING_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([], []))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = ConceptLearningAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "new_concepts": [],
                "updated_concepts": [],
                "noise_memories": [],
                "total_concepts_found": 0,
                "confidence": 0.7,
                "rationale": "llm",
            }
        )
        with patch("app.agents.concept_learning_agent.settings") as mock_settings, patch(
            "app.agents.concept_learning_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.CONCEPT_LEARNING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx([], []))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = ConceptLearningAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.concept_learning_agent.settings") as mock_settings, patch(
            "app.agents.concept_learning_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.CONCEPT_LEARNING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx([], []))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="ConceptLearningAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.7,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "new_concepts": [],
            "updated_concepts": [],
            "noise_memories": [],
            "total_concepts_found": 0,
            "confidence": 0.7,
        }

    def test_validate_outputs_passes(self):
        ConceptLearningAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_new_concepts_type_raises(self):
        with pytest.raises(ValueError, match="new_concepts"):
            ConceptLearningAgent().validate_outputs(self._result(dict(self._valid_outputs(), new_concepts="x")))

    def test_updated_concepts_type_raises(self):
        with pytest.raises(ValueError, match="updated_concepts"):
            ConceptLearningAgent().validate_outputs(self._result(dict(self._valid_outputs(), updated_concepts="x")))

    def test_noise_memories_type_raises(self):
        with pytest.raises(ValueError, match="noise_memories"):
            ConceptLearningAgent().validate_outputs(self._result(dict(self._valid_outputs(), noise_memories="x")))

    def test_total_concepts_found_type_raises(self):
        with pytest.raises(ValueError, match="total_concepts_found"):
            ConceptLearningAgent().validate_outputs(self._result(dict(self._valid_outputs(), total_concepts_found="x")))

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="ConceptLearningAgent",
            agent_version="v1",
            memory_id="m1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        ConceptLearningAgent().validate_outputs(res)


class TestRegistry:
    def test_registry_alias_primary(self):
        assert isinstance(get_agent("concept_learning"), ConceptLearningAgent)

    def test_registry_alias_compact(self):
        assert isinstance(get_agent("conceptlearning"), ConceptLearningAgent)

    def test_registry_alias_agent(self):
        assert isinstance(get_agent("ConceptLearningAgent"), ConceptLearningAgent)

    def test_registry_alias_short(self):
        assert isinstance(get_agent("concepts"), ConceptLearningAgent)


class TestConceptRegistryService:
    @pytest.mark.asyncio
    async def test_upsert_inserts_new_concepts_count(self, db_session, test_org_id: str):
        svc = ConceptRegistryService()
        affected = await svc.upsert_concepts(
            db=db_session,
            org_id=test_org_id,
            new_concepts=[
                {
                    "concept_name": "database_latency",
                    "member_ids": ["m1", "m2"],
                    "canonical_terms": ["database", "latency"],
                    "confidence": 0.8,
                }
            ],
            updated_concepts=[],
        )
        assert affected == 1

    @pytest.mark.asyncio
    async def test_get_concepts_for_org(self, db_session, test_org_id: str):
        svc = ConceptRegistryService()
        await svc.upsert_concepts(
            db=db_session,
            org_id=test_org_id,
            new_concepts=[{"concept_name": "db_latency", "member_ids": ["m1"], "canonical_terms": ["db"], "confidence": 0.7}],
            updated_concepts=[],
        )
        rows = await svc.get_concepts_for_org(db=db_session, org_id=test_org_id, limit=50)
        assert rows

    @pytest.mark.asyncio
    async def test_find_concept_for_memory(self, db_session, test_org_id: str):
        svc = ConceptRegistryService()
        await svc.upsert_concepts(
            db=db_session,
            org_id=test_org_id,
            new_concepts=[{"concept_name": "db_latency", "member_ids": ["m1", "m2"], "canonical_terms": ["db"], "confidence": 0.7}],
            updated_concepts=[],
        )
        row = await svc.find_concept_for_memory(db=db_session, org_id=test_org_id, memory_id="m2")
        assert row is not None
        assert row.concept_name == "db_latency"

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, db_session, test_org_id: str):
        svc = ConceptRegistryService()
        await svc.upsert_concepts(
            db=db_session,
            org_id=test_org_id,
            new_concepts=[{"concept_name": "db_latency", "member_ids": ["m1"], "canonical_terms": ["db"], "confidence": 0.7}],
            updated_concepts=[],
        )
        affected = await svc.upsert_concepts(
            db=db_session,
            org_id=test_org_id,
            new_concepts=[],
            updated_concepts=[{"concept_name": "db_latency", "member_ids": ["m2"], "canonical_terms": ["db", "latency"], "confidence": 0.8}],
        )
        assert affected == 1
        row = await svc.find_concept_for_memory(db=db_session, org_id=test_org_id, memory_id="m2")
        assert row is not None

    @pytest.mark.asyncio
    async def test_find_concept_for_missing_memory_returns_none(self, db_session, test_org_id: str):
        row = await ConceptRegistryService().find_concept_for_memory(
            db=db_session,
            org_id=test_org_id,
            memory_id="missing-id",
        )
        assert row is None


class TestModelShape:
    def test_model_has_required_fields(self):
        fields = LearnedConcept.__table__.columns.keys()
        for name in ("org_id", "concept_name", "member_memory_ids", "canonical_terms", "occurrence_count", "first_seen", "last_seen", "confidence"):
            assert name in fields
