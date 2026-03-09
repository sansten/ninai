"""Tests for FeatureReadinessService — Phase 28."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.feature_readiness_service import (
    AGENT_SPECS,
    FeatureReadinessService,
    ReadinessSpec,
    _evaluate_spec,
    _fetch_data_snapshot,
    _build_response_item,
    _build_uneval_item,
)
from app.models.feature_readiness import FeatureReadinessConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _snapshot(memory_count: int = 0, active_days: int = 0, source_count: int = 1) -> dict:
    return {
        "memory_count": memory_count,
        "active_days": active_days,
        "source_count": source_count,
    }


def _spec(
    min_memories: int = 10,
    min_days_active: int = 3,
    min_sources: int = 1,
    default_params: dict | None = None,
) -> ReadinessSpec:
    return ReadinessSpec(
        description="Test agent",
        min_memories=min_memories,
        min_days_active=min_days_active,
        min_sources=min_sources,
        default_params=default_params or {},
    )


class _MockConfig:
    """Lightweight stand-in for FeatureReadinessConfig (avoids SQLAlchemy mapping)."""

    def __init__(
        self,
        agent_name: str = "TestAgent",
        enabled: bool = True,
        custom_params: dict | None = None,
        readiness_state: dict | None = None,
        evaluated_at: datetime | None = None,
    ):
        self.organization_id = "org-1"
        self.agent_name = agent_name
        self.enabled = enabled
        self.custom_params = custom_params
        self.readiness_state = readiness_state if readiness_state is not None else {}
        self.evaluated_at = evaluated_at
        self.notes = None


def _config(
    agent_name: str = "TestAgent",
    enabled: bool = True,
    custom_params: dict | None = None,
    readiness_state: dict | None = None,
    evaluated_at: datetime | None = None,
) -> _MockConfig:
    return _MockConfig(
        agent_name=agent_name,
        enabled=enabled,
        custom_params=custom_params,
        readiness_state=readiness_state,
        evaluated_at=evaluated_at,
    )


# ---------------------------------------------------------------------------
# TestAgentSpecs
# ---------------------------------------------------------------------------

class TestAgentSpecs:
    def test_all_expected_agents_present(self):
        expected = {
            "SemanticNormalizationAgent",
            "EntityResolutionAgent",
            "ContextAmplifierAgent",
            "SiloPropagationAgent",
            "OrgAttentionAgent",
            "ProactiveMemoryPushAgent",
            "CausalReasoningAgent",
            "ConflictDetectionAgent",
            "AdaptiveConflictResolutionAgent",
            "MemoryDecayAgent",
            "MemoryConsolidationAgent",
            "TemporalReasoningAgent",
            "EpisodicGroupingAgent",
            "CredibilityAgent",
            "PlaybookAgent",
            "GoalDecompositionAgent",
            "UncertaintyReportingAgent",
            "NarrativeSynthesisAgent",
            "FeedbackIntegrationAgent",
            "AnomalyDetectionAgent",
            "QueryIntelligenceAgent",
        }
        assert expected == set(AGENT_SPECS.keys())

    def test_each_spec_has_description(self):
        for name, spec in AGENT_SPECS.items():
            assert spec.description, f"{name} has empty description"

    def test_semantic_normalization_lowest_threshold(self):
        spec = AGENT_SPECS["SemanticNormalizationAgent"]
        assert spec.min_memories == 1
        assert spec.min_days_active == 0
        assert spec.min_sources == 1

    def test_query_intelligence_lowest_threshold(self):
        spec = AGENT_SPECS["QueryIntelligenceAgent"]
        assert spec.min_memories == 1
        assert spec.min_days_active == 0

    def test_memory_decay_requires_30_days(self):
        spec = AGENT_SPECS["MemoryDecayAgent"]
        assert spec.min_days_active >= 30

    def test_silo_propagation_requires_two_sources(self):
        spec = AGENT_SPECS["SiloPropagationAgent"]
        assert spec.min_sources >= 2

    def test_entity_resolution_higher_threshold_than_semantic(self):
        era = AGENT_SPECS["EntityResolutionAgent"]
        sna = AGENT_SPECS["SemanticNormalizationAgent"]
        assert era.min_memories > sna.min_memories

    def test_anomaly_detection_requires_history(self):
        spec = AGENT_SPECS["AnomalyDetectionAgent"]
        assert spec.min_memories >= 100
        assert spec.min_days_active >= 14

    def test_all_min_memories_positive(self):
        for name, spec in AGENT_SPECS.items():
            assert spec.min_memories >= 1, f"{name} min_memories must be >= 1"

    def test_all_default_params_are_dicts(self):
        for name, spec in AGENT_SPECS.items():
            assert isinstance(spec.default_params, dict), f"{name} default_params not a dict"


# ---------------------------------------------------------------------------
# TestEvaluateSpec
# ---------------------------------------------------------------------------

class TestEvaluateSpec:
    def test_all_conditions_met_returns_ready(self):
        s = _spec(min_memories=10, min_days_active=3, min_sources=2)
        snap = _snapshot(memory_count=50, active_days=10, source_count=3)
        result = _evaluate_spec(s, snap)
        assert result["is_ready"] is True
        assert result["blocking_reasons"] == []
        assert len(result["reasons"]) == 3

    def test_insufficient_memories_blocks(self):
        s = _spec(min_memories=100)
        snap = _snapshot(memory_count=5, active_days=30, source_count=2)
        result = _evaluate_spec(s, snap)
        assert result["is_ready"] is False
        assert any("memories" in r for r in result["blocking_reasons"])

    def test_insufficient_days_blocks(self):
        s = _spec(min_memories=1, min_days_active=30, min_sources=1)
        snap = _snapshot(memory_count=200, active_days=5, source_count=1)
        result = _evaluate_spec(s, snap)
        assert result["is_ready"] is False
        assert any("days" in r for r in result["blocking_reasons"])

    def test_insufficient_sources_blocks(self):
        s = _spec(min_memories=1, min_days_active=0, min_sources=2)
        snap = _snapshot(memory_count=200, active_days=30, source_count=1)
        result = _evaluate_spec(s, snap)
        assert result["is_ready"] is False
        assert any("sources" in r for r in result["blocking_reasons"])

    def test_multiple_blocking_conditions_all_reported(self):
        s = _spec(min_memories=100, min_days_active=30, min_sources=3)
        snap = _snapshot(memory_count=1, active_days=0, source_count=1)
        result = _evaluate_spec(s, snap)
        assert result["is_ready"] is False
        assert len(result["blocking_reasons"]) == 3

    def test_data_snapshot_included(self):
        s = _spec()
        snap = _snapshot(memory_count=99, active_days=7, source_count=2)
        result = _evaluate_spec(s, snap)
        assert result["data_snapshot"] == snap

    def test_exact_boundary_passes(self):
        s = _spec(min_memories=50, min_days_active=14, min_sources=2)
        snap = _snapshot(memory_count=50, active_days=14, source_count=2)
        result = _evaluate_spec(s, snap)
        assert result["is_ready"] is True

    def test_one_below_boundary_fails(self):
        s = _spec(min_memories=50, min_days_active=14, min_sources=2)
        snap = _snapshot(memory_count=49, active_days=14, source_count=2)
        result = _evaluate_spec(s, snap)
        assert result["is_ready"] is False

    def test_zero_minimum_always_passes(self):
        s = _spec(min_memories=1, min_days_active=0, min_sources=1)
        snap = _snapshot(memory_count=1, active_days=0, source_count=1)
        result = _evaluate_spec(s, snap)
        assert result["is_ready"] is True


# ---------------------------------------------------------------------------
# TestBuildResponseItem
# ---------------------------------------------------------------------------

class TestBuildResponseItem:
    def test_active_when_enabled_and_ready(self):
        s = _spec()
        cfg = _config(readiness_state={"is_ready": True})
        item = _build_response_item("TestAgent", s, cfg, {})
        assert item["active"] is True
        assert item["is_ready"] is True
        assert item["enabled"] is True

    def test_not_active_when_disabled_even_if_ready(self):
        s = _spec()
        cfg = _config(enabled=False, readiness_state={"is_ready": True})
        item = _build_response_item("TestAgent", s, cfg, {})
        assert item["active"] is False
        assert item["is_ready"] is True
        assert item["enabled"] is False

    def test_not_active_when_not_ready(self):
        s = _spec()
        cfg = _config(enabled=True, readiness_state={"is_ready": False})
        item = _build_response_item("TestAgent", s, cfg, {})
        assert item["active"] is False

    def test_thresholds_populated(self):
        s = _spec(min_memories=50, min_days_active=14, min_sources=2)
        cfg = _config()
        item = _build_response_item("TestAgent", s, cfg, {})
        assert item["thresholds"]["min_memories"] == 50
        assert item["thresholds"]["min_days_active"] == 14
        assert item["thresholds"]["min_sources"] == 2

    def test_effective_params_in_response(self):
        s = _spec()
        cfg = _config()
        params = {"threshold": 0.85, "max": 5}
        item = _build_response_item("TestAgent", s, cfg, params)
        assert item["params"] == params

    def test_description_in_response(self):
        s = _spec()
        cfg = _config()
        item = _build_response_item("TestAgent", s, cfg, {})
        assert item["description"] == "Test agent"

    def test_empty_readiness_state_treated_as_not_ready(self):
        s = _spec()
        cfg = _config(readiness_state={})
        item = _build_response_item("TestAgent", s, cfg, {})
        assert item["is_ready"] is False


# ---------------------------------------------------------------------------
# TestBuildUnevalItem
# ---------------------------------------------------------------------------

class TestBuildUnevalItem:
    def test_unevaluated_item_not_ready(self):
        s = _spec(default_params={"k": 1})
        item = _build_uneval_item("FooAgent", s)
        assert item["is_ready"] is False
        assert item["active"] is False
        assert item["enabled"] is True
        assert item["evaluated_at"] is None

    def test_default_params_copied(self):
        s = _spec(default_params={"alpha": 0.5})
        item = _build_uneval_item("FooAgent", s)
        assert item["params"] == {"alpha": 0.5}


# ---------------------------------------------------------------------------
# TestFeatureReadinessService
# ---------------------------------------------------------------------------

class TestFeatureReadinessService:
    """Unit tests using mocked DB sessions."""

    def _make_db(self):
        db = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_get_all_triggers_evaluate_when_no_rows(self):
        service = FeatureReadinessService()
        # Simulate no configs in DB → should call evaluate_all
        db = self._make_db()
        # execute returns empty scalars
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        db.execute.return_value = mock_result

        with patch.object(service, "evaluate_all", new=AsyncMock(return_value=[])) as mock_eval:
            await service.get_all(org_id="org-x", db=db)
            mock_eval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_config_raises_on_unknown_agent(self):
        service = FeatureReadinessService()
        db = self._make_db()
        with pytest.raises(ValueError, match="Unknown agent"):
            await service.update_config(
                org_id="org-1",
                agent_name="NonExistentAgent",
                enabled=True,
                custom_params=None,
                notes=None,
                db=db,
            )

    @pytest.mark.asyncio
    async def test_update_config_merges_custom_params(self):
        service = FeatureReadinessService()
        db = self._make_db()
        db.commit = AsyncMock()

        existing_cfg = _config(
            agent_name="EntityResolutionAgent",
            custom_params={"similarity_threshold": 0.90},
        )

        with patch.object(
            service, "_get_or_create_config", new=AsyncMock(return_value=existing_cfg)
        ):
            await service.update_config(
                org_id="org-1",
                agent_name="EntityResolutionAgent",
                enabled=None,
                custom_params={"max_aliases": 10},
                notes="Tuned for large org",
                db=db,
            )

        assert existing_cfg.custom_params["similarity_threshold"] == 0.90
        assert existing_cfg.custom_params["max_aliases"] == 10
        assert existing_cfg.notes == "Tuned for large org"

    @pytest.mark.asyncio
    async def test_update_config_sets_enabled(self):
        service = FeatureReadinessService()
        db = self._make_db()
        db.commit = AsyncMock()

        existing_cfg = _config(agent_name="CredibilityAgent", enabled=True)

        with patch.object(
            service, "_get_or_create_config", new=AsyncMock(return_value=existing_cfg)
        ):
            await service.update_config(
                org_id="org-1",
                agent_name="CredibilityAgent",
                enabled=False,
                custom_params=None,
                notes=None,
                db=db,
            )

        assert existing_cfg.enabled is False

    @pytest.mark.asyncio
    async def test_update_config_creates_new_row_when_missing(self):
        service = FeatureReadinessService()
        db = self._make_db()
        db.commit = AsyncMock()

        new_cfg = _config(agent_name="CredibilityAgent", enabled=True)

        with patch.object(
            service, "_get_or_create_config", new=AsyncMock(return_value=new_cfg)
        ):
            result = await service.update_config(
                org_id="org-1",
                agent_name="CredibilityAgent",
                enabled=False,
                custom_params=None,
                notes=None,
                db=db,
            )

        assert result["enabled"] is False


# ---------------------------------------------------------------------------
# TestReadinessSpec (dataclass contract)
# ---------------------------------------------------------------------------

class TestReadinessSpec:
    def test_frozen_immutable(self):
        spec = ReadinessSpec(description="x")
        with pytest.raises((AttributeError, TypeError)):
            spec.min_memories = 99  # type: ignore[misc]

    def test_defaults(self):
        spec = ReadinessSpec(description="y")
        assert spec.min_memories == 1
        assert spec.min_days_active == 0
        assert spec.min_sources == 1
        assert spec.default_params == {}

    def test_custom_values(self):
        spec = ReadinessSpec(
            description="z",
            min_memories=200,
            min_days_active=60,
            min_sources=3,
            default_params={"k": "v"},
        )
        assert spec.min_memories == 200
        assert spec.min_days_active == 60
        assert spec.min_sources == 3
        assert spec.default_params == {"k": "v"}


# ---------------------------------------------------------------------------
# TestFetchDataSnapshot (integration-light)
# ---------------------------------------------------------------------------

class TestFetchDataSnapshot:
    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        db = AsyncMock()

        def _scalar(val):
            r = MagicMock()
            r.scalar_one.return_value = val
            return r

        db.execute.side_effect = [
            _scalar(42),   # memory_count
            _scalar(None), # first_memory_at (no memories)
            _scalar(2),    # source_count
        ]

        result = await _fetch_data_snapshot("org-1", db)
        assert result["memory_count"] == 42
        assert result["active_days"] == 0
        assert result["source_count"] == 2

    @pytest.mark.asyncio
    async def test_active_days_computed_from_first_memory(self):
        from datetime import timedelta

        db = AsyncMock()

        first_at = datetime.now(timezone.utc) - timedelta(days=45)

        def _scalar(val):
            r = MagicMock()
            r.scalar_one.return_value = val
            return r

        db.execute.side_effect = [
            _scalar(100),
            _scalar(first_at),
            _scalar(3),
        ]

        result = await _fetch_data_snapshot("org-1", db)
        assert result["active_days"] == 45

    @pytest.mark.asyncio
    async def test_none_count_treated_as_zero(self):
        db = AsyncMock()

        def _scalar(val):
            r = MagicMock()
            r.scalar_one.return_value = val
            return r

        db.execute.side_effect = [
            _scalar(None),
            _scalar(None),
            _scalar(None),
        ]

        result = await _fetch_data_snapshot("org-1", db)
        assert result["memory_count"] == 0
        assert result["active_days"] == 0
        assert result["source_count"] == 0


# ---------------------------------------------------------------------------
# TestEffectiveParams
# ---------------------------------------------------------------------------

class TestEffectiveParams:
    """Custom params override defaults but defaults survive for untouched keys."""

    def test_custom_params_override_defaults(self):
        spec = AGENT_SPECS["EntityResolutionAgent"]
        cfg = _config(
            agent_name="EntityResolutionAgent",
            custom_params={"similarity_threshold": 0.70},
            readiness_state={"is_ready": True},
        )
        effective = dict(spec.default_params)
        if cfg.custom_params:
            effective.update(cfg.custom_params)
        item = _build_response_item("EntityResolutionAgent", spec, cfg, effective)
        assert item["params"]["similarity_threshold"] == 0.70

    def test_unoverridden_defaults_remain(self):
        spec = AGENT_SPECS["EntityResolutionAgent"]
        cfg = _config(
            agent_name="EntityResolutionAgent",
            custom_params={"similarity_threshold": 0.70},
            readiness_state={"is_ready": True},
        )
        effective = dict(spec.default_params)
        if cfg.custom_params:
            effective.update(cfg.custom_params)
        item = _build_response_item("EntityResolutionAgent", spec, cfg, effective)
        # max_aliases was not overridden, should keep default
        assert "max_aliases" in item["params"]

    def test_no_custom_params_uses_defaults(self):
        spec = AGENT_SPECS["EntityResolutionAgent"]
        cfg = _config(
            agent_name="EntityResolutionAgent",
            custom_params=None,
            readiness_state={"is_ready": True},
        )
        effective = dict(spec.default_params)
        item = _build_response_item("EntityResolutionAgent", spec, cfg, effective)
        assert item["params"] == dict(spec.default_params)
