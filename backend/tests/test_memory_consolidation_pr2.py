"""Test Suite for PR-2 Memory Consolidation (Sleep Cycle)."""

from unittest.mock import MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_arc import MemoryArc
from app.models.memory_consolidation_session import ConsolidationSession
from app.services.memory_consolidation_service import MemoryConsolidationService


class TestPR2Models:
    def test_consolidation_session_model_exists(self):
        assert ConsolidationSession is not None
        assert hasattr(ConsolidationSession, "__tablename__")
        assert ConsolidationSession.__tablename__ == "consolidation_sessions"

    def test_memory_arc_model_exists(self):
        assert MemoryArc is not None
        assert hasattr(MemoryArc, "__tablename__")
        assert MemoryArc.__tablename__ == "memory_arcs"


class TestPR2Service:
    def test_service_instantiation(self):
        db = MagicMock(spec=AsyncSession)
        service = MemoryConsolidationService(session=db, user_id="user_1", org_id="org_1")

        assert service is not None
        assert service.user_id == "user_1"
        assert service.org_id == "org_1"

    def test_service_has_required_methods(self):
        db = MagicMock(spec=AsyncSession)
        service = MemoryConsolidationService(session=db, user_id="user_1", org_id="org_1")

        required = [
            "start_consolidation_session",
            "discover_connections",
            "merge_redundant_facts",
            "apply_forgetting_curve",
            "compute_memory_trajectories",
            "dream_like_association",
            "finalize_session",
            "run_full_consolidation_cycle",
            "pin_memory",
            "unpin_memory",
        ]
        for method_name in required:
            assert hasattr(service, method_name), f"Missing method: {method_name}"
            assert callable(getattr(service, method_name)), f"{method_name} is not callable"

    def test_retention_score_decreases_over_time(self):
        early = MemoryConsolidationService.retention_score(days_since_access=1.0, stability_days=30.0)
        late = MemoryConsolidationService.retention_score(days_since_access=90.0, stability_days=30.0)

        assert early > late
        assert 0.0 < late < 1.0
        assert 0.0 < early <= 1.0

    def test_infer_trend_strengthening(self):
        trend = MemoryConsolidationService.infer_trend(
            [
                {"strength": 0.2},
                {"strength": 0.4},
                {"strength": 0.6},
            ]
        )
        assert trend == "strengthening"

    def test_infer_trend_weakening(self):
        trend = MemoryConsolidationService.infer_trend(
            [
                {"strength": 0.8},
                {"strength": 0.5},
                {"strength": 0.3},
            ]
        )
        assert trend == "weakening"


class TestPR2APIAndSchemas:
    def test_endpoint_import(self):
        from app.api.v1.endpoints import consolidation_sleep

        assert consolidation_sleep is not None
        assert hasattr(consolidation_sleep, "router")

    def test_schema_import(self):
        from app.schemas.consolidation_pr2 import (
            ConsolidationStartRequest,
            ConsolidationSessionResponse,
            ConsolidationSessionsResponse,
            MemoryArcResponse,
            PinMemoryResponse,
        )

        assert ConsolidationStartRequest is not None
        assert ConsolidationSessionResponse is not None
        assert ConsolidationSessionsResponse is not None
        assert MemoryArcResponse is not None
        assert PinMemoryResponse is not None


class TestPR2Migration:
    def test_migration_file_exists(self):
        import pathlib

        migration_path = (
            pathlib.Path(__file__).parent.parent
            / "alembic"
            / "versions"
            / "2026_03_03_001_add_memory_consolidation_sleep.py"
        )

        assert migration_path.exists(), f"Migration file not found at {migration_path}"
