"""Backtest/significance/drift orchestration tests for PolicyVersioningService."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from app.services.policy_versioning_service import PolicyVersioningService


@pytest.fixture
def service() -> PolicyVersioningService:
    return PolicyVersioningService(db=AsyncMock(), organization_id=uuid.uuid4())


@pytest.mark.asyncio
async def test_historical_replay_backtest_returns_audit_and_significance(service: PolicyVersioningService):
    policy_id = uuid.uuid4()
    baseline = [{"success": False, "false_positive": True} for _ in range(140)] + [
        {"success": True, "false_positive": False} for _ in range(60)
    ]
    candidate = [{"success": False, "false_positive": True} for _ in range(88)] + [
        {"success": True, "false_positive": False} for _ in range(132)
    ]

    out = await service.run_historical_replay_backtest(
        policy_id=policy_id,
        baseline_records=baseline,
        candidate_records=candidate,
        current_stage="canary",
    )

    assert out["mode"] == "historical_replay"
    assert out["decision"] in {"promote", "demote", "hold", "revert"}
    assert "significance" in out
    assert "governance_audit" in out


def test_canary_significance_reports_significant_when_large_lift(service: PolicyVersioningService):
    stats = service.evaluate_canary_significance(
        stable_successes=100,
        stable_total=200,
        candidate_successes=130,
        candidate_total=200,
    )

    assert stats["significant"] is True
    assert stats["uplift"] > 0


def test_canary_significance_handles_insufficient_sample(service: PolicyVersioningService):
    stats = service.evaluate_canary_significance(
        stable_successes=0,
        stable_total=0,
        candidate_successes=10,
        candidate_total=20,
    )

    assert stats["significant"] is False
    assert stats["reason"] == "insufficient sample size"


@pytest.mark.asyncio
async def test_drift_alert_critical_triggers_auto_revert(service: PolicyVersioningService):
    policy_id = uuid.uuid4()
    target_id = uuid.uuid4()
    service.rollback_policy = AsyncMock(return_value={"new_policy_id": str(target_id)})

    out = await service.handle_drift_alert(
        policy_id=policy_id,
        drift_severity="critical",
        target_policy_id=target_id,
    )

    assert out["trigger"] == "drift_alert"
    assert out["decision"] == "revert"
    assert out["auto_revert"] is True
    service.rollback_policy.assert_awaited_once()


@pytest.mark.asyncio
async def test_drift_alert_low_does_not_force_revert(service: PolicyVersioningService):
    policy_id = uuid.uuid4()
    service.promote_to_production = AsyncMock(return_value={"new_state": "production"})
    service.rollback_policy = AsyncMock(return_value={})

    out = await service.handle_drift_alert(
        policy_id=policy_id,
        drift_severity="low",
    )

    assert out["trigger"] == "drift_alert"
    assert out["decision"] in {"hold", "promote", "demote"}
