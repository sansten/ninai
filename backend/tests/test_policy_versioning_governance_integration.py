"""Integration tests for governance-wired policy rollout decisions."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from app.services.policy_versioning_service import PolicyVersioningService


@pytest.fixture
def service() -> PolicyVersioningService:
    db = AsyncMock()
    return PolicyVersioningService(db=db, organization_id=uuid.uuid4())


@pytest.mark.asyncio
async def test_canary_promote_path_calls_promote_to_production(service: PolicyVersioningService):
    policy_id = uuid.uuid4()
    service.promote_to_production = AsyncMock(return_value={"new_state": "production"})

    result = await service.evaluate_canary_governance(
        policy_id=policy_id,
        baseline_metrics={"win_rate": 0.50, "false_positive_rate": 0.20, "sample_count": 220},
        candidate_metrics={"win_rate": 0.56, "false_positive_rate": 0.15, "sample_count": 230},
        current_stage="canary",
    )

    assert result["decision"] == "promote"
    service.promote_to_production.assert_awaited_once()


@pytest.mark.asyncio
async def test_canary_critical_drift_calls_rollback(service: PolicyVersioningService):
    policy_id = uuid.uuid4()
    target_id = uuid.uuid4()
    service.rollback_policy = AsyncMock(return_value={"reason": "auto revert"})

    result = await service.evaluate_canary_governance(
        policy_id=policy_id,
        target_policy_id=target_id,
        baseline_metrics={"win_rate": 0.50, "false_positive_rate": 0.20, "sample_count": 220},
        candidate_metrics={
            "win_rate": 0.56,
            "false_positive_rate": 0.15,
            "sample_count": 230,
            "drift_severity": "critical",
        },
        current_stage="canary",
    )

    assert result["decision"] == "revert"
    assert result["auto_revert"] is True
    service.rollback_policy.assert_awaited_once()


@pytest.mark.asyncio
async def test_canary_hold_path_does_not_call_transition_methods(service: PolicyVersioningService):
    policy_id = uuid.uuid4()
    service.promote_to_production = AsyncMock()
    service.rollback_policy = AsyncMock()

    result = await service.evaluate_canary_governance(
        policy_id=policy_id,
        baseline_metrics={"win_rate": 0.50, "false_positive_rate": 0.20, "sample_count": 220},
        candidate_metrics={"win_rate": 0.51, "false_positive_rate": 0.195, "sample_count": 120},
        current_stage="canary",
    )

    assert result["decision"] == "hold"
    service.promote_to_production.assert_not_awaited()
    service.rollback_policy.assert_not_awaited()


@pytest.mark.asyncio
async def test_governance_payload_contains_audit_section(service: PolicyVersioningService):
    policy_id = uuid.uuid4()
    service.promote_to_production = AsyncMock(return_value={"new_state": "production"})

    result = await service.evaluate_canary_governance(
        policy_id=policy_id,
        baseline_metrics={"win_rate": 0.50, "false_positive_rate": 0.20, "sample_count": 220},
        candidate_metrics={"win_rate": 0.56, "false_positive_rate": 0.15, "sample_count": 230},
        current_stage="canary",
    )

    assert "governance_audit" in result
    audit = result["governance_audit"]
    assert audit["strategy_id"] == str(policy_id)
    assert "baseline" in audit and "candidate" in audit
