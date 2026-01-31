"""Tests for admission controller and resource budget tracking.

Tests cover:
- Admission decisions based on token/storage quotas
- Token reservation and consumption
- Budget exhaustion and admission blocking
- SLO breach recording
- Throttling at high utilization
- Graceful degradation
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.resource_budget import ResourceBudget, BudgetPeriod
from app.services.admission_controller import AdmissionController


@pytest.fixture
def test_org_id():
    return str(uuid4())


@pytest.fixture
def test_user_id():
    return str(uuid4())


@pytest.mark.asyncio
async def test_get_or_create_budget(db_session: AsyncSession, test_org_id):
    """Test creating a new budget for current period."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    budget = await controller.get_or_create_budget(
        organization_id=test_org_id,
        period=BudgetPeriod.DAILY.value,
    )

    assert budget.organization_id == test_org_id
    assert budget.period == BudgetPeriod.DAILY.value
    assert budget.token_budget == 1000000
    assert budget.tokens_used == 0
    assert budget.admission_blocked is False


@pytest.mark.asyncio
async def test_admission_allowed_with_sufficient_quota(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that admission is allowed when quota is sufficient."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    decision = await controller.check_admission(
        organization_id=test_org_id,
        estimated_tokens=1000,
        actor_user_id=test_user_id,
    )

    assert decision.admitted is True
    assert decision.reason == "Admitted"
    assert decision.throttle_rate == 1.0


@pytest.mark.asyncio
async def test_admission_rejected_insufficient_tokens(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that admission is rejected when token quota is insufficient."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Create budget and exhaust most tokens
    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    budget.tokens_used = 999000  # Only 1000 tokens left

    decision = await controller.check_admission(
        organization_id=test_org_id,
        estimated_tokens=5000,  # Need more than available
        actor_user_id=test_user_id,
    )

    assert decision.admitted is False
    assert "Insufficient token budget" in decision.reason
    assert decision.retry_after_seconds is not None


@pytest.mark.asyncio
async def test_admission_rejected_insufficient_storage(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that admission is rejected when storage quota is insufficient."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Create budget and exhaust most storage
    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    budget.storage_used_mb = 9900  # Only 100MB left

    decision = await controller.check_admission(
        organization_id=test_org_id,
        estimated_storage_mb=500,  # Need more than available
        actor_user_id=test_user_id,
    )

    assert decision.admitted is False
    assert "Insufficient storage budget" in decision.reason


@pytest.mark.asyncio
async def test_admission_blocked_when_quota_exhausted(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that admission is blocked when quota is exhausted."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Create budget and set admission blocked
    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    budget.admission_blocked = True

    decision = await controller.check_admission(
        organization_id=test_org_id,
        estimated_tokens=100,
        actor_user_id=test_user_id,
    )

    assert decision.admitted is False
    assert "Quota exhausted" in decision.reason
    assert decision.retry_after_seconds is not None


@pytest.mark.asyncio
async def test_throttling_at_high_utilization(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that throttling is applied at high utilization."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Create budget at 85% utilization
    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    budget.tokens_used = 850000  # 85% of 1M

    decision = await controller.check_admission(
        organization_id=test_org_id,
        estimated_tokens=1000,
        actor_user_id=test_user_id,
    )

    assert decision.admitted is True
    assert decision.throttle_rate < 1.0  # Throttled
    assert decision.throttle_rate >= 0.6  # Should be 0.6 at 85%


@pytest.mark.asyncio
async def test_severe_throttling_at_very_high_utilization(db_session: AsyncSession, test_org_id, test_user_id):
    """Test severe throttling at very high utilization (95%+)."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Create budget at 96% utilization
    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    budget.tokens_used = 960000  # 96% of 1M

    decision = await controller.check_admission(
        organization_id=test_org_id,
        estimated_tokens=1000,
        actor_user_id=test_user_id,
    )

    assert decision.admitted is True
    assert decision.throttle_rate == 0.1  # Severely throttled


@pytest.mark.asyncio
async def test_reserve_tokens(db_session: AsyncSession, test_org_id):
    """Test token reservation."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Reserve tokens
    success = await controller.reserve_tokens(
        organization_id=test_org_id,
        tokens=5000,
    )

    assert success is True

    # Check budget updated
    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    assert budget.tokens_reserved == 5000
    assert budget.tokens_available == 1000000 - 5000


@pytest.mark.asyncio
async def test_reserve_tokens_insufficient_quota(db_session: AsyncSession, test_org_id):
    """Test that token reservation fails when quota insufficient."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Exhaust quota
    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    budget.tokens_used = 999000

    # Try to reserve more than available
    success = await controller.reserve_tokens(
        organization_id=test_org_id,
        tokens=5000,
    )

    assert success is False


@pytest.mark.asyncio
async def test_consume_resources(db_session: AsyncSession, test_org_id):
    """Test resource consumption tracking."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    await controller.consume_resources(
        organization_id=test_org_id,
        tokens_used=1000,
        storage_mb_used=50,
        latency_ms=2000,
    )

    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    assert budget.tokens_used == 1000
    assert budget.storage_used_mb == 50
    assert budget.latency_consumed_ms == 2000
    assert budget.requests_used == 1


@pytest.mark.asyncio
async def test_consume_resources_with_reservation(db_session: AsyncSession, test_org_id):
    """Test that reserved tokens are released when consumed."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Reserve tokens first
    await controller.reserve_tokens(organization_id=test_org_id, tokens=5000)

    # Consume less than reserved
    await controller.consume_resources(
        organization_id=test_org_id,
        tokens_used=3000,
        was_reserved=True,
    )

    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    assert budget.tokens_used == 3000
    assert budget.tokens_reserved == 2000  # 5000 - 3000


@pytest.mark.asyncio
async def test_auto_block_on_quota_exhaustion(db_session: AsyncSession, test_org_id):
    """Test that admission is auto-blocked when quota exhausted."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Consume resources to 96% (triggers auto-block at 95%)
    await controller.consume_resources(
        organization_id=test_org_id,
        tokens_used=960000,
    )

    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    assert budget.admission_blocked is True
    assert budget.degraded_mode is True


@pytest.mark.asyncio
async def test_record_slo_breach(db_session: AsyncSession, test_org_id):
    """Test SLO breach recording."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    await controller.record_slo_breach(
        organization_id=test_org_id,
        breach_details={"latency_ms": 35000, "sla_ms": 30000},
    )

    budget = await controller.get_or_create_budget(organization_id=test_org_id)
    assert budget.slo_breach_count == 1


@pytest.mark.asyncio
async def test_get_budget_summary(db_session: AsyncSession, test_org_id):
    """Test budget summary retrieval."""
    controller = AdmissionController(db_session)
    db_session.info["auto_commit"] = False

    # Consume some resources
    await controller.consume_resources(
        organization_id=test_org_id,
        tokens_used=250000,
        storage_mb_used=1000,
    )

    summary = await controller.get_budget_summary(organization_id=test_org_id)

    assert summary["token_budget"] == 1000000
    assert summary["tokens_used"] == 250000
    assert summary["token_utilization"] == 0.25
    assert summary["storage_used_mb"] == 1000
    assert "period_start" in summary
    assert "period_end" in summary
