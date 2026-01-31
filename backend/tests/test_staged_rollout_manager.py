"""Tests for staged rollout manager and policy versioning."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.policy_version import PolicyVersion, RolloutStatus
from app.services.staged_rollout_manager import StagedRolloutManager


@pytest.fixture
def test_org_id():
    return str(uuid4())


@pytest.mark.asyncio
async def test_create_policy_version(db_session: AsyncSession, test_org_id, test_user_id):
    """Test creating a new policy version."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="access_control",
        policy_type="access_control",
        policy_config={"rules": ["allow_admin", "deny_anonymous"]},
        description="Basic access control policy",
        created_by_user_id=test_user_id,
        change_notes="Initial version",
    )
    
    assert policy.version == 1
    assert policy.rollout_status == RolloutStatus.DRAFT.value
    assert policy.rollout_percentage == 0.0
    assert policy.policy_config == {"rules": ["allow_admin", "deny_anonymous"]}


@pytest.mark.asyncio
async def test_create_incremental_versions(db_session: AsyncSession, test_org_id):
    """Test version numbers increment automatically."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    v1 = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="rate_limit",
        policy_type="rate_limit",
        policy_config={"max_requests": 100},
    )
    await db_session.commit()
    
    v2 = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="rate_limit",
        policy_type="rate_limit",
        policy_config={"max_requests": 200},
    )
    
    assert v1.version == 1
    assert v2.version == 2


@pytest.mark.asyncio
async def test_deploy_to_canary(db_session: AsyncSession, test_org_id, test_user_id):
    """Test deploying to canary group."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="validation",
        policy_type="validation",
        policy_config={"strict": True},
    )
    await db_session.commit()
    
    updated = await manager.deploy_to_canary(
        policy_id=policy.id,
        canary_group_ids=["user1", "user2", "org1"],
        actor_user_id=test_user_id,
    )
    
    assert updated.rollout_status == RolloutStatus.CANARY.value
    assert updated.canary_group_ids == ["user1", "user2", "org1"]
    assert updated.rollout_percentage == 0.0


@pytest.mark.asyncio
async def test_deploy_to_canary_rejects_non_draft(db_session: AsyncSession, test_org_id):
    """Test canary deployment only works on DRAFT policies."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="safety",
        policy_type="safety",
        policy_config={"level": "high"},
    )
    await db_session.commit()
    
    # Deploy to canary first
    await manager.deploy_to_canary(
        policy_id=policy.id,
        canary_group_ids=["user1"],
    )
    await db_session.commit()
    
    # Try to deploy to canary again
    with pytest.raises(ValueError, match="Only DRAFT policies"):
        await manager.deploy_to_canary(
            policy_id=policy.id,
            canary_group_ids=["user2"],
        )


@pytest.mark.asyncio
async def test_promote_to_staged(db_session: AsyncSession, test_org_id, test_user_id):
    """Test promoting to staged rollout."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="routing",
        policy_type="routing",
        policy_config={"strategy": "round_robin"},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(
        policy_id=policy.id,
        canary_group_ids=["user1"],
    )
    await db_session.commit()
    
    updated = await manager.promote_to_staged(
        policy_id=policy.id,
        rollout_percentage=0.25,
        actor_user_id=test_user_id,
    )
    
    assert updated.rollout_status == RolloutStatus.STAGED.value
    assert updated.rollout_percentage == 0.25


@pytest.mark.asyncio
async def test_promote_to_staged_validates_percentage(db_session: AsyncSession, test_org_id):
    """Test staged rollout validates percentage range."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="test",
        policy_type="validation",
        policy_config={},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=policy.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        await manager.promote_to_staged(
            policy_id=policy.id,
            rollout_percentage=1.5,
        )


@pytest.mark.asyncio
async def test_activate_fully(db_session: AsyncSession, test_org_id, test_user_id):
    """Test full activation (100% rollout)."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="access",
        policy_type="access_control",
        policy_config={"allow_all": True},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=policy.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    await manager.promote_to_staged(policy_id=policy.id, rollout_percentage=0.5)
    await db_session.commit()
    
    updated = await manager.activate_fully(
        policy_id=policy.id,
        actor_user_id=test_user_id,
    )
    
    assert updated.rollout_status == RolloutStatus.ACTIVE.value
    assert updated.rollout_percentage == 1.0
    assert updated.activated_at is not None


@pytest.mark.asyncio
async def test_activate_supersedes_previous(db_session: AsyncSession, test_org_id):
    """Test activation supersedes previously active version."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    # Create and activate v1
    v1 = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="limit",
        policy_type="rate_limit",
        policy_config={"max": 100},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=v1.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    await manager.activate_fully(policy_id=v1.id)
    await db_session.commit()
    
    # Create and activate v2
    v2 = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="limit",
        policy_type="rate_limit",
        policy_config={"max": 200},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=v2.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    await manager.activate_fully(policy_id=v2.id)
    await db_session.commit()
    
    # Refresh v1 from DB
    await db_session.refresh(v1)
    
    assert v1.rollout_status == RolloutStatus.SUPERSEDED.value
    assert v1.superseded_at is not None
    assert v1.superseded_by_version == 2


@pytest.mark.asyncio
async def test_rollback_to_previous(db_session: AsyncSession, test_org_id, test_user_id):
    """Test rolling back to previous version."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    # Create and activate v1
    v1 = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="safety",
        policy_type="safety",
        policy_config={"level": "medium"},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=v1.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    await manager.activate_fully(policy_id=v1.id)
    await db_session.commit()
    
    # Create and activate v2
    v2 = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="safety",
        policy_type="safety",
        policy_config={"level": "high"},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=v2.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    await manager.activate_fully(policy_id=v2.id)
    await db_session.commit()
    
    # Rollback v2
    rolled_back = await manager.rollback(
        policy_id=v2.id,
        reason="High error rate detected",
        actor_user_id=test_user_id,
    )
    await db_session.commit()
    
    # Refresh v1 from DB
    await db_session.refresh(v1)
    
    assert rolled_back.rollout_status == RolloutStatus.ROLLED_BACK.value
    assert rolled_back.rollback_reason == "High error rate detected"
    assert rolled_back.rolled_back_to_version == 1
    
    # v1 should be reactivated
    assert v1.rollout_status == RolloutStatus.ACTIVE.value


@pytest.mark.asyncio
async def test_rollback_to_specific_version(db_session: AsyncSession, test_org_id):
    """Test rolling back to a specific version."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    # Create v1, v2, v3
    policies = []
    for i in range(3):
        policy = await manager.create_policy_version(
            organization_id=test_org_id,
            policy_name="test",
            policy_type="validation",
            policy_config={"version": i+1},
        )
        await db_session.commit()
        
        await manager.deploy_to_canary(policy_id=policy.id, canary_group_ids=["user1"])
        await db_session.commit()
        
        await manager.activate_fully(policy_id=policy.id)
        await db_session.commit()
        
        policies.append(policy)
    
    # Rollback v3 to v1 (skipping v2)
    await manager.rollback(
        policy_id=policies[2].id,
        reason="Rollback to v1",
        rollback_to_version=1,
    )
    await db_session.commit()
    
    # Refresh from DB
    await db_session.refresh(policies[0])
    await db_session.refresh(policies[2])
    
    assert policies[2].rolled_back_to_version == 1
    assert policies[0].rollout_status == RolloutStatus.ACTIVE.value


@pytest.mark.asyncio
async def test_record_evaluation_updates_metrics(db_session: AsyncSession, test_org_id):
    """Test recording evaluation results updates metrics."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="test",
        policy_type="validation",
        policy_config={},
    )
    await db_session.commit()
    
    # Record successes and failures
    for _ in range(90):
        await manager.record_evaluation(policy_id=policy.id, success=True)
    for _ in range(10):
        await manager.record_evaluation(policy_id=policy.id, success=False)
    
    await db_session.commit()
    await db_session.refresh(policy)
    
    assert policy.success_count == 90
    assert policy.failure_count == 10
    assert policy.error_rate == 0.1


@pytest.mark.asyncio
async def test_auto_rollback_on_high_error_rate(db_session: AsyncSession, test_org_id):
    """Test automatic rollback when error rate exceeds threshold."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    # Create and activate v1
    v1 = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="test",
        policy_type="validation",
        policy_config={"version": 1},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=v1.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    await manager.activate_fully(policy_id=v1.id)
    await db_session.commit()
    
    # Create and activate v2
    v2 = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="test",
        policy_type="validation",
        policy_config={"version": 2},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=v2.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    await manager.activate_fully(policy_id=v2.id)
    await db_session.commit()
    
    # Record high error rate (20% errors)
    for _ in range(80):
        await manager.record_evaluation(policy_id=v2.id, success=True)
    for _ in range(20):
        await manager.record_evaluation(policy_id=v2.id, success=False)
    
    await db_session.commit()
    
    # Check auto-rollback (threshold 10%)
    rolled_back = await manager.check_auto_rollback(
        policy_id=v2.id,
        error_rate_threshold=0.1,
        min_evaluations=100,
    )
    await db_session.commit()
    
    assert rolled_back is True
    
    # Refresh from DB
    await db_session.refresh(v1)
    await db_session.refresh(v2)
    
    assert v2.rollout_status == RolloutStatus.ROLLED_BACK.value
    assert "Auto-rollback" in v2.rollback_reason
    assert v1.rollout_status == RolloutStatus.ACTIVE.value


@pytest.mark.asyncio
async def test_auto_rollback_requires_min_evaluations(db_session: AsyncSession, test_org_id):
    """Test auto-rollback requires minimum evaluations."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="test",
        policy_type="validation",
        policy_config={},
    )
    await db_session.commit()
    
    # Record only 10 evaluations with 50% error rate
    for _ in range(5):
        await manager.record_evaluation(policy_id=policy.id, success=True)
    for _ in range(5):
        await manager.record_evaluation(policy_id=policy.id, success=False)
    
    await db_session.commit()
    
    # Should NOT rollback due to insufficient evaluations
    rolled_back = await manager.check_auto_rollback(
        policy_id=policy.id,
        error_rate_threshold=0.1,
        min_evaluations=100,
    )
    
    assert rolled_back is False


@pytest.mark.asyncio
async def test_get_active_policy(db_session: AsyncSession, test_org_id):
    """Test getting currently active policy."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    # Create and activate policy
    policy = await manager.create_policy_version(
        organization_id=test_org_id,
        policy_name="active_test",
        policy_type="validation",
        policy_config={"active": True},
    )
    await db_session.commit()
    
    await manager.deploy_to_canary(policy_id=policy.id, canary_group_ids=["user1"])
    await db_session.commit()
    
    await manager.activate_fully(policy_id=policy.id)
    await db_session.commit()
    
    # Retrieve active policy
    active = await manager.get_active_policy(
        organization_id=test_org_id,
        policy_name="active_test",
    )
    
    assert active is not None
    assert active.id == policy.id
    assert active.rollout_status == RolloutStatus.ACTIVE.value


@pytest.mark.asyncio
async def test_list_policy_versions(db_session: AsyncSession, test_org_id):
    """Test listing all versions of a policy."""
    manager = StagedRolloutManager(db_session)
    db_session.info["auto_commit"] = False
    
    # Create 3 versions
    for i in range(3):
        policy = await manager.create_policy_version(
            organization_id=test_org_id,
            policy_name="list_test",
            policy_type="validation",
            policy_config={"version": i+1},
        )
        await db_session.commit()
    
    versions = await manager.list_policy_versions(
        organization_id=test_org_id,
        policy_name="list_test",
    )
    
    assert len(versions) == 3
    assert versions[0].version == 3  # Descending order
    assert versions[1].version == 2
    assert versions[2].version == 1
