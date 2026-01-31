"""Tests for backup service and memory snapshots."""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_snapshot import MemorySnapshot, SnapshotStatus, SnapshotType
from app.services.backup_service import BackupService


@pytest.fixture
def test_org_id():
    return str(uuid4())


@pytest.fixture
def test_user_id():
    return str(uuid4())


@pytest.mark.asyncio
async def test_create_full_snapshot(db_session: AsyncSession, test_org_id, test_user_id):
    """Test creating a full snapshot."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Weekly Backup",
        snapshot_type=SnapshotType.FULL.value,
        retention_days=30,
        created_by_user_id=test_user_id,
    )

    assert snapshot.snapshot_type == SnapshotType.FULL.value
    assert snapshot.status == SnapshotStatus.PENDING.value
    assert snapshot.retention_days == 30
    assert snapshot.expires_at is not None


@pytest.mark.asyncio
async def test_create_incremental_snapshot(db_session: AsyncSession, test_org_id, test_user_id):
    """Test creating an incremental snapshot."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    # Create full snapshot first
    full = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Full Backup",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    # Create incremental
    incremental = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Incremental Backup",
        snapshot_type=SnapshotType.INCREMENTAL.value,
        parent_snapshot_id=full.id,
        created_by_user_id=test_user_id,
    )

    assert incremental.snapshot_type == SnapshotType.INCREMENTAL.value
    assert incremental.parent_snapshot_id == full.id


@pytest.mark.asyncio
async def test_incremental_requires_parent(db_session: AsyncSession, test_org_id, test_user_id):
    """Test incremental snapshot requires parent."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    with pytest.raises(ValueError, match="requires parent_snapshot_id"):
        await service.create_snapshot(
            organization_id=test_org_id,
            snapshot_name="Incremental",
            snapshot_type=SnapshotType.INCREMENTAL.value,
            created_by_user_id=test_user_id,
        )


@pytest.mark.asyncio
async def test_start_snapshot(db_session: AsyncSession, test_org_id, test_user_id):
    """Test starting a snapshot."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Test Snapshot",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    updated = await service.start_snapshot(snapshot_id=snapshot.id)

    assert updated.status == SnapshotStatus.IN_PROGRESS.value
    assert updated.started_at is not None


@pytest.mark.asyncio
async def test_complete_snapshot(db_session: AsyncSession, test_org_id, test_user_id):
    """Test completing a snapshot."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Test Snapshot",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    await service.start_snapshot(snapshot_id=snapshot.id)
    await db_session.commit()

    updated = await service.complete_snapshot(
        snapshot_id=snapshot.id,
        snapshot_size_bytes=1024 * 1024 * 100,  # 100 MB
        memory_count=1000,
        embedding_count=500,
        checksum="abc123def456",
    )

    assert updated.status == SnapshotStatus.COMPLETED.value
    assert updated.completed_at is not None
    assert updated.snapshot_size_bytes == 1024 * 1024 * 100
    assert updated.memory_count == 1000
    assert updated.embedding_count == 500
    assert updated.checksum == "abc123def456"
    assert updated.duration_seconds is not None


@pytest.mark.asyncio
async def test_fail_snapshot(db_session: AsyncSession, test_org_id, test_user_id):
    """Test failing a snapshot."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Test Snapshot",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    await service.start_snapshot(snapshot_id=snapshot.id)
    await db_session.commit()

    updated = await service.fail_snapshot(
        snapshot_id=snapshot.id,
        error_message="Storage I/O error",
    )

    assert updated.status == SnapshotStatus.FAILED.value
    assert updated.failed_at is not None
    assert updated.error_message == "Storage I/O error"


@pytest.mark.asyncio
async def test_verify_snapshot(db_session: AsyncSession, test_org_id, test_user_id):
    """Test verifying snapshot integrity."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Test Snapshot",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    await service.start_snapshot(snapshot_id=snapshot.id)
    await db_session.commit()

    await service.complete_snapshot(
        snapshot_id=snapshot.id,
        snapshot_size_bytes=1024 * 1024,
        memory_count=100,
        embedding_count=50,
        checksum="test_checksum",
    )
    await db_session.commit()

    verified, error = await service.verify_snapshot(snapshot_id=snapshot.id)

    assert verified is True
    assert error is None

    await db_session.refresh(snapshot)
    assert snapshot.verified is True
    assert snapshot.verified_at is not None


@pytest.mark.asyncio
async def test_verify_incomplete_snapshot(db_session: AsyncSession, test_org_id, test_user_id):
    """Test verification fails for incomplete snapshot."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Test Snapshot",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    verified, error = await service.verify_snapshot(snapshot_id=snapshot.id)

    assert verified is False
    assert "not completed" in error


@pytest.mark.asyncio
async def test_mark_replicated(db_session: AsyncSession, test_org_id, test_user_id):
    """Test marking snapshot as replicated."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Test Snapshot",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    updated = await service.mark_replicated(
        snapshot_id=snapshot.id,
        target_region="us-west-2",
        target_location="s3://backup-us-west-2/snapshot.tar.gz",
    )

    assert updated.replicated is True
    assert len(updated.replication_targets) == 1
    assert updated.replication_targets[0]["region"] == "us-west-2"
    assert updated.replication_targets[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_cleanup_expired_snapshots(db_session: AsyncSession, test_org_id, test_user_id):
    """Test cleaning up expired snapshots."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    # Create snapshot that expires immediately
    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Expired Snapshot",
        snapshot_type=SnapshotType.FULL.value,
        retention_days=0,  # Expires immediately
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    # Complete it
    await service.start_snapshot(snapshot_id=snapshot.id)
    await db_session.commit()

    await service.complete_snapshot(
        snapshot_id=snapshot.id,
        snapshot_size_bytes=1024,
        memory_count=10,
        embedding_count=5,
    )
    await db_session.commit()

    # Manually set expiry to past
    snapshot.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    await db_session.commit()

    # Clean up
    deleted_count = await service.cleanup_expired_snapshots(
        organization_id=test_org_id
    )

    assert deleted_count == 1

    await db_session.refresh(snapshot)
    assert snapshot.status == SnapshotStatus.EXPIRED.value


@pytest.mark.asyncio
async def test_list_snapshots(db_session: AsyncSession, test_org_id, test_user_id):
    """Test listing snapshots."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    # Create multiple snapshots
    for i in range(3):
        await service.create_snapshot(
            organization_id=test_org_id,
            snapshot_name=f"Snapshot {i}",
            snapshot_type=SnapshotType.FULL.value,
            created_by_user_id=test_user_id,
        )
        await db_session.commit()

    snapshots = await service.list_snapshots(organization_id=test_org_id)

    assert len(snapshots) == 3
    assert snapshots[0].snapshot_name == "Snapshot 2"  # Most recent first


@pytest.mark.asyncio
async def test_list_snapshots_with_filters(db_session: AsyncSession, test_org_id, test_user_id):
    """Test listing snapshots with filters."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    # Create full snapshot
    full = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Full",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    # Create incremental
    await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Incremental",
        snapshot_type=SnapshotType.INCREMENTAL.value,
        parent_snapshot_id=full.id,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    # Filter by type
    full_snapshots = await service.list_snapshots(
        organization_id=test_org_id,
        snapshot_type=SnapshotType.FULL.value,
    )

    assert len(full_snapshots) == 1
    assert full_snapshots[0].snapshot_type == SnapshotType.FULL.value


@pytest.mark.asyncio
async def test_get_latest_full_snapshot(db_session: AsyncSession, test_org_id, test_user_id):
    """Test getting latest full snapshot."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    # Create full snapshots
    s1 = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Full 1",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    await service.start_snapshot(snapshot_id=s1.id)
    await db_session.commit()

    await service.complete_snapshot(
        snapshot_id=s1.id,
        snapshot_size_bytes=1024,
        memory_count=10,
        embedding_count=5,
    )
    await db_session.commit()

    s2 = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Full 2",
        snapshot_type=SnapshotType.FULL.value,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    await service.start_snapshot(snapshot_id=s2.id)
    await db_session.commit()

    await service.complete_snapshot(
        snapshot_id=s2.id,
        snapshot_size_bytes=2048,
        memory_count=20,
        embedding_count=10,
    )
    await db_session.commit()

    latest = await service.get_latest_full_snapshot(organization_id=test_org_id)

    assert latest is not None
    assert latest.snapshot_name == "Full 2"


@pytest.mark.asyncio
async def test_get_snapshot_stats(db_session: AsyncSession, test_org_id, test_user_id):
    """Test getting snapshot statistics."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    # Create and complete snapshots
    for i in range(3):
        snapshot = await service.create_snapshot(
            organization_id=test_org_id,
            snapshot_name=f"Snapshot {i}",
            snapshot_type=SnapshotType.FULL.value,
            created_by_user_id=test_user_id,
        )
        await db_session.commit()

        await service.start_snapshot(snapshot_id=snapshot.id)
        await db_session.commit()

        await service.complete_snapshot(
            snapshot_id=snapshot.id,
            snapshot_size_bytes=1024 * (i + 1),
            memory_count=10,
            embedding_count=5,
        )
        await db_session.commit()

    stats = await service.get_snapshot_stats(organization_id=test_org_id)

    assert stats["total_count"] == 3
    assert stats["completed_count"] == 3
    assert stats["total_size_bytes"] == 1024 + 2048 + 3072


@pytest.mark.asyncio
async def test_snapshot_properties(db_session: AsyncSession, test_org_id, test_user_id):
    """Test snapshot model properties."""
    service = BackupService(db_session)
    db_session.info["auto_commit"] = False

    snapshot = await service.create_snapshot(
        organization_id=test_org_id,
        snapshot_name="Test",
        snapshot_type=SnapshotType.FULL.value,
        retention_days=30,
        created_by_user_id=test_user_id,
    )
    await db_session.commit()

    # Test is_complete
    assert snapshot.is_complete is False

    await service.start_snapshot(snapshot_id=snapshot.id)
    await db_session.commit()

    await service.complete_snapshot(
        snapshot_id=snapshot.id,
        snapshot_size_bytes=1024,
        memory_count=10,
        embedding_count=5,
    )
    await db_session.commit()

    await db_session.refresh(snapshot)
    assert snapshot.is_complete is True
    assert snapshot.duration_seconds is not None

    # Test is_expired
    assert snapshot.is_expired is False

    snapshot.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    await db_session.commit()
    await db_session.refresh(snapshot)

    assert snapshot.is_expired is True
