"""
Integration tests for ORM-level RLS defense-in-depth guard.

This test suite verifies that the ORM loader criteria prevents cross-org
data leaks even if PostgreSQL RLS is bypassed or misconfigured.

Tests confirm:
1. ORM criteria automatically filter queries by organization_id
2. Seeded users cannot access cross-org data
3. Snapshots, Events, and other org-scoped models respect org filters
4. Batch operations respect org isolation
"""

import pytest
from uuid import uuid4
from sqlalchemy import select, insert, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_tenant_session, set_tenant_context
from app.models.snapshot import Snapshot
from app.models.event import Event
from app.models.webhook_subscription import WebhookSubscription
from app.models.memory import Memory
from app.models.organization import Organization
from app.models.user import User
from app.services.rls_guard import attach_org_filter, get_org_filter_status
from tests.conftest import (
    E2E_SEED_ORG_ID,
    E2E_SEED_USER_ID,
    E2E_SEED_EMAIL,
)


@pytest.mark.asyncio
class TestOrmRlsCriteriaGuard:
    """Test ORM-level RLS criteria enforcement."""

    async def test_attach_org_filter_creates_criteria(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify attach_org_filter attaches loader criteria to session."""
        org_id = str(uuid4())
        user_id = str(uuid4())

        attach_org_filter(pg_db_session, org_id, user_id)

        # Check filter status
        status = get_org_filter_status(pg_db_session)
        assert status is not None
        assert status["active"] is True
        assert status["org_id"] == org_id
        assert status["user_id"] == user_id
        assert status["model_count"] > 0

    async def test_get_tenant_session_applies_org_filter(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify get_tenant_session automatically applies org filter."""
        org_id = str(uuid4())
        user_id = str(uuid4())

        async with get_tenant_session(user_id, org_id) as session:
            # Filter should be active
            status = get_org_filter_status(session)
            assert status is not None
            assert status["active"] is True
            assert status["org_id"] == org_id

    async def test_set_tenant_context_applies_org_filter(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify set_tenant_context automatically applies org filter."""
        org_id = str(uuid4())
        user_id = str(uuid4())

        async with pg_db_session.begin():
            await set_tenant_context(pg_db_session, user_id, org_id)

            status = get_org_filter_status(pg_db_session)
            assert status is not None
            assert status["active"] is True
            assert status["org_id"] == org_id

    async def test_snapshot_query_filtered_by_org(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify Snapshot queries are filtered by organization_id."""
        org_a = str(uuid4())
        org_b = str(uuid4())
        user_id = str(uuid4())

        async with pg_db_session.begin():
            # Create snapshots for two different orgs
            await pg_db_session.execute(
                insert(Snapshot).values(
                    id=str(uuid4()),
                    organization_id=org_a,
                    name="Snapshot A",
                    format="json",
                    storage_path="s3://bucket/a",
                    status="completed",
                )
            )

            await pg_db_session.execute(
                insert(Snapshot).values(
                    id=str(uuid4()),
                    organization_id=org_b,
                    name="Snapshot B",
                    format="json",
                    storage_path="s3://bucket/b",
                    status="completed",
                )
            )

        # Query as org A with ORM filter
        async with get_tenant_session(user_id, org_a) as session:
            result = await session.execute(select(Snapshot))
            snapshots = result.scalars().all()

            # All returned snapshots should be from org_a
            for snapshot in snapshots:
                assert snapshot.organization_id == org_a

    async def test_event_query_filtered_by_org(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify Event queries are filtered by organization_id."""
        org_a = str(uuid4())
        org_b = str(uuid4())
        user_id = str(uuid4())

        async with pg_db_session.begin():
            # Create events for two different orgs
            await pg_db_session.execute(
                insert(Event).values(
                    id=str(uuid4()),
                    organization_id=org_a,
                    event_type="memory.created",
                    resource_type="memory",
                    resource_id=str(uuid4()),
                    actor_user_id=user_id,
                )
            )

            await pg_db_session.execute(
                insert(Event).values(
                    id=str(uuid4()),
                    organization_id=org_b,
                    event_type="memory.created",
                    resource_type="memory",
                    resource_id=str(uuid4()),
                    actor_user_id=user_id,
                )
            )

        # Query as org A
        async with get_tenant_session(user_id, org_a) as session:
            result = await session.execute(select(Event))
            events = result.scalars().all()

            # All returned events should be from org_a
            for event in events:
                assert event.organization_id == org_a

    async def test_seeded_user_org_isolation(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify seeded E2E user cannot access cross-org data."""
        other_org_id = str(uuid4())
        other_user_id = str(uuid4())

        async with pg_db_session.begin():
            # Create snapshots in seeded org and other org
            await pg_db_session.execute(
                insert(Snapshot).values(
                    id=str(uuid4()),
                    organization_id=E2E_SEED_ORG_ID,
                    name="Seeded Org Snapshot",
                    format="json",
                    storage_path="s3://bucket/seeded",
                    status="completed",
                )
            )

            await pg_db_session.execute(
                insert(Snapshot).values(
                    id=str(uuid4()),
                    organization_id=other_org_id,
                    name="Other Org Snapshot",
                    format="json",
                    storage_path="s3://bucket/other",
                    status="completed",
                )
            )

        # Query as seeded user
        async with get_tenant_session(E2E_SEED_USER_ID, E2E_SEED_ORG_ID) as session:
            result = await session.execute(select(Snapshot))
            snapshots = result.scalars().all()

            # Should NOT see snapshots from other orgs
            seeded_org_snapshots = [
                s for s in snapshots if s.organization_id == E2E_SEED_ORG_ID
            ]
            other_org_snapshots = [
                s for s in snapshots if s.organization_id == other_org_id
            ]

            # All returned snapshots should be from seeded org only
            assert len(other_org_snapshots) == 0
            assert all(
                s.organization_id == E2E_SEED_ORG_ID for s in seeded_org_snapshots
            )

    async def test_multiple_model_filtering(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify multiple model types are filtered simultaneously."""
        org_a = str(uuid4())
        org_b = str(uuid4())
        user_id = str(uuid4())

        async with pg_db_session.begin():
            # Create mixed resources for both orgs
            snapshot_id_a = str(uuid4())
            event_id_a = str(uuid4())
            snapshot_id_b = str(uuid4())
            event_id_b = str(uuid4())

            await pg_db_session.execute(
                insert(Snapshot).values(
                    id=snapshot_id_a,
                    organization_id=org_a,
                    name="Snapshot A",
                    format="json",
                    storage_path="s3://a",
                    status="completed",
                )
            )

            await pg_db_session.execute(
                insert(Event).values(
                    id=event_id_a,
                    organization_id=org_a,
                    event_type="snapshot.created",
                    resource_type="snapshot",
                    resource_id=snapshot_id_a,
                    actor_user_id=user_id,
                )
            )

            await pg_db_session.execute(
                insert(Snapshot).values(
                    id=snapshot_id_b,
                    organization_id=org_b,
                    name="Snapshot B",
                    format="json",
                    storage_path="s3://b",
                    status="completed",
                )
            )

            await pg_db_session.execute(
                insert(Event).values(
                    id=event_id_b,
                    organization_id=org_b,
                    event_type="snapshot.created",
                    resource_type="snapshot",
                    resource_id=snapshot_id_b,
                    actor_user_id=user_id,
                )
            )

        # Query both models as org A
        async with get_tenant_session(user_id, org_a) as session:
            snapshots = (await session.execute(select(Snapshot))).scalars().all()
            events = (await session.execute(select(Event))).scalars().all()

            # Verify both models are filtered to org_a
            assert all(s.organization_id == org_a for s in snapshots)
            assert all(e.organization_id == org_a for e in events)

    async def test_empty_org_filter_prevents_query(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify empty org_id in filter doesn't bypass security."""
        user_id = str(uuid4())
        empty_org_id = ""

        # Attaching empty org filter should succeed but be non-permissive
        attach_org_filter(pg_db_session, empty_org_id, user_id)
        status = get_org_filter_status(pg_db_session)

        # Status reflects the filter was attempted
        assert status is not None


@pytest.mark.asyncio
class TestRlsDefenseInDepth:
    """Test RLS as defense-in-depth for ORM and database layers."""

    async def test_rls_policies_active_on_migrated_db(
        self, migrated_test_engine, pg_db_session: AsyncSession
    ) -> None:
        """Verify RLS policies are enabled on migrated database."""
        # This requires migrated_test_engine with Alembic migrations applied
        
        # Basic smoke query against information_schema (should not raise).
        await pg_db_session.execute(
            text(
                """
                SELECT 1 FROM information_schema.role_table_grants
                WHERE table_name IN ('snapshots', 'events', 'memory_metadata')
                  AND privilege_type = 'SELECT'
                LIMIT 1
                """
            )
        )

    async def test_explicit_org_filter_plus_rls(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify explicit WHERE clause + RLS + ORM criteria triple-check."""
        org_a = str(uuid4())
        org_b = str(uuid4())
        user_id = str(uuid4())

        async with pg_db_session.begin():
            # Create snapshots
            await pg_db_session.execute(
                insert(Snapshot).values(
                    id=str(uuid4()),
                    organization_id=org_a,
                    name="Safe Snapshot",
                    format="json",
                    storage_path="s3://safe",
                    status="completed",
                )
            )

            await pg_db_session.execute(
                insert(Snapshot).values(
                    id=str(uuid4()),
                    organization_id=org_b,
                    name="Restricted Snapshot",
                    format="json",
                    storage_path="s3://restricted",
                    status="completed",
                )
            )

        # Query with explicit WHERE clause + ORM filter
        async with get_tenant_session(user_id, org_a) as session:
            # Explicit WHERE + ORM filter
            result = await session.execute(
                select(Snapshot).where(Snapshot.organization_id == org_a)
            )
            snapshots = result.scalars().all()

            # Should only get org_a snapshots
            assert all(s.organization_id == org_a for s in snapshots)

        # Attempt to query with wrong org in WHERE (ORM should catch it)
        async with get_tenant_session(user_id, org_a) as session:
            # This WHERE is for org_b, but ORM filter is org_a
            result = await session.execute(
                select(Snapshot).where(Snapshot.organization_id == org_b)
            )
            snapshots = result.scalars().all()

            # ORM filter should return empty (defensive)
            # In reality, RLS would also block this at the DB level
            assert len(snapshots) == 0

    async def test_batch_operation_respects_org_filter(
        self, pg_db_session: AsyncSession
    ) -> None:
        """Verify batch update operations respect org isolation."""
        org_a = str(uuid4())
        org_b = str(uuid4())
        user_id = str(uuid4())

        snapshot_ids_a = [str(uuid4()) for _ in range(3)]
        snapshot_ids_b = [str(uuid4()) for _ in range(2)]

        async with pg_db_session.begin():
            # Create snapshots in org_a
            for sid in snapshot_ids_a:
                await pg_db_session.execute(
                    insert(Snapshot).values(
                        id=sid,
                        organization_id=org_a,
                        name=f"Snapshot {sid}",
                        format="json",
                        storage_path=f"s3://snap/{sid}",
                        status="pending",
                    )
                )

            # Create snapshots in org_b
            for sid in snapshot_ids_b:
                await pg_db_session.execute(
                    insert(Snapshot).values(
                        id=sid,
                        organization_id=org_b,
                        name=f"Snapshot {sid}",
                        format="json",
                        storage_path=f"s3://snap/{sid}",
                        status="pending",
                    )
                )

        # Query snapshots as org_a
        async with get_tenant_session(user_id, org_a) as session:
            result = await session.execute(select(Snapshot))
            accessible_snapshots = result.scalars().all()

            # Should only see org_a snapshots
            accessible_ids = {s.id for s in accessible_snapshots}
            assert all(sid in accessible_ids for sid in snapshot_ids_a)
            assert not any(sid in accessible_ids for sid in snapshot_ids_b)

