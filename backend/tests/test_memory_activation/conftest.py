"""
Test fixtures for memory_activation tests.

This module provides fixtures for testing memory activation features with
real database sessions.
"""

import pytest
import pytest_asyncio
from uuid import uuid4
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.organization import Organization
from app.models.user import User


@pytest.fixture
def test_org_id() -> str:
    """Provide a test organization ID."""
    return str(uuid4())


@pytest.fixture
def test_user_id() -> str:
    """Provide a test user ID."""
    return str(uuid4())


@pytest_asyncio.fixture
async def test_org(migrated_test_engine, test_org_id: str) -> Organization:
    """Create and seed a test organization."""
    async_session = async_sessionmaker(
        migrated_test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        org = Organization(
            id=test_org_id,
            name="Test Organization",
            slug=f"test-org-{test_org_id[:8]}",
            is_active=True,
        )
        session.add(org)
        await session.commit()
        return org


@pytest_asyncio.fixture
async def test_user(migrated_test_engine, test_user_id: str, test_org: Organization) -> User:
    """Create and seed a test user."""
    async_session = async_sessionmaker(
        migrated_test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        user = User(
            id=test_user_id,
            email=f"test-{test_user_id[:8]}@example.com",
            hashed_password="test_hash",
            full_name="Test User",
            is_active=True,
        )
        session.add(user)
        await session.commit()
        return user


@pytest_asyncio.fixture
async def test_memory(migrated_test_engine, test_org: Organization, test_user: User):
    """Create and seed a test memory."""
    import hashlib
    from app.models.memory import MemoryMetadata
    
    async_session = async_sessionmaker(
        migrated_test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        content = "Test memory content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        memory = MemoryMetadata(
            organization_id=str(test_org.id),
            owner_id=str(test_user.id),
            scope="personal",
            content_preview=content,
            content_hash=content_hash,
        )
        session.add(memory)
        await session.commit()
        await session.refresh(memory)
        return memory


@pytest_asyncio.fixture
async def other_org(migrated_test_engine) -> Organization:
    """Create and seed a different organization for cross-org tests."""
    org_id = str(uuid4())
    
    async_session = async_sessionmaker(
        migrated_test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        org = Organization(
            id=org_id,
            name="Other Organization",
            slug=f"other-org-{org_id[:8]}",
            is_active=True,
        )
        session.add(org)
        await session.commit()
        return org


@pytest_asyncio.fixture
async def async_session(migrated_test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an async session for direct database access in tests."""
    async_session = async_sessionmaker(
        migrated_test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()
