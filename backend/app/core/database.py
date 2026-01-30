"""
Database Configuration
======================

SQLAlchemy async database setup with connection pooling and session management.
Includes support for setting RLS session variables per-transaction.
Also includes ORM-level defense-in-depth filtering via loader criteria.
"""

from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from sqlalchemy.pool import NullPool

from app.core.config import settings


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for all models.
    
    All models should inherit from this base class.
    """
    pass


# Create async engine with connection pooling
# In tests on Windows, pooled asyncpg connections are a common source of
# "Event loop is closed" / Proactor transport errors during teardown.
_engine_kwargs = dict(
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

if settings.APP_ENV == "test":
    _engine_kwargs["poolclass"] = NullPool
else:
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20

engine: AsyncEngine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

# Session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def create_db_and_tables() -> None:
    """
    Create database tables if they don't exist.
    
    Note: In production, use Alembic migrations instead.
    This is primarily for development convenience.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Database session dependency.
    
    Creates a new database session for each request and ensures
    proper cleanup after the request is complete.
    
    Yields:
        AsyncSession: Database session for the request
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_tenant_session(
    user_id: str,
    org_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "",
) -> AsyncGenerator[AsyncSession, None]:
    """
    Create a database session with tenant context variables set.
    
    This context manager sets PostgreSQL session variables that are used
    by RLS policies to enforce row-level security. All variables are set
    using SET LOCAL so they only apply to the current transaction.
    
    ALSO attaches ORM-level loader criteria for defense-in-depth:
    even if Postgres RLS is bypassed, the ORM layer filters by org_id.
    
    Args:
        user_id: UUID of the current user
        org_id: UUID of the current organization
        roles: Comma-separated list of role names
        clearance_level: User's security clearance level (0-4)
        justification: Optional justification for sensitive access
    
    Yields:
        AsyncSession: Database session with tenant context set
    
    Example:
        async with get_tenant_session(user_id, org_id) as session:
            result = await session.execute(select(Memory))
            # RLS policies + ORM criteria automatically filter to user's org
    """
    # Lazy import to avoid circular dependency
    from app.services.rls_guard import attach_org_filter
    
    async with async_session_factory() as session:
        async with session.begin():
            # Set session variables for RLS policies
            # Using SET LOCAL ensures these only apply to this transaction
            # PostgreSQL SET doesn't support parameterized values
            def escape(val: str) -> str:
                return val.replace("'", "''") if val else ""
            
            await session.execute(text(f"SET LOCAL app.current_user_id = '{escape(user_id)}'"))
            await session.execute(text(f"SET LOCAL app.current_org_id = '{escape(org_id)}'"))
            await session.execute(text(f"SET LOCAL app.current_roles = '{escape(roles)}'"))
            await session.execute(text(f"SET LOCAL app.current_clearance_level = '{clearance_level}'"))
            await session.execute(text(f"SET LOCAL app.current_justification = '{escape(justification)}'"))
            
            # Attach ORM-level criteria for defense-in-depth
            attach_org_filter(session, org_id, user_id)
            
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


async def set_tenant_context(
    session: AsyncSession,
    user_id: str,
    org_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "",
) -> None:
    """
    Set tenant context variables on an existing session.
    
    Use this when you need to set context on a session that was
    obtained through dependency injection rather than the context manager.
    
    ALSO attaches ORM-level loader criteria for defense-in-depth.
    
    Args:
        session: Existing database session
        user_id: UUID of the current user
        org_id: UUID of the current organization
        roles: Comma-separated list of role names
        clearance_level: User's security clearance level
        justification: Optional justification for sensitive access
    
    Warning:
        Must be called within a transaction context!
        PostgreSQL SET doesn't support parameterized values, so we use
        string formatting with proper quoting.
    """
    # Lazy import to avoid circular dependency
    from app.services.rls_guard import attach_org_filter
    
    # Escape single quotes in values to prevent SQL injection
    def escape(val: str) -> str:
        return val.replace("'", "''") if val else ""
    
    await session.execute(text(f"SET LOCAL app.current_user_id = '{escape(user_id)}'"))
    await session.execute(text(f"SET LOCAL app.current_org_id = '{escape(org_id or '')}'"))
    await session.execute(text(f"SET LOCAL app.current_roles = '{escape(roles)}'"))
    await session.execute(text(f"SET LOCAL app.current_clearance_level = '{clearance_level}'"))
    await session.execute(text(f"SET LOCAL app.current_justification = '{escape(justification)}'"))
    
    # Attach ORM-level criteria for defense-in-depth
    attach_org_filter(session, org_id, user_id)
