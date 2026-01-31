"""Pytest configuration.

This repo uses Pydantic Settings with environment-based DB config.
To keep unit tests import-safe (even when a local .env isn't present),
we set minimal dev defaults here before importing the app.
"""

import os


os.environ.setdefault("SECRET_KEY", "dev-test-secret")
os.environ.setdefault("APP_NAME", "Ninai")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("API_PREFIX", "/api/v1")
# pydantic-settings parses List[str] from env/.env as JSON; force a safe value
# to keep tests import-safe regardless of local developer .env contents.
os.environ["CORS_ORIGINS"] = "[]"
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
os.environ.setdefault("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")

# For Postgres-backed integration tests we want deterministic credentials.
# Allow explicit overrides via POSTGRES_TEST_*; otherwise default to docker-compose.yml.
_run_pg = os.environ.get("RUN_POSTGRES_TESTS", "").lower() in {"1", "true", "yes"}
_pg_host = os.environ.get("POSTGRES_TEST_HOST") or "localhost"
_pg_port = os.environ.get("POSTGRES_TEST_PORT") or "5432"
_pg_user = os.environ.get("POSTGRES_TEST_USER") or "ninai"
_pg_password = os.environ.get("POSTGRES_TEST_PASSWORD") or "ninai_dev_password"
_pg_db = os.environ.get("POSTGRES_TEST_BASE_DB") or "ninai"
_test_db_name_early = os.environ.get("POSTGRES_TEST_DB") or f"{_pg_db}_test"

if _run_pg:
    os.environ["POSTGRES_HOST"] = _pg_host
    os.environ["POSTGRES_PORT"] = _pg_port
    os.environ["POSTGRES_USER"] = _pg_user
    os.environ["POSTGRES_PASSWORD"] = _pg_password
    os.environ["POSTGRES_DB"] = _pg_db
else:
    os.environ.setdefault("POSTGRES_HOST", _pg_host)
    os.environ.setdefault("POSTGRES_PORT", _pg_port)
    os.environ.setdefault("POSTGRES_USER", _pg_user)
    os.environ.setdefault("POSTGRES_PASSWORD", _pg_password)
    os.environ.setdefault("POSTGRES_DB", _pg_db)

# Ensure the application-level engine/session helpers use the per-test DB.
# This must be set before importing app.core.config/app.core.database.
os.environ.setdefault(
    "TEST_DATABASE_URL",
    f"postgresql+asyncpg://{_pg_user}:{_pg_password}@{_pg_host}:{_pg_port}/{_test_db_name_early}",
)
os.environ.setdefault(
    "TEST_DATABASE_URL_SYNC",
    f"postgresql://{_pg_user}:{_pg_password}@{_pg_host}:{_pg_port}/{_test_db_name_early}",
)

import asyncio
from typing import AsyncGenerator
from uuid import uuid4
from unittest.mock import AsyncMock
from urllib.parse import urlparse
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from app.main import app
from app.core.config import settings
from app.core.database import get_db
from app.core.database import Base as CoreBase
from app.models.base import Base
from app.models.mfa import TOTPDevice, SMSDevice, WebAuthnDevice, MFAEnrollment  # Ensure MFA models are imported
from app.models.capability_token import CapabilityToken  # noqa: F401
from app.models.knowledge import Knowledge  # noqa: F401
from app.models.event import Event  # noqa: F401
from app.models.snapshot import Snapshot  # noqa: F401
from app.models.webhook_subscription import WebhookSubscription  # noqa: F401


# Test database URL (use a separate test database).
# Prefer explicit TEST_DATABASE_URL/POSTGRES_TEST_DB, otherwise use <POSTGRES_DB>_test.
_test_db_name = os.environ.get("POSTGRES_TEST_DB") or f"{(settings.POSTGRES_DB or 'ninai')}_test"
_explicit_test_url = os.environ.get("TEST_DATABASE_URL")
TEST_DATABASE_URL = _explicit_test_url or (
    f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{_test_db_name}"
)


def _to_sync_db_url(async_url: str) -> str:
    return async_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _recreate_test_database(test_db_url: str) -> None:
    """Drop+create the test DB and initialize required extensions.

    This keeps integration tests isolated while avoiding DROP TABLE ordering
    issues from cyclic FKs in the full metadata.
    """
    import asyncpg

    parsed = urlparse(test_db_url.replace("postgresql+asyncpg://", "postgresql://", 1))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    user = parsed.username or "ninai"
    password = parsed.password or ""
    test_db = (parsed.path or "/").lstrip("/")

    admin_db = os.environ.get("POSTGRES_ADMIN_DB") or "postgres"
    admin_dsn = f"postgresql://{user}:{password}@{host}:{port}/{admin_db}"
    fallback_admin_dsn = f"postgresql://{user}:{password}@{host}:{port}/{settings.POSTGRES_DB or 'ninai'}"

    conn = None
    try:
        try:
            conn = await asyncpg.connect(admin_dsn)
        except Exception:
            conn = await asyncpg.connect(fallback_admin_dsn)

        # Terminate active connections so DROP DATABASE works reliably.
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
            test_db,
        )
        await conn.execute(f'DROP DATABASE IF EXISTS "{test_db}"')
        await conn.execute(f'CREATE DATABASE "{test_db}"')
    finally:
        if conn is not None:
            await conn.close()

    # Create required extensions inside the test DB.
    test_dsn = f"postgresql://{user}:{password}@{host}:{port}/{test_db}"
    conn2 = await asyncpg.connect(test_dsn)
    try:
        await conn2.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        await conn2.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
        await conn2.execute('CREATE EXTENSION IF NOT EXISTS "ltree"')
    finally:
        await conn2.close()


async def _can_connect_to_postgres(url: str, timeout_seconds: float = 1.0) -> bool:
    try:
        parsed = urlparse(url.replace("postgresql+asyncpg://", "postgresql://", 1))
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432

        conn = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(conn, timeout=timeout_seconds)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


async def _require_postgres_or_skip(url: str) -> None:
    if await _can_connect_to_postgres(url):
        return

    message = (
        "Postgres is not reachable for integration tests. "
        "Start it (e.g. docker compose up -d postgres) or set DB env vars to a running instance. "
        "On Windows, ensure Docker Desktop is running if using docker compose."
    )

    if _run_pg:
        pytest.fail(f"RUN_POSTGRES_TESTS=1 but {message}")

    pytest.skip(message)


@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create a test database engine."""
    await _require_postgres_or_skip(TEST_DATABASE_URL)

    await _recreate_test_database(TEST_DATABASE_URL)
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
    
    async with engine.begin() as conn:
        # Some models still inherit from app.core.database.Base (CoreBase) while
        # most inherit from app.models.base.Base. Create tables for both to keep
        # metadata-based tests (db_session) consistent.
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(CoreBase.metadata.create_all)
    
    yield engine

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def migrated_test_engine():
    """Create a Postgres test database with Alembic migrations applied.

    This is the fixture to use when tests need RLS policies / SQL-side defaults
    that are only present after migrations.
    """

    await _require_postgres_or_skip(TEST_DATABASE_URL)

    await _recreate_test_database(TEST_DATABASE_URL)

    engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)

    async_url = engine.url.render_as_string(hide_password=False)
    sync_url = _to_sync_db_url(async_url)

    def _upgrade() -> None:
        from alembic import command
        from alembic.config import Config

        backend_dir = Path(__file__).resolve().parents[1]
        alembic_ini = backend_dir / "alembic.ini"
        alembic_dir = backend_dir / "alembic"

        cfg = Config(str(alembic_ini))
        cfg.set_main_option("script_location", str(alembic_dir))

        # Ensure migrations target the per-test database (RLS policies require migrations).
        # Set via config to avoid relying on env propagation across threads.
        cfg.set_main_option("sqlalchemy.url", sync_url)

        os.environ["ALEMBIC_DATABASE_URL_SYNC"] = sync_url
        # Use "heads" to support multiple branches that haven't been merged yet
        command.upgrade(cfg, "heads")

    await asyncio.to_thread(_upgrade)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def pg_client(migrated_test_engine) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client backed by a real Postgres DB (migrated).

    This is skipped automatically if Postgres is not reachable.
    """

    session_factory = async_sessionmaker(
        migrated_test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def db(migrated_test_engine):
    """Synchronous Session fixture for admin tests.

    Provides a scoped sync Session for seeding data and overrides FastAPI's
    get_db dependency to use the same migrated test database via that sync Session.
    """

    # Sync session for direct model writes in tests
    sync_url = _to_sync_db_url(migrated_test_engine.url.render_as_string(hide_password=False))
    sync_engine = create_engine(sync_url)
    SessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)
    sync_session = SessionLocal()

    # Async session override for API routes exercised via TestClient
    def override_get_db():
        yield sync_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        yield sync_session
    finally:
        sync_session.close()
        sync_engine.dispose()
        app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def pg_db_session(migrated_test_engine) -> AsyncGenerator[AsyncSession, None]:
    """AsyncSession bound to migrated_test_engine for direct DB seeding/assertions."""

    session_factory = async_sessionmaker(
        migrated_test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def db_session(
    test_engine, test_org_id: str, test_user_id: str
) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session.

    Seeds a minimal Organization + User so tests that insert rows with
    FK constraints (e.g., MemoryMetadata.organization_id/owner_id) don't
    fail on foreign key violations.
    """
    async_session = async_sessionmaker(
        test_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        from app.models.organization import Organization
        from app.models.user import User

        org_id = str(test_org_id)
        user_id = str(test_user_id)

        # Seed tenant + user for FK-referencing tests.
        if await session.get(Organization, org_id) is None:
            session.add(
                Organization(
                    id=org_id,
                    name="Test Organization",
                    slug=f"test-org-{org_id[:8]}",
                    is_active=True,
                )
            )

        if await session.get(User, user_id) is None:
            session.add(
                User(
                    id=user_id,
                    email=f"test-{user_id[:8]}@example.com",
                    hashed_password="$2b$12$placeholder",
                    full_name="Test User",
                    is_active=True,
                    role="org_admin",
                )
            )

        await session.commit()

        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client."""

    # Default test client is DB-less to keep unit tests runnable without Postgres.
    async def override_get_db():
        yield AsyncMock(spec=AsyncSession)
    
    app.dependency_overrides[get_db] = override_get_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
def test_org_id() -> str:
    """Generate a test organization ID."""
    return str(uuid4())


@pytest.fixture
def test_user_id() -> str:
    """Generate a test user ID."""
    return str(uuid4())


@pytest.fixture
def auth_headers(test_org_id: str, test_user_id: str) -> dict:
    """Generate auth headers for testing."""
    from app.core.security import create_access_token

    token = create_access_token(
        user_id=test_user_id,
        org_id=test_org_id,
        roles=["org_admin"],
    )
    return {
        "Authorization": f"Bearer {token}",
    }


# E2E Test Fixtures
# These are deterministic seeded users for reproducible E2E testing

E2E_SEED_ORG_ID = "00000000-0000-0000-0000-000000e2e001"
E2E_SEED_USER_ID = "00000000-0000-0000-0000-000000e2e002"
E2E_SEED_EMAIL = "e2e-test@example.com"


@pytest_asyncio.fixture
async def e2e_seeded_user(pg_db_session: AsyncSession) -> dict:
    """
    Create a deterministic seeded user for E2E testing.
    
    This fixture creates a user with a fixed org_id and user_id for reproducible
    E2E tests. All snapshots/memories created by this user will be tied to this org.
    
    Returns:
        dict with user_id, org_id, email, and auth token
    """
    from sqlalchemy import select, insert
    from app.models.user import User
    from app.models.organization import Organization
    from app.core.security import create_access_token
    
    async with pg_db_session.begin():
        # Check if org exists
        org_result = await pg_db_session.execute(
            select(Organization).where(Organization.id == E2E_SEED_ORG_ID)
        )
        org = org_result.scalars().first()
        
        if not org:
            # Create seeded organization
            org_stmt = insert(Organization).values(
                id=E2E_SEED_ORG_ID,
                name="E2E Test Organization",
                slug="e2e-test-org",
                is_active=True,
            )
            await pg_db_session.execute(org_stmt)
        
        # Check if user exists
        user_result = await pg_db_session.execute(
            select(User).where(User.id == E2E_SEED_USER_ID)
        )
        user = user_result.scalars().first()
        
        if not user:
            # Create seeded user
            user_stmt = insert(User).values(
                id=E2E_SEED_USER_ID,
                email=E2E_SEED_EMAIL,
                hashed_password="$2b$12$placeholder",  # Placeholder hash
                full_name="E2E Test User",
                is_active=True,
                role="org_admin",
            )
            await pg_db_session.execute(user_stmt)
        
        await pg_db_session.commit()
    
    # Generate token for this seeded user
    token = create_access_token(
        user_id=E2E_SEED_USER_ID,
        org_id=E2E_SEED_ORG_ID,
        roles=["org_admin"],
    )
    
    return {
        "user_id": E2E_SEED_USER_ID,
        "org_id": E2E_SEED_ORG_ID,
        "email": E2E_SEED_EMAIL,
        "token": token,
        "auth_header": {
            "Authorization": f"Bearer {token}",
        },
    }


@pytest.fixture
def e2e_auth_token(e2e_seeded_user: dict) -> str:
    """
    Get JWT token for seeded E2E user.
    
    Use this to set E2E_AUTH_TOKEN environment variable for Playwright tests.
    """
    return e2e_seeded_user["token"]


@pytest.fixture
def e2e_auth_headers(e2e_seeded_user: dict) -> dict:
    """
    Get auth headers for seeded E2E user (for pytest HTTP requests).
    """
    return e2e_seeded_user["auth_header"]


@pytest.fixture
def admin_token(e2e_seeded_user: dict) -> str:
    """
    Get JWT token for seeded admin/org_admin user.
    Alias for e2e_auth_token for backward compatibility.
    """
    return e2e_seeded_user["token"]


@pytest.fixture
def mfa_auth_headers(test_org_id: str, test_user_id: str) -> dict:
    """
    Get auth headers for MFA tests.
    """
    from app.core.security import create_access_token
    token = create_access_token(user_id=test_user_id, org_id=test_org_id, roles=["user"])
    return {"Authorization": f"Bearer {token}"}

