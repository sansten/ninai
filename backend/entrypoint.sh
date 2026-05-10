#!/bin/bash
# =============================================================================
# Ninai Backend Entrypoint Script
# =============================================================================
# This script ensures the database is properly initialized before starting
# the backend application.
# =============================================================================

set -e

ENTRYPOINT_MODE="${ENTRYPOINT_MODE:-api}"
CELERY_APP_MODULE="${CELERY_APP_MODULE:-app.core.celery_app:celery_app}"
CELERY_LOG_LEVEL="${CELERY_LOG_LEVEL:-INFO}"
CELERY_QUEUES="${CELERY_QUEUES:-q.agent_enrich,q.agent_topics,q.agent_patterns,q.agent_graph,q.agent_reasoning,q.agent_feedback,q.cognitive_loop,q.meta_agent,q.maintenance,q.webhooks}"

echo "============================================================================="
echo "Ninai Backend Starting..."
echo "============================================================================="
echo "Entrypoint mode: ${ENTRYPOINT_MODE}"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
python << END
import sys
import time
import psycopg2
from urllib.parse import quote_plus

max_retries = 30
retry_interval = 2

for i in range(max_retries):
    try:
        conn = psycopg2.connect(
            host="${POSTGRES_HOST}",
            port="${POSTGRES_PORT}",
            user="${POSTGRES_USER}",
            password="${POSTGRES_PASSWORD}",
            database="${POSTGRES_DB}"
        )
        conn.close()
        print("✓ PostgreSQL is ready!")
        sys.exit(0)
    except psycopg2.OperationalError as e:
        print(f"⏳ Waiting for PostgreSQL... (attempt {i+1}/{max_retries})")
        time.sleep(retry_interval)

print("✗ PostgreSQL is not available after {max_retries * retry_interval} seconds")
sys.exit(1)
END

# Initialize database tables and seed default data
echo "Initializing database..."
python << END
import asyncio
import sys
import json
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os

async def init_database():
    # Create async engine
    postgres_url = f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    engine = create_async_engine(postgres_url, echo=False)
    
    try:
        # Import all models to register them with Base
        from app.models import Base
        # Import all model modules to register them with Base
        import app.models.user
        import app.models.organization
        import app.models.memory
        import app.models.agent
        import app.models.team
        import app.models.admin
        
        print("Creating database tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✓ Database tables created")
        
        # Check if default data exists
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            # Check for default organization
            result = await session.execute(
                text("SELECT COUNT(*) FROM organizations WHERE slug = 'default'")
            )
            org_count = result.scalar()
            
            if org_count == 0:
                print("Seeding default data...")
                
                # Import password context for proper hash generation
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                admin_password_hash = pwd_context.hash("admin123")
                
                # Create default organization
                await session.execute(text("""
                    INSERT INTO organizations (id, name, slug, description, settings, is_active, created_at, updated_at)
                    VALUES (
                        '550e8400-e29b-41d4-a716-446655440000'::uuid,
                        'Default Organization',
                        'default',
                        'Default organization for initial setup',
                        '{}'::jsonb,
                        true,
                        now(),
                        now()
                    )
                """))
                
                # Create system roles
                await session.execute(text("""
                    INSERT INTO roles (id, name, display_name, description, permissions, is_system, is_default, created_at, updated_at)
                    VALUES 
                    (
                        '550e8400-e29b-41d4-a716-446655440010'::uuid,
                        'admin',
                        'Administrator',
                        'Full system administrator with all permissions',
                        ARRAY['*']::varchar[],
                        true,
                        false,
                        now(),
                        now()
                    ),
                    (
                        '550e8400-e29b-41d4-a716-446655440011'::uuid,
                        'member',
                        'Member',
                        'Standard member with read and write permissions',
                        ARRAY['read', 'write', 'memory:create', 'memory:read', 'memory:update', 'memory:delete']::varchar[],
                        true,
                        true,
                        now(),
                        now()
                    ),
                    (
                        '550e8400-e29b-41d4-a716-446655440012'::uuid,
                        'viewer',
                        'Viewer',
                        'Read-only access',
                        ARRAY['read', 'memory:read']::varchar[],
                        true,
                        false,
                        now(),
                        now()
                    )
                """))
                
                # Create default admin user with properly hashed password
                preferences_json = json.dumps({})
                
                await session.execute(
                    text("""
                        INSERT INTO users (id, email, full_name, hashed_password, is_active, is_superuser, is_admin, role, clearance_level, preferences, created_at, updated_at)
                        VALUES (
                            :id,
                            :email,
                            :full_name,
                            :password_hash,
                            :is_active,
                            :is_superuser,
                            :is_admin,
                            :role,
                            :clearance_level,
                            cast(:preferences as jsonb),
                            now(),
                            now()
                        )
                    """),
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440001",
                        "email": "admin@ninai.dev",
                        "full_name": "System Administrator",
                        "password_hash": admin_password_hash,
                        "is_active": True,
                        "is_superuser": True,
                        "is_admin": True,
                        "role": "admin",
                        "clearance_level": 4,
                        "preferences": preferences_json
                    }
                )
                
                # Assign admin user to organization with admin role
                await session.execute(text("""
                    INSERT INTO user_roles (id, user_id, organization_id, role_id, created_at, updated_at)
                    VALUES (
                        '550e8400-e29b-41d4-a716-446655440020'::uuid,
                        '550e8400-e29b-41d4-a716-446655440001'::uuid,
                        '550e8400-e29b-41d4-a716-446655440000'::uuid,
                        '550e8400-e29b-41d4-a716-446655440010'::uuid,
                        now(),
                        now()
                    )
                """))
                
                await session.commit()
                print("✓ Default data seeded successfully")
                print("")
                print("Default Login Credentials:")
                print("  Email: admin@ninai.dev")
                print("  Password: admin123")
                print("")
                print("IMPORTANT: Change the default password after first login!")
            else:
                print("✓ Default data already exists")
        
        await engine.dispose()
        return 0
        
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

sys.exit(asyncio.run(init_database()))
END

if [ $? -ne 0 ]; then
    echo "✗ Database initialization failed!"
    exit 1
fi

echo "============================================================================="
echo "✓ Backend initialization complete"
echo "============================================================================="
echo ""

# Start the application
case "${ENTRYPOINT_MODE}" in
  migrate)
    echo "Running Alembic migrations..."
    exec alembic upgrade head
    ;;
  worker)
    echo "Starting Celery worker..."
    exec celery -A "${CELERY_APP_MODULE}" worker --loglevel="${CELERY_LOG_LEVEL}" --queues="${CELERY_QUEUES}"
    ;;
  api|server|"")
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ;;
  *)
    echo "Unknown ENTRYPOINT_MODE: ${ENTRYPOINT_MODE}"
    exit 1
    ;;
esac
