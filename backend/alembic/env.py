"""
Alembic Environment Configuration
=================================

This module configures Alembic to work with our SQLAlchemy setup.
Uses synchronous operations for migrations.
"""

import os

from logging.config import fileConfig

from sqlalchemy import pool, create_engine
from sqlalchemy.engine import Connection

from alembic import context

# Import your models' metadata
from app.core.config import settings
from app.models.base import Base

# Import all models to ensure they're registered
from app.models import (
    Organization,
    OrganizationHierarchy,
    User,
    Role,
    UserRole,
    Team,
    TeamMember,
    Agent,
    MemoryMetadata,
    MemorySharing,
    MemoryAttachment,
    AuditEvent,
    MemoryAccessLog,
    AgentRun,
    MemoryEdge,
    MemoryPromotionHistory,
    MemoryTopic,
    MemoryTopicMembership,
    MemoryPattern,
    MemoryPatternEvidence,
    MemoryLogseqExport,
    LogseqExportFile,
    OrgFeedbackLearningConfig,
    OrgLogseqExportConfig,
    CognitiveSession,
    CognitiveIteration,
    ToolCallLog,
    EvaluationReport,
)

# Alembic Config object
config = context.config

# Set SQLAlchemy URL from settings (use sync URL for Alembic)
#
# Tests may override the target DB without mutating app settings by setting:
#   ALEMBIC_DATABASE_URL_SYNC=postgresql://...
config.set_main_option(
    "sqlalchemy.url",
    os.environ.get("ALEMBIC_DATABASE_URL_SYNC") or settings.DATABASE_URL_SYNC,
)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    Creates a synchronous Engine and associates a connection with the context.
    """
    url = config.get_main_option("sqlalchemy.url")
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
