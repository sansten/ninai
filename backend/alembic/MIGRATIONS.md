# Ninai OSS - Database Migrations

## Overview

This directory contains Alembic database migrations for the Ninai OSS (Community) edition.

Migrations are applied in sequence order when running:
```bash
alembic upgrade head
```

## Migration Strategy

### Schema Versioning

- **Naming Convention**: `YYYY_MM_DD_HHNN_<description>.py`
- **Idempotent**: All migrations can be run multiple times safely
- **Backward Compatible**: Older schema versions are supported
- **Automatic Alembic Tracking**: Version history stored in `alembic_version` table

### Rollback

To rollback to a specific revision:
```bash
alembic downgrade <revision_id>
```

To rollback one step:
```bash
alembic downgrade -1
```

## OSS-Only Migrations

These migrations are part of the open-source Ninai core:

- Multi-tenant RLS policies and tenant isolation
- Memory storage and vector indexing
- Knowledge review workflows
- Agent runs and execution history
- Audit logging and compliance foundations
- API keys and webhooks
- Export and data portability
- Cognitive loops and goal tracking

## Enterprise Migrations

Enterprise-specific migrations are maintained in the separate `ninai-enterprise` repository and include:

- Meta-agent supervision and calibration tables
- Admin UI foundation and MFA models
- Advanced backup and recovery models
- Full-text search enhancements
- Event publishing infrastructure
- Organization-level feedback learning configuration

### Installing Enterprise Migrations

When the enterprise package is installed:
```bash
pip install -e ../ninai-enterprise
```

Enterprise migrations are applied automatically before app startup. See `../ninai-enterprise/alembic/versions/` for details.

## Development Workflow

### Creating New Migrations

```bash
# Auto-generate from model changes
alembic revision --autogenerate -m "describe your change"

# Manual migration
alembic revision -m "manual change description"
```

### Testing Migrations

```bash
# Create a test database
createdb ninai_test

# Apply migrations
export DATABASE_URL="postgresql://user:pass@localhost/ninai_test"
alembic upgrade head

# Test rollback
alembic downgrade -1
alembic upgrade head
```

## Production Deployment

### Pre-Deployment Checklist

1. ✅ Test migrations on staging database
2. ✅ Verify rollback procedure works
3. ✅ Confirm backups are in place
4. ✅ Review alembic_version history

### Deployment Steps

```bash
# 1. Backup production database
pg_dump $DATABASE_URL > backup-$(date +%s).sql

# 2. Apply migrations
alembic upgrade head

# 3. Verify application startup
# If issues, rollback:
alembic downgrade -1
```

## Adding Migrations to OSS

When adding new OSS functionality:

1. Create migration in `backend/alembic/versions/`
2. Update this file with migration description
3. Test both upgrade and downgrade paths
4. Ensure RLS policies are properly set for multi-tenancy
5. Add comments explaining security-critical changes

## See Also

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [README.md](../README.md) - Quick Start and Architecture
- `../ninai-enterprise/alembic/` - Enterprise migrations
