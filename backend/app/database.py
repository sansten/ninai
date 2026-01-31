"""Compatibility shim for legacy imports.

Exports database helpers from app.core.database to maintain older import paths
(e.g., tests/test_admin.py, admin routes).
"""

from app.core.database import (
    Base,
    engine,
    async_session_factory,
    create_db_and_tables,
    get_db,
    get_tenant_session,
    set_tenant_context,
)

__all__ = [
    "Base",
    "engine",
    "async_session_factory",
    "create_db_and_tables",
    "get_db",
    "get_tenant_session",
    "set_tenant_context",
]
