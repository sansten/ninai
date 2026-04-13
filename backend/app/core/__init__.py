"""Core module initialization.

Exports are lazy-loaded to keep package import side effects minimal.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "settings",
    "Base",
    "engine",
    "get_db",
    "get_tenant_session",
    "create_access_token",
    "create_refresh_token",
    "verify_password",
    "get_password_hash",
    "verify_token",
    "RedisClient",
    "get_redis",
    "QdrantService",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "settings": ("app.core.config", "settings"),
    "Base": ("app.models.base", "Base"),
    "engine": ("app.core.database", "engine"),
    "get_db": ("app.core.database", "get_db"),
    "get_tenant_session": ("app.core.database", "get_tenant_session"),
    "create_access_token": ("app.core.security", "create_access_token"),
    "create_refresh_token": ("app.core.security", "create_refresh_token"),
    "verify_password": ("app.core.security", "verify_password"),
    "get_password_hash": ("app.core.security", "get_password_hash"),
    "verify_token": ("app.core.security", "verify_token"),
    "RedisClient": ("app.core.redis", "RedisClient"),
    "get_redis": ("app.core.redis", "get_redis"),
    "QdrantService": ("app.core.qdrant", "QdrantService"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'app.core' has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
