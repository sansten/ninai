"""Services module initialization.

Keep package exports lazy so importing one service module does not force-load
all service dependencies (notably SQLAlchemy on Windows test runners).
"""

from importlib import import_module
from typing import Any

__all__ = [
    "PermissionChecker",
    "AuditService",
    "MemoryService",
    "SearchQueryParser",
    "parse_search_query",
    "KnowledgeSynthesisService",
    "MetricsEnhancementService",
    "get_metrics_service",
    "ReplicationService",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "PermissionChecker": ("app.services.permission_checker", "PermissionChecker"),
    "AuditService": ("app.services.audit_service", "AuditService"),
    "MemoryService": ("app.services.memory_service", "MemoryService"),
    "SearchQueryParser": ("app.services.search_query_parser", "SearchQueryParser"),
    "parse_search_query": ("app.services.search_query_parser", "parse_search_query"),
    "KnowledgeSynthesisService": (
        "app.services.knowledge_synthesis_service",
        "KnowledgeSynthesisService",
    ),
    "MetricsEnhancementService": (
        "app.services.metrics_enhancement_service",
        "MetricsEnhancementService",
    ),
    "get_metrics_service": ("app.services.metrics_enhancement_service", "get_metrics_service"),
    "ReplicationService": ("app.services.replication_service", "ReplicationService"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'app.services' has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value