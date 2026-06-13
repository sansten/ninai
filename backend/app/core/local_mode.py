"""Local-first/edge mode helpers.

Feature 23 requires an explicit way to disable federated/cloud behavior in
edge deployments while preserving local cognitive capabilities.
"""

from __future__ import annotations

from app.core.config import settings


def is_edge_deployment() -> bool:
    return str(getattr(settings, "DEPLOYMENT_MODE", "cloud") or "cloud").strip().lower() == "edge"


def is_local_first_mode() -> bool:
    return bool(getattr(settings, "LOCAL_FIRST_MODE", False)) or is_edge_deployment()


def external_federation_enabled() -> bool:
    """Whether cross-org/cross-network federation is allowed in this runtime."""
    return not is_local_first_mode()


def VLLM_BASE_URL_for_runtime() -> str:
    """Resolve the expected vLLM base URL in local-first mode.

    In edge mode we still respect explicit VLLM_BASE_URL; if unset we default
    to the in-cluster service name.
    """
    configured = str(getattr(settings, "VLLM_BASE_URL", "") or "").strip()
    if configured:
        return configured
    if is_edge_deployment():
        return "http://vllm:8000"
    return "http://localhost:11434"
