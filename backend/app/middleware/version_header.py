"""Add X-API-Version response header and rewrite paths for version-pinned tenants."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


class APIVersionHeaderMiddleware(BaseHTTPMiddleware):
    """
    1. Injects X-API-Version into every response so clients can detect the served version.
    2. If an authenticated tenant has pinned_api_version set, transparently rewrites
       /api/vN/ -> /api/<pinned>/ so the tenant's old clients keep working after a
       breaking version ships without any client-side changes.

    The rewrite reads request.state.org which is populated by the tenant context
    dependency on authenticated requests. The middleware runs before routing, so
    the rewrite fires before FastAPI matches the route.
    """

    async def dispatch(self, request, call_next):
        # Version-pin path rewrite for authenticated tenants
        org = getattr(request.state, "org", None)
        if org:
            pinned = getattr(org, "pinned_api_version", None)
            if pinned:
                path = request.url.path
                # Only rewrite /api/vN/... paths
                parts = path.split("/")  # ['', 'api', 'v1', ...]
                if len(parts) >= 3 and parts[1] == "api" and parts[2].startswith("v"):
                    if parts[2] != pinned:
                        parts[2] = pinned
                        request.scope["path"] = "/".join(parts)

        response = await call_next(request)
        response.headers["X-API-Version"] = settings.CURRENT_API_VERSION
        return response
