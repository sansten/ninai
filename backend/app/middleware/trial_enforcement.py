"""Enforce trial limits and org lifecycle status on every authenticated request.

Fail-open: if org subscription state is not in request.state, the request
passes through unchanged. This keeps unauthenticated and public routes working.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings

logger = logging.getLogger(__name__)

# Paths that bypass enforcement regardless of org status
_EXEMPT_PREFIXES = (
    "/api/v1/auth",
    "/api/v1/signup",
    "/api/v1/health",
    "/api/v1/billing",
    "/api/v1/onboarding",
    "/docs",
    "/openapi.json",
    "/redoc",
)


class TrialEnforcementMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        sub = getattr(request.state, "org_subscription", None)
        if sub is None:
            # No subscription loaded (unauthenticated or public route) — pass through
            return await call_next(request)

        if sub.status == "deleted":
            return JSONResponse(status_code=404, content={"error": "account_not_found"})

        if sub.status == "suspended":
            return JSONResponse(
                status_code=402,
                content={
                    "error": "account_suspended",
                    "detail": "This account is suspended. Contact support.",
                    "support_url": f"{settings.FRONTEND_URL}/support",
                },
            )

        if sub.status == "trialing" and sub.trial_ends_at:
            if datetime.now(timezone.utc) > sub.trial_ends_at:
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "trial_expired",
                        "detail": "Your trial has ended. Please upgrade to continue.",
                        "upgrade_url": f"{settings.FRONTEND_URL}/billing",
                    },
                )

        return await call_next(request)
