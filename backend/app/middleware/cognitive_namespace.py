"""Cognitive namespace isolation middleware (Feature 15)."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.security import verify_token


_COGNITIVE_PREFIXES = (
    "/api/v1/cognitive",
    "/api/v1/sse",
    "/api/v1/ws",
)


def _extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    return token or None


async def cognitive_namespace_middleware(request: Request, call_next):
    path = request.url.path
    if not any(path.startswith(prefix) for prefix in _COGNITIVE_PREFIXES):
        return await call_next(request)

    header_org_id = (request.headers.get("X-Org-Id") or "").strip()
    token_org_id = None

    token = _extract_bearer_token(request)
    if token:
        token_data = verify_token(token)
        if token_data is not None:
            token_org_id = str(token_data.org_id or "").strip() or None

    if header_org_id and token_org_id and header_org_id != token_org_id:
        return JSONResponse(status_code=403, content={"detail": "Org namespace mismatch"})

    response = await call_next(request)

    resolved_org = token_org_id or (header_org_id if header_org_id else None)
    if resolved_org:
        response.headers["X-Ninai-Org-Id"] = resolved_org

    return response
