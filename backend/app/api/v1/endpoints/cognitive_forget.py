"""Cognitive forgetting endpoint (Feature 24.11)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.cognitive_forget_service import CognitiveForgetService

router = APIRouter()


@router.post("/forget")
async def cognitive_forget(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Selective unlearning endpoint.

    Request:
      {"subject": "user@company.com", "domains": ["hr"], "reason": "gdpr_erasure"}
    """
    subject = str(payload.get("subject") or "").strip()
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="subject is required",
        )

    reason = str(payload.get("reason") or "gdpr_erasure").strip() or "gdpr_erasure"
    domains = list(payload.get("domains") or [])

    svc = CognitiveForgetService(db)
    cert = await svc.forget(
        organization_id=tenant.org_id,
        subject=subject,
        domains=domains,
        reason=reason,
        requested_by_user_id=tenant.user_id,
    )

    return {
        "certificate_id": cert.certificate_id,
        "organization_id": cert.organization_id,
        "subject": cert.subject,
        "reason": cert.reason,
        "domains": cert.domains,
        "erased_memory_count": cert.erased_memory_count,
        "invalidated_causal_edges": cert.invalidated_causal_edges,
        "recomputed_memory_count": cert.recomputed_memory_count,
        "knowledge_erased_event_emitted": cert.knowledge_erased_event_emitted,
        "generated_at": cert.generated_at,
    }
