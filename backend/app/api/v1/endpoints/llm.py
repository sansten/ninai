"""Minimal LLM helper endpoints.

These are intentionally small and admin-gated:
- The core product runs LLM calls server-side via agents/services.
- This endpoint exists to support SDK examples, debugging, and notebooks.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.schemas.llm import CompleteJsonRequest, CompleteJsonResponse
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


router = APIRouter()


@router.post("/complete-json", response_model=CompleteJsonResponse)
async def complete_json(
    body: CompleteJsonRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    # Hard-gate to org admins to avoid accidental model exposure.
    if bool(getattr(settings, "LLM_ADMIN_ONLY", True)) and not tenant.is_org_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    if bool(getattr(settings, "SANDBOX_LLM_ENABLED", False)):
        hint = body.schema_hint or {}

        # Minimal deterministic stubs for SDK/book examples.
        if "create_goal" in hint and "goal" in hint:
            data = {
                "create_goal": True,
                "goal": {
                    "title": "Draft goal",
                    "description": "Sandbox LLM stub output",
                    "goal_type": "task",
                    "visibility_scope": "team",
                    "priority": 2,
                },
                "nodes": [
                    {"temp_id": "n1", "node_type": "task", "title": "Collect inputs"},
                    {"temp_id": "n2", "node_type": "task", "title": "Produce draft"},
                ],
                "edges": [{"from_temp_id": "n1", "to_temp_id": "n2", "edge_type": "depends_on"}],
                "confidence": 0.75,
            }
        elif "links" in hint:
            data = {
                "links": [],
                "confidence": 0.5,
            }
        else:
            data = {}
    else:
        client = create_ollama_client(
            base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
            model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
            timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 10.0)),
            max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
            use_circuit_breaker=True,
        )

        data = await client.complete_json(prompt=body.prompt, schema_hint=body.schema_hint or {})

    return CompleteJsonResponse(data=data)
